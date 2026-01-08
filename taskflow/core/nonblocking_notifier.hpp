#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <memory>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <algorithm>
#include <numeric>
#include <cassert>
#include "../utility/os.hpp"

/**
@file nonblocking_notifier.hpp
@brief non-blocking notifier include file
*/

namespace tf {

/** @class NonblockingNotifier

@brief class to create a non-blocking notifier

A non-blocking notifier enables threads to wait for user-defined
predicates without blocking locks or protecting the predicate with
a mutex. Conceptually, it is similar to a condition variable, but the
wait predicate is evaluated optimistically and does not require
mutual exclusion.

A waiting thread follows this pattern:

@code{.cpp}
wid = this_waiter_id();
if (predicate) {
  return act();
}
notifier.prepare_wait(wid);    // enter the two-phase wait protocol
if (predicate) {
  notifier.cancel_wait(wid);
  return act();
}
notifier.commit_wait(&w);      // park (e.g., preempted by OS until notified)
@endcode

A notifying thread performs:

@code{.cpp}
wid = this_notifier_id;
predicate = true;
notifier.notify_one(wid);
@endcode

The `notify` operation is inexpensive when no threads are waiting.
The `prepare_wait` and `commit_wait` operations are more costly, but
they are only executed when the initial predicate check fails.
The flow diagram for notifier and waiter is shown below:

@dotfile images/nonblocking_notifier.dot


The synchronization algorithm relies on two shared variables: 
a user-defined predicate and an internal state variable. 
To avoid lost wake-ups, the protocol follows a two-phase @em update-then-check pattern. 
A waiting thread publishes its intent to wait by updating the state before rechecking the predicate. 
Conversely, a notifying thread updates the predicate before inspecting the state. 
This interaction is governed by a memory barrier of sequential consistency that guarantees at least one thread will observe the other's progress. 
Consequently, the waiter either detects the work and stays active, or the notifier detects the waiter and issues a wakeup.
It is impossible for both threads to miss each other's updates.
  
The state has the following layout, which consists of the following three parts:
  + `STACK_BITS` is a stack of committed waiters
  + `PREWAITER_BITS` is the count of waiters in the pre-waiting stage
  + `EPOCH_BITS` is the modification counter

@dotfile images/nonblocking_notifier_state_layout.dot

Reference: https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/ThreadPool/EventCount.h
*/
class NonblockingNotifier {

  friend class Executor;
  
  struct Waiter {
    alignas (TF_CACHELINE_SIZE) std::atomic<Waiter*> next;
    uint64_t epoch;

    enum : unsigned {
      kNotSignaled = 0,
      kWaiting,
      kSignaled,
    };
    std::atomic<unsigned> state {0};

    //mutable std::mutex mu;
    //std::condition_variable cv;
    //unsigned state;
  };

  public:
  
  // The state variable consists of the following three parts:
  // - low STACK_BITS is a stack of waiters committed wait.
  // - next PREWAITER_BITS is count of waiters in prewait state.
  // - next EPOCH_BITS is modification counter.
  // [ 32-bit epoch | 16-bit pre-waiter count | 16-bit pre-waiter stack]

  /// Number of bits used to encode the waiter stack index.
  static const uint64_t STACK_BITS = 16;

  /// Bit mask for extracting the waiter stack index.
  static const uint64_t STACK_MASK = (1ull << STACK_BITS) - 1;

  /// Number of bits used to encode the pre-waiter ticket.
  static const uint64_t PREWAITER_BITS = 16;

  /// Bit shift of the pre-waiter ticket field.
  static const uint64_t PREWAITER_SHIFT = 16;

  /// Bit mask for extracting the pre-waiter ticket field.
  static const uint64_t PREWAITER_MASK = ((1ull << PREWAITER_BITS) - 1) << PREWAITER_SHIFT;

  /// Increment value for advancing the pre-waiter ticket.
  static const uint64_t PREWAITER_INC = 1ull << PREWAITER_BITS;

  /// Number of bits used to encode the epoch counter.
  static const uint64_t EPOCH_BITS = 32;

  /// Bit shift of the epoch field.
  static const uint64_t EPOCH_SHIFT = 32;

  /// Bit mask for extracting the epoch field.
  static const uint64_t EPOCH_MASK = ((1ull << EPOCH_BITS) - 1) << EPOCH_SHIFT;

  /// Increment value for advancing the epoch counter.
  static const uint64_t EPOCH_INC = 1ull << EPOCH_SHIFT;

  /**
  @brief constructs a notifier with `N` waiters

  @param N number of waiters

  Constructs a notifier that supports up to `N` waiters. 
  The maximum allowable number of waiters can be acquired by calling `capacity()`,
  which is equal to 2<sup>STACK_BITS</sup>.
  */
  explicit NonblockingNotifier(size_t N) : _state(STACK_MASK), _waiters(N) {
    if(_waiters.size() >= ((1 << PREWAITER_BITS) - 1)) {
      TF_THROW("nonblocking waiter supports only up to ", (1<<PREWAITER_BITS)-1, " waiters");
    }
    //assert(_waiters.size() < (1 << PREWAITER_BITS) - 1);
    // Initialize epoch to something close to overflow to test overflow.
    //_state = STACK_MASK | (EPOCH_MASK - EPOCH_INC * _waiters.size() * 2);
  }
  
  /**
  @brief destructs the notifier
  */
  ~NonblockingNotifier() {
    // Ensure there are no waiters.
    assert((_state.load() & (STACK_MASK | PREWAITER_MASK)) == STACK_MASK);
  }
  
  /**
  @brief returns the number of committed waiters
  
  @return the number of committed waiters at the time of the call.
  
  A committed waiter is a thread that has completed the pre-waiting stage
  and is fully registered in the waiting set via commit_wait().
  */
  size_t num_waiters() const {
    size_t n = 0;
    for(auto& w : _waiters) {
      n += (w.state.load(std::memory_order_relaxed) == Waiter::kWaiting);
      //std::scoped_lock lock(w.mu);
      //n += (w.state == Waiter::kWaiting);
    }
    return n;
  }
  
  /**
  @brief returns the maximum number of waiters supported by this notifier
  
  The maximum number of waiters supported by this non-blocking notifier is
  equal to 2<sup>STACK_BITS</sup>.
  */
  size_t capacity() const {
    return 1 << STACK_BITS;
  }

  /**
  @brief prepares the calling thread to enter the waiting set
  
  @param wid identifier of the calling thread in the range of `[0, N)`, 
         where `N` represents the number of waiters used to construct this notifier
  
  This function places the thread into the pre-waiting stage. After calling
  `prepare_wait()`, the thread must re-check the wait predicate and then
  complete the protocol by calling either `commit_wait()` or `cancel_wait()`
  with the same waiter identifier.
  
  A thread in the pre-waiting stage is not yet considered a committed waiter,
  and its waiting status is considered incomplete.
  Failing to follow `prepare_wait()` with exactly one call to
  `commit_wait()` or `cancel_wait()` results in undefined behavior.
  */
  void prepare_wait(size_t wid) {
    _waiters[wid].epoch = _state.fetch_add(PREWAITER_INC, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
  }

  /**
  @brief commits a previously prepared wait operation
  
  @param wid identifier of the calling thread in the range of `[0, N)`, 
         where `N` represents the number of waiters used to construct this notifier
  
  This function completes the waiting protocol for a thread that has
  previously called `prepare_wait()`. Upon successful completion, the
  thread becomes a committed waiter and will park until being notified.
  
  The thread must have re-checked the wait predicate before calling
  `commit_wait()`. Once committed, the thread may be awakened by
  `notify_one()`, `notify_n()`, or `notify_all()`.
  
  Each call to `prepare_wait()` must be followed by exactly one call
  to either `commit_wait()` or `cancel_wait()` using the same
  thread identifier.
  */
  void commit_wait(size_t wid) {

    auto w = &_waiters[wid];

    w->state.store(Waiter::kNotSignaled, std::memory_order_relaxed);
    
    /*
    Epoch and ticket semantics.
    
      `sepoch` = _state & EPOCH_MASK
      `wepoch` = w->epoch & EPOCH_MASK
      `ticket` = w->epoch & PREWAITER_MASK
    
    Each waiter entering the pre-waiting stage is assigned a monotonically
    increasing ticket that determines the processing order (e.g.,
    cancel_wait, commit_wait, notify). Ticket 0 is processed first, followed
    by ticket 1, and so on.
    
    The global epoch `sepoch` is incremented whenever a request is fulfilled.
    Therefore, the difference `sepoch - wepoch` indicates which ticket is
    currently ready to be handled:
    
      - `sepoch - wepoch == ticket` : this waiter's turn
      - `sepoch - wepoch >  ticket` : this waiter's ticket has expired
      - `sepoch - wepoch <  ticket` : this waiter's ticket has not yet reached
    
    Unsigned wraparound does not affect correctness. All epoch arithmetic is
    performed using unsigned integers, which obey modulo-2^N arithmetic.
    Converting the unsigned difference to a signed value yields the correct
    result as long as the true difference lies within the signed range.
    
    In general:
      - Unsigned range: [0, 2^N − 1]
      - Signed range  : [−2^(N−1), 2^(N−1) − 1]
    
    When overflow occurs, unsigned subtraction computes:
    
      (sepoch − wepoch) mod 2^N
    
    If the true value of `sepoch − wepoch` is within the signed range
    [−2^(N−1), 2^(N−1) − 1], reinterpreting this result as a signed integer
    produces the correct mathematical difference.
    
    Example (3-bit arithmetic):
    
      a  b | true a−b | unsigned (bin / dec) | signed (dec)
      ----------------------------------------------------
      1  0 |   1      | 001 / 1              | +1
      1  1 |   0      | 000 / 0              |  0
      1  2 |  -1      | 111 / 7              | -1
      1  3 |  -2      | 110 / 6              | -2
      1  4 |  -3      | 101 / 5              | -3
      1  5 |  -4      | 100 / 4              | -4
      1  6 |  -5      | 011 / 3              | +3 (wrap around)
      1  7 |  -6      | 010 / 2              | +2 (wrap around)
    
    Signed interpretation is correct only when the true difference lies
    within [−4, +3].
    
    In this implementation, `sepoch − wepoch` is guaranteed not to exceed
    2^16 in magnitude, which is far smaller than 2^(EPOCH_BITS − 1).
    Consequently, the expression:
    
      int64_t((state & EPOCH_MASK) - epoch)
    
    remains correct even if `sepoch` and `wepoch` individually overflow.
    */
    uint64_t epoch =
        (w->epoch & EPOCH_MASK) +
        (((w->epoch & PREWAITER_MASK) >> PREWAITER_SHIFT) << EPOCH_SHIFT);
    uint64_t state = _state.load(std::memory_order_seq_cst);
    for (;;) {
      if (int64_t((state & EPOCH_MASK) - epoch) < 0) {
        // The preceding waiter has not decided on its fate. Wait until it
        // calls either cancel_wait or commit_wait, or is notified.
        std::this_thread::yield();
        state = _state.load(std::memory_order_seq_cst);
        continue;
      }
      // We've already been notified.
      if (int64_t((state & EPOCH_MASK) - epoch) > 0) {
        return;
      }
      // Remove this thread from prewait counter and add it to the waiter stack.
      assert((state & PREWAITER_MASK) != 0);
      uint64_t newstate = state - PREWAITER_INC + EPOCH_INC;
      newstate = (newstate & ~STACK_MASK) | wid;

      // stack is empty -> this waiter is at the top of the stack, pointing to nothing
      if ((state & STACK_MASK) == STACK_MASK) {
        w->next.store(nullptr, std::memory_order_relaxed);
      }
      // stack is non-empty -> this waiter is at the top of the stack, pointing to the origin top
      else {
        w->next.store(&_waiters[state & STACK_MASK], std::memory_order_relaxed);
      }
      if (_state.compare_exchange_weak(state, newstate, std::memory_order_release)) {
        break;
      }
    }
    _park(w);
  }

  /**
  @brief cancels a previously prepared wait operation
  
  @param wid identifier of the calling thread in the range of `[0, N)`, 
         where `N` represents the number of waiters used to construct this notifier
  
  This function aborts the waiting protocol for a thread that has
  previously called `prepare_wait()`. After cancellation, the thread
  does not become a committed waiter and will return to user-side control.
  
  `cancel_wait()` must be called after the wait predicate has been
  re-checked and found to be false. This allows a thread to safely
  abandon waiting without blocking or being notified.
  
  Each call to `prepare_wait()` must be followed by exactly one call
  to either `commit_wait()` or `cancel_wait()` using the same
  thread identifier.
  */
  void cancel_wait(size_t wid) {
    uint64_t epoch =
      (_waiters[wid].epoch & EPOCH_MASK) +
      (((_waiters[wid].epoch & PREWAITER_MASK) >> PREWAITER_SHIFT) << EPOCH_SHIFT);
    uint64_t state = _state.load(std::memory_order_relaxed);
    for (;;) {
      if (int64_t((state & EPOCH_MASK) - epoch) < 0) {
        // The preceding waiter has not decided on its fate. Wait until it
        // calls either cancel_wait or commit_wait, or is notified.
        std::this_thread::yield();
        state = _state.load(std::memory_order_relaxed);
        continue;
      }
      // We've already been notified.
      if (int64_t((state & EPOCH_MASK) - epoch) > 0) {
        return;
      }
      // Remove this thread from prewait counter.
      assert((state & PREWAITER_MASK) != 0);
      if (_state.compare_exchange_weak(state, state - PREWAITER_INC + EPOCH_INC,
                                       std::memory_order_relaxed)) {
        return;
      }
    }
  }
  
  /**
  @brief notifies one waiter from the waiting set

  Wakes up one waiter from the waiting set, including those in the pre-waiting stage.

  The function is cheap when no threads are waiting.
  */
  void notify_one() {
    _notify(false);
  }
  
  /**
  @brief notifies all waiter from the waiting set

  Wakes up all waiters from the waiting set, including those in the pre-waiting stage.
  
  The function is cheap when no threads are waiting.
  */
  void notify_all() {
    _notify(true);
  }
  
  /**
  @brief notifies up to `n` waiters from the waiting set
  
  @param n maximum number of waiters to notify
  
  Wakes up at most @p n waiters from the waiting set.
  If @p n is greater than or equal to the maximum number of waiters in this notifier,
  this function behaves identically to `notify_all()`.
  
  The function is cheap when no threads are waiting.
  */
  void notify_n(size_t n) {
    if(n >= _waiters.size()) {
      _notify(true);
    }
    else {
      for(size_t k=0; k<n; ++k) {
        _notify(false);
      }
    }
  }
  
  /**
  @brief returns the number of waiters supported by this notifier

  @return the number of waiters supported by this notifier

  The size of a notifier is equal to the number used to construct that notifier.
  */
  size_t size() const {
    return _waiters.size();
  }

 private:

  std::atomic<uint64_t> _state;
  std::vector<Waiter> _waiters;
  
  // only this waiter can park itself, with the following two possible paths:
  // 1. kNotSignaled (this) -> in-stack -> kWaiting (this) -> wait
  // 2. kNotSignaled (this) -> in-stack -> kSignaled -> unwait
  void _park(Waiter* w) {
    unsigned target = Waiter::kNotSignaled;
    if(w->state.compare_exchange_strong(target, Waiter::kWaiting, std::memory_order_relaxed
                                                                , std::memory_order_relaxed)) {
      w->state.wait(Waiter::kWaiting, std::memory_order_relaxed);
    }
    //std::unique_lock<std::mutex> lock(w->mu);
    //while (w->state != Waiter::kSignaled) {
    //  w->state = Waiter::kWaiting;
    //  w->cv.wait(lock);
    //}
  }
  
  // others can unpark
  void _unpark(Waiter* waiters) {
    Waiter* next = nullptr;
    for (Waiter* w = waiters; w; w = next) {
      next = w->next.load(std::memory_order_relaxed);

      // We only notify if the other is waiting - this is why we use tri-state
      // variable instead of binary-state variable (i.e., atomic_flag)
      // Performance is about 0.1% faster
      if(w->state.exchange(Waiter::kSignaled, std::memory_order_relaxed) == Waiter::kWaiting) {
        w->state.notify_one();
      }

      //unsigned state;
      //{
      //  std::unique_lock<std::mutex> lock(w->mu);
      //  state = w->state;
      //  w->state = Waiter::kSignaled;
      //}
      //// Avoid notifying if it wasn't waiting.
      //if (state == Waiter::kWaiting) w->cv.notify_one();
    }
  }
  
  // notify wakes one or all waiting threads.
  // Must be called after changing the associated wait predicate.
  void _notify(bool all) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    uint64_t state = _state.load(std::memory_order_acquire);
    for (;;) {
      // Easy case: no waiters.
      if ((state & STACK_MASK) == STACK_MASK && (state & PREWAITER_MASK) == 0) {
        return;
      }
      uint64_t num_prewaiters = (state & PREWAITER_MASK) >> PREWAITER_SHIFT;
      uint64_t newstate;
      if (all) {
        // Reset prewait counter and empty wait list.
        newstate = (state & EPOCH_MASK) + (EPOCH_INC * num_prewaiters) + STACK_MASK;
      } else if (num_prewaiters) {
        // There is a thread in pre-wait state, unblock it.
        newstate = state + EPOCH_INC - PREWAITER_INC;
      } else {
        // Pop a waiter from list and unpark it.
        Waiter* w = &_waiters[state & STACK_MASK];
        Waiter* wnext = w->next.load(std::memory_order_relaxed);
        uint64_t next = STACK_MASK;
        //if (wnext != nullptr) next = wnext - &_waiters[0];
        if (wnext != nullptr) {
          next = static_cast<uint64_t>(wnext - &_waiters[0]);
        }
        // Note: we don't add EPOCH_INC here. ABA problem on the lock-free stack
        // can't happen because a waiter is re-pushed onto the stack only after
        // it was in the pre-wait state which inevitably leads to epoch increment.
        newstate = (state & EPOCH_MASK) + next;
      }
      if (_state.compare_exchange_weak(state, newstate,
                                      std::memory_order_acquire)) {
        if(!all && num_prewaiters) return; // unblocked pre-wait thread
        if ((state & STACK_MASK) == STACK_MASK) return;
        Waiter* w = &_waiters[state & STACK_MASK];
        if(!all) {
          w->next.store(nullptr, std::memory_order_relaxed);
        }
        _unpark(w);
        return;
      }
    }
  }
};


}  // namespace tf ------------------------------------------------------------

