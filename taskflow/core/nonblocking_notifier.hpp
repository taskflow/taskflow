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
wid = this_worker_id();
if (predicate)
  return act();

ec.prepare_wait(wid);

if (predicate) {
  ec.cancel_wait(wid);
  return act();
}

ec.commit_wait(&w);
@endcode

A notifying thread performs:

@code{.cpp}
predicate = true;
ec.notify(true);
@endcode

The `notify` operation is inexpensive when no threads are waiting.
The `prepare_wait` and `commit_wait` operations are more costly, but
they are only executed when the initial optimistic predicate check
fails.

The algorithm is driven by two shared variables:
  - the user-managed predicate, and
  - an internal state variable (`_state`).

A waiting thread first publishes its intent to wait by updating
`_state` and then rechecks the predicate. Conversely, a notifying
thread first updates the predicate and then inspects `_state`.
Sequentially consistent memory fences between these operations ensure
that one of the following outcomes must occur:
  - the waiting thread observes the predicate change and does not
    block,
  - the notifying thread observes the waiting state and wakes the
    waiter, or
  - both observations occur.

It is impossible for both threads to miss each other’s updates, which
guarantees freedom from missed wake-ups and prevents deadlock.

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
  static const uint64_t STACK_BITS = 16;
  static const uint64_t STACK_MASK = (1ull << STACK_BITS) - 1;
  static const uint64_t PREWAITER_BITS = 16;
  static const uint64_t PREWAITER_SHIFT = 16;
  static const uint64_t PREWAITER_MASK = ((1ull << PREWAITER_BITS) - 1) << PREWAITER_SHIFT;
  static const uint64_t PREWAITER_INC = 1ull << PREWAITER_BITS;
  static const uint64_t EPOCH_BITS = 32;
  static const uint64_t EPOCH_SHIFT = 32;
  static const uint64_t EPOCH_MASK = ((1ull << EPOCH_BITS) - 1) << EPOCH_SHIFT;
  static const uint64_t EPOCH_INC = 1ull << EPOCH_SHIFT;

  explicit NonblockingNotifier(size_t N) : _state(STACK_MASK), _waiters(N) {
    assert(_waiters.size() < (1 << PREWAITER_BITS) - 1);
    // Initialize epoch to something close to overflow to test overflow.
    //_state = STACK_MASK | (EPOCH_MASK - EPOCH_INC * _waiters.size() * 2);
  }

  ~NonblockingNotifier() {
    // Ensure there are no waiters.
    assert((_state.load() & (STACK_MASK | PREWAITER_MASK)) == STACK_MASK);
  }

  size_t num_waiters() const {
    size_t n = 0;
    for(auto& w : _waiters) {
      n += (w.state.load(std::memory_order_relaxed) == Waiter::kWaiting);
      //std::scoped_lock lock(w.mu);
      //n += (w.state == Waiter::kWaiting);
    }
    return n;
  }

  // prepare_wait prepares for waiting.
  // After calling this function the thread must re-check the wait predicate
  // and call either cancel_wait or commit_wait passing the same Waiter object.
  void prepare_wait(size_t wid) {
    _waiters[wid].epoch = _state.fetch_add(PREWAITER_INC, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
  }

  // commit_wait commits waiting.
  // only the waiter itself can call
  void commit_wait(size_t wid) {

    auto w = &_waiters[wid];

    w->state.store(Waiter::kNotSignaled, std::memory_order_relaxed);

    // Modification epoch of this waiter.
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
      if ((state & STACK_MASK) == STACK_MASK) {
        w->next.store(nullptr, std::memory_order_relaxed);
      }
      else {
        w->next.store(&_waiters[state & STACK_MASK], std::memory_order_relaxed);
      }
      if (_state.compare_exchange_weak(state, newstate, std::memory_order_release)) {
        break;
      }
    }
    _park(w);
  }

  // cancel_wait cancels effects of the previous prepare_wait call.
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

  void notify_one() {
    _notify<false>();
  }

  void notify_all() {
    _notify<true>();
  }

  // notify n workers
  void notify_n(size_t n) {
    if(n >= _waiters.size()) {
      _notify<true>();
    }
    else {
      for(size_t k=0; k<n; ++k) {
        _notify<false>();
      }
    }
  }

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
  template <bool all>
  void _notify() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    uint64_t state = _state.load(std::memory_order_acquire);
    for (;;) {
      // Easy case: no waiters.
      if ((state & STACK_MASK) == STACK_MASK && (state & PREWAITER_MASK) == 0) {
        return;
      }
      uint64_t num_prewaiters = (state & PREWAITER_MASK) >> PREWAITER_SHIFT;
      uint64_t newstate;
      if constexpr (all) {
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
        if constexpr (!all) { if(num_prewaiters) return; }  // unblocked pre-wait thread
        if ((state & STACK_MASK) == STACK_MASK) return;
        Waiter* w = &_waiters[state & STACK_MASK];
        if constexpr (!all) {
          w->next.store(nullptr, std::memory_order_relaxed);
        }
        _unpark(w);
        return;
      }
    }
  }
};


}  // namespace tf ------------------------------------------------------------

