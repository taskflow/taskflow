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

// Notifier allows to wait for arbitrary predicates in non-blocking
// algorithms. Think of condition variable, but wait predicate does not need to
// be protected by a mutex. Usage:
// Waiting thread does:
//   
//   wid = this_worker_id();
//   if (predicate)
//     return act();
//   ec.prepare_wait(wid);
//   if (predicate) {
//     ec.cancel_wait(wid);
//     return act();
//   }
//   ec.commit_wait(&w);
//
// Notifying thread does:
//
//   predicate = true;
//   ec.notify(true);
//
// notify is cheap if there are no waiting threads. prepare_wait/commit_wait are not
// cheap, but they are executed only if the preceding predicate check has
// failed.
//
// Algorithm outline:
// There are two main variables: predicate (managed by user) and _state.
// Operation closely resembles Dekker mutual algorithm:
// https://en.wikipedia.org/wiki/Dekker%27s_algorithm
// Waiting thread sets _state then checks predicate, Notifying thread sets
// predicate then checks _state. Due to seq_cst fences in between these
// operations it is guaranteed than either waiter will see predicate change
// and won't block, or notifying thread will see _state change and will unblock
// the waiter, or both. But it can't happen that both threads don't see each
// other changes, which would lead to deadlock.
//
// Reference: https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/ThreadPool/EventCount.h

class NonblockingNotifier {

  friend class Executor;

  public:

  struct Waiter {
    alignas (TF_CACHELINE_SIZE) std::atomic<Waiter*> next;
    uint64_t epoch;
    enum : unsigned {
      kNotSignaled = 0,
      kWaiting,
      kSignaled,
    };

    mutable std::mutex mu;
    std::condition_variable cv;
    unsigned state;
  };

  explicit NonblockingNotifier(size_t N) : _state(kStackMask), _waiters(N) {
    assert(_waiters.size() < (1 << kPreWaiterBits) - 1);
    // Initialize epoch to something close to overflow to test overflow.
    //_state = kStackMask | (kEpochMask - kEpochInc * _waiters.size() * 2);
  }

  ~NonblockingNotifier() {
    // Ensure there are no waiters.
    assert((_state.load() & (kStackMask | kPreWaiterMask)) == kStackMask);
  }

  size_t num_waiters() const {
    size_t n = 0;
    for(auto& w : _waiters) {
      std::scoped_lock lock(w.mu);
      n += (w.state == Waiter::kWaiting);
    }
    return n;
  }

  // prepare_wait prepares for waiting.
  // After calling this function the thread must re-check the wait predicate
  // and call either cancel_wait or commit_wait passing the same Waiter object.
  void prepare_wait(size_t wid) {
    _waiters[wid].epoch = _state.fetch_add(kPreWaiterInc, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
  }

  // commit_wait commits waiting.
  // only the waiter itself can call
  void commit_wait(size_t wid) {

    auto w = &_waiters[wid];

    w->state = Waiter::kNotSignaled;
    // Modification epoch of this waiter.
    uint64_t epoch =
        (w->epoch & kEpochMask) +
        (((w->epoch & kPreWaiterMask) >> kPreWaiterShift) << kEpochShift);
    uint64_t state = _state.load(std::memory_order_seq_cst);
    for (;;) {
      if (int64_t((state & kEpochMask) - epoch) < 0) {
        // The preceding waiter has not decided on its fate. Wait until it
        // calls either cancel_wait or commit_wait, or is notified.
        std::this_thread::yield();
        state = _state.load(std::memory_order_seq_cst);
        continue;
      }
      // We've already been notified.
      if (int64_t((state & kEpochMask) - epoch) > 0) return;
      // Remove this thread from prewait counter and add it to the waiter list.
      assert((state & kPreWaiterMask) != 0);
      uint64_t newstate = state - kPreWaiterInc + kEpochInc;
      newstate = (newstate & ~kStackMask) | wid;
      if ((state & kStackMask) == kStackMask) {
        w->next.store(nullptr, std::memory_order_relaxed);
      }
      else {
        w->next.store(&_waiters[state & kStackMask], std::memory_order_relaxed);
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
      (_waiters[wid].epoch & kEpochMask) +
      (((_waiters[wid].epoch & kPreWaiterMask) >> kPreWaiterShift) << kEpochShift);
    uint64_t state = _state.load(std::memory_order_relaxed);
    for (;;) {
      if (int64_t((state & kEpochMask) - epoch) < 0) {
        // The preceding waiter has not decided on its fate. Wait until it
        // calls either cancel_wait or commit_wait, or is notified.
        std::this_thread::yield();
        state = _state.load(std::memory_order_relaxed);
        continue;
      }
      // We've already been notified.
      if (int64_t((state & kEpochMask) - epoch) > 0) {
        return;
      }
      // Remove this thread from prewait counter.
      assert((state & kPreWaiterMask) != 0);
      if (_state.compare_exchange_weak(state, state - kPreWaiterInc + kEpochInc,
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

  // State_ layout:
  // - low kStackBits is a stack of waiters committed wait.
  // - next kPreWaiterBits is count of waiters in prewait state.
  // - next kEpochBits is modification counter.
  static const uint64_t kStackBits = 16;
  static const uint64_t kStackMask = (1ull << kStackBits) - 1;
  static const uint64_t kPreWaiterBits = 16;
  static const uint64_t kPreWaiterShift = 16;
  static const uint64_t kPreWaiterMask = ((1ull << kPreWaiterBits) - 1)
                                      << kPreWaiterShift;
  static const uint64_t kPreWaiterInc = 1ull << kPreWaiterBits;
  static const uint64_t kEpochBits = 32;
  static const uint64_t kEpochShift = 32;
  static const uint64_t kEpochMask = ((1ull << kEpochBits) - 1) << kEpochShift;
  static const uint64_t kEpochInc = 1ull << kEpochShift;

  std::atomic<uint64_t> _state;
  std::vector<Waiter> _waiters;

  void _park(Waiter* w) {
    std::unique_lock<std::mutex> lock(w->mu);
    while (w->state != Waiter::kSignaled) {
      w->state = Waiter::kWaiting;
      w->cv.wait(lock);
    }
  }

  void _unpark(Waiter* waiters) {
    Waiter* next = nullptr;
    for (Waiter* w = waiters; w; w = next) {
      next = w->next.load(std::memory_order_relaxed);
      unsigned state;
      {
        std::unique_lock<std::mutex> lock(w->mu);
        state = w->state;
        w->state = Waiter::kSignaled;
      }
      // Avoid notifying if it wasn't waiting.
      if (state == Waiter::kWaiting) w->cv.notify_one();
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
      if ((state & kStackMask) == kStackMask && (state & kPreWaiterMask) == 0) {
        return;
      }
      uint64_t num_pre_waiters = (state & kPreWaiterMask) >> kPreWaiterShift;
      uint64_t newstate;
      if constexpr (all) {
        // Reset prewait counter and empty wait list.
        newstate = (state & kEpochMask) + (kEpochInc * num_pre_waiters) + kStackMask;
      } else if (num_pre_waiters) {
        // There is a thread in pre-wait state, unblock it.
        newstate = state + kEpochInc - kPreWaiterInc;
      } else {
        // Pop a waiter from list and unpark it.
        Waiter* w = &_waiters[state & kStackMask];
        Waiter* wnext = w->next.load(std::memory_order_relaxed);
        uint64_t next = kStackMask;
        //if (wnext != nullptr) next = wnext - &_waiters[0];
        if (wnext != nullptr) {
          next = static_cast<uint64_t>(wnext - &_waiters[0]);
        }
        // Note: we don't add kEpochInc here. ABA problem on the lock-free stack
        // can't happen because a waiter is re-pushed onto the stack only after
        // it was in the pre-wait state which inevitably leads to epoch
        // increment.
        newstate = (state & kEpochMask) + next;
      }
      if (_state.compare_exchange_weak(state, newstate,
                                      std::memory_order_acquire)) {
        if constexpr (!all) { if(num_pre_waiters) return; }  // unblocked pre-wait thread
        if ((state & kStackMask) == kStackMask) return;
        Waiter* w = &_waiters[state & kStackMask];
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

