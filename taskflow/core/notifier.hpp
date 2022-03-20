// 2019/02/09 - created by Tsung-Wei Huang
//  - modified the event count from Eigen

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

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

namespace tf {

// Notifier allows to wait for arbitrary predicates in non-blocking
// algorithms. Think of condition variable, but wait predicate does not need to
// be protected by a mutex. Usage:
// Waiting thread does:
//
//   if (predicate)
//     return act();
//   Notifier::Waiter& w = waiters[my_index];
//   ec.prepare_wait(&w);
//   if (predicate) {
//     ec.cancel_wait(&w);
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
// cheap, but they are executed only if the preceeding predicate check has
// failed.
//
// Algorihtm outline:
// There are two main variables: predicate (managed by user) and _state.
// Operation closely resembles Dekker mutual algorithm:
// https://en.wikipedia.org/wiki/Dekker%27s_algorithm
// Waiting thread sets _state then checks predicate, Notifying thread sets
// predicate then checks _state. Due to seq_cst fences in between these
// operations it is guaranteed than either waiter will see predicate change
// and won't block, or notifying thread will see _state change and will unblock
// the waiter, or both. But it can't happen that both threads don't see each
// other changes, which would lead to deadlock.
class Notifier {

  friend class Executor;

  public:

  struct Waiter {
    std::atomic<Waiter*> next;
    std::mutex mu;
    std::condition_variable cv;
    uint64_t epoch;
    unsigned state;
    enum {
      kNotSignaled,
      kWaiting,
      kSignaled,
    };
  };

  explicit Notifier(size_t N) : _waiters{N} {
    assert(_waiters.size() < (1 << kWaiterBits) - 1);
    // Initialize epoch to something close to overflow to test overflow.
    _state = kStackMask | (kEpochMask - kEpochInc * _waiters.size() * 2);
  }

  ~Notifier() {
    // Ensure there are no waiters.
    assert((_state.load() & (kStackMask | kWaiterMask)) == kStackMask);
  }

  // prepare_wait prepares for waiting.
  // After calling this function the thread must re-check the wait predicate
  // and call either cancel_wait or commit_wait passing the same Waiter object.
  void prepare_wait(Waiter* w) {
    w->epoch = _state.fetch_add(kWaiterInc, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
  }

  // commit_wait commits waiting.
  void commit_wait(Waiter* w) {
    w->state = Waiter::kNotSignaled;
    // Modification epoch of this waiter.
    uint64_t epoch =
        (w->epoch & kEpochMask) +
        (((w->epoch & kWaiterMask) >> kWaiterShift) << kEpochShift);
    uint64_t state = _state.load(std::memory_order_seq_cst);
    for (;;) {
      if (int64_t((state & kEpochMask) - epoch) < 0) {
        // The preceeding waiter has not decided on its fate. Wait until it
        // calls either cancel_wait or commit_wait, or is notified.
        std::this_thread::yield();
        state = _state.load(std::memory_order_seq_cst);
        continue;
      }
      // We've already been notified.
      if (int64_t((state & kEpochMask) - epoch) > 0) return;
      // Remove this thread from prewait counter and add it to the waiter list.
      assert((state & kWaiterMask) != 0);
      uint64_t newstate = state - kWaiterInc + kEpochInc;
      //newstate = (newstate & ~kStackMask) | (w - &_waiters[0]);
      newstate = static_cast<uint64_t>((newstate & ~kStackMask) | static_cast<uint64_t>(w - &_waiters[0]));
      if ((state & kStackMask) == kStackMask)
        w->next.store(nullptr, std::memory_order_relaxed);
      else
        w->next.store(&_waiters[state & kStackMask], std::memory_order_relaxed);
      if (_state.compare_exchange_weak(state, newstate,
                                       std::memory_order_release))
        break;
    }
    _park(w);
  }

  // cancel_wait cancels effects of the previous prepare_wait call.
  void cancel_wait(Waiter* w) {
    uint64_t epoch =
        (w->epoch & kEpochMask) +
        (((w->epoch & kWaiterMask) >> kWaiterShift) << kEpochShift);
    uint64_t state = _state.load(std::memory_order_relaxed);
    for (;;) {
      if (int64_t((state & kEpochMask) - epoch) < 0) {
        // The preceeding waiter has not decided on its fate. Wait until it
        // calls either cancel_wait or commit_wait, or is notified.
        std::this_thread::yield();
        state = _state.load(std::memory_order_relaxed);
        continue;
      }
      // We've already been notified.
      if (int64_t((state & kEpochMask) - epoch) > 0) return;
      // Remove this thread from prewait counter.
      assert((state & kWaiterMask) != 0);
      if (_state.compare_exchange_weak(state, state - kWaiterInc + kEpochInc,
                                       std::memory_order_relaxed))
        return;
    }
  }

  // notify wakes one or all waiting threads.
  // Must be called after changing the associated wait predicate.
  void notify(bool all) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    uint64_t state = _state.load(std::memory_order_acquire);
    for (;;) {
      // Easy case: no waiters.
      if ((state & kStackMask) == kStackMask && (state & kWaiterMask) == 0)
        return;
      uint64_t waiters = (state & kWaiterMask) >> kWaiterShift;
      uint64_t newstate;
      if (all) {
        // Reset prewait counter and empty wait list.
        newstate = (state & kEpochMask) + (kEpochInc * waiters) + kStackMask;
      } else if (waiters) {
        // There is a thread in pre-wait state, unblock it.
        newstate = state + kEpochInc - kWaiterInc;
      } else {
        // Pop a waiter from list and unpark it.
        Waiter* w = &_waiters[state & kStackMask];
        Waiter* wnext = w->next.load(std::memory_order_relaxed);
        uint64_t next = kStackMask;
        //if (wnext != nullptr) next = wnext - &_waiters[0];
        if (wnext != nullptr) next = static_cast<uint64_t>(wnext - &_waiters[0]);
        // Note: we don't add kEpochInc here. ABA problem on the lock-free stack
        // can't happen because a waiter is re-pushed onto the stack only after
        // it was in the pre-wait state which inevitably leads to epoch
        // increment.
        newstate = (state & kEpochMask) + next;
      }
      if (_state.compare_exchange_weak(state, newstate,
                                       std::memory_order_acquire)) {
        if (!all && waiters) return;  // unblocked pre-wait thread
        if ((state & kStackMask) == kStackMask) return;
        Waiter* w = &_waiters[state & kStackMask];
        if (!all) w->next.store(nullptr, std::memory_order_relaxed);
        _unpark(w);
        return;
      }
    }
  }

  // notify n workers
  void notify_n(size_t n) {
    if(n >= _waiters.size()) {
      notify(true);
    }
    else {
      for(size_t k=0; k<n; ++k) {
        notify(false);
      }
    }
  }

  size_t size() const {
    return _waiters.size();
  }

 private:

  // State_ layout:
  // - low kStackBits is a stack of waiters committed wait.
  // - next kWaiterBits is count of waiters in prewait state.
  // - next kEpochBits is modification counter.
  static const uint64_t kStackBits = 16;
  static const uint64_t kStackMask = (1ull << kStackBits) - 1;
  static const uint64_t kWaiterBits = 16;
  static const uint64_t kWaiterShift = 16;
  static const uint64_t kWaiterMask = ((1ull << kWaiterBits) - 1)
                                      << kWaiterShift;
  static const uint64_t kWaiterInc = 1ull << kWaiterBits;
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

};



}  // namespace tf ------------------------------------------------------------

