#pragma once

#include <atomic>
#include <thread>
#include <vector>
#include <assert.h>
#include "../utility/os.hpp"

namespace tf {

//-----------------------------------------------------------------------------

/**

@class AtomicNotifier

@brief class to create a notifier for inter-thread synchronization

A notifier is a lightweight synchronization primitive for waiting on arbitrary predicates
without traditional mutex-based blocking
The notifier enables non-blocking coordination between threads. Conceptually, it plays
a similar role to a condition variable, but the waiting condition (predicate) does not
need to be guarded by a mutex.

**Typical usage**

In a waiting thread:

@code{.cpp}
if (predicate)
  return act();
Notifier::Waiter& w = waiters[my_index];
ec.prepare_wait(&w);
if (predicate) {
  ec.cancel_wait(&w);
  return act();
}
ec.commit_wait(&w);
@endcode

In a notifying thread:

@code{.cpp}
predicate = true;
ec.notify(true);
@endcode

The `notify` operation is inexpensive when there are no waiting threads.
`prepare_wait` and `commit_wait` are more costly but are only executed when
the predicate check fails initially.

**Algorithm overview**

The design relies on two key variables: the user-managed `predicate` and an internal
`_state`. The synchronization behavior is reminiscent of Dekker’s algorithm
(see: https://en.wikipedia.org/wiki/Dekker%27s_algorithm).

- A waiting thread first updates `_state`, then checks the predicate.
- A notifying thread updates the predicate, then checks `_state`.

Sequentially consistent fences ensure visibility between threads. As a result,
at least one side will observe the other's update—either:
 - The waiter sees the predicate become true and avoids blocking, or
 - The notifier sees `_state` change and wakes the waiter.

This protocol prevents the deadlock case where both threads miss each other’s updates.
*/

class AtomicNotifier {

  friend class Executor;

  public:
  
  struct Waiter {
    alignas (2*TF_CACHELINE_SIZE) uint32_t epoch;
  };

  AtomicNotifier(size_t N) noexcept : _state(0), _waiters(N) {}
  ~AtomicNotifier() { assert((_state.load() & WAITER_MASK) == 0); } 

  void notify_one() noexcept;
  void notify_all() noexcept;
  void notify_n(size_t) noexcept;
  void prepare_wait(size_t) noexcept;
  void cancel_wait(size_t) noexcept;
  void commit_wait(size_t) noexcept;

  size_t size() const noexcept;
  size_t num_waiters() const noexcept;

 private:

  AtomicNotifier(const AtomicNotifier&) = delete;
  AtomicNotifier(AtomicNotifier&&) = delete;
  AtomicNotifier& operator=(const AtomicNotifier&) = delete;
  AtomicNotifier& operator=(AtomicNotifier&&) = delete;

  // This requires 64-bit
  static_assert(sizeof(int) == 4, "bad platform");
  static_assert(sizeof(uint32_t) == 4, "bad platform");
  static_assert(sizeof(uint64_t) == 8, "bad platform");
  static_assert(sizeof(std::atomic<uint64_t>) == 8, "bad platform");

  // _state stores the epoch in the most significant 32 bits and the
  // waiter count in the least significant 32 bits.
  std::atomic<uint64_t> _state;
  std::vector<Waiter> _waiters;

  static constexpr uint64_t WAITER_INC  {1};
  static constexpr uint64_t EPOCH_SHIFT {32};
  static constexpr uint64_t EPOCH_INC   {uint64_t(1) << EPOCH_SHIFT};
  static constexpr uint64_t WAITER_MASK {EPOCH_INC - 1};
};

inline size_t AtomicNotifier::size() const noexcept {
  return _waiters.size();
}

inline size_t AtomicNotifier::num_waiters() const noexcept {
  return _state.load(std::memory_order_relaxed) & WAITER_MASK;
}

inline void AtomicNotifier::notify_one() noexcept {
  std::atomic_thread_fence(std::memory_order_seq_cst);
  for(uint64_t state = _state.load(std::memory_order_relaxed); state & WAITER_MASK;) {
    if(_state.compare_exchange_weak(state, state + EPOCH_INC, std::memory_order_relaxed)) {
      _state.notify_one(); 
      break;
    }
  }
}

inline void AtomicNotifier::notify_all() noexcept {
  std::atomic_thread_fence(std::memory_order_seq_cst);
  for(uint64_t state = _state.load(std::memory_order_relaxed); state & WAITER_MASK;) {
    if(_state.compare_exchange_weak(state, state + EPOCH_INC, std::memory_order_relaxed)) {
      _state.notify_all(); 
      break;
    }
  }
}
  
inline void AtomicNotifier::notify_n(size_t n) noexcept {
  if(n >= _waiters.size()) {
    notify_all();
  }
  else {
    for(size_t k=0; k<n; ++k) {
      notify_one();
    }
  }
}

inline void AtomicNotifier::prepare_wait(size_t w) noexcept {
  auto prev = _state.fetch_add(WAITER_INC, std::memory_order_relaxed);
  _waiters[w].epoch = (prev >> EPOCH_SHIFT);
  std::atomic_thread_fence(std::memory_order_seq_cst);
}

inline void AtomicNotifier::cancel_wait(size_t) noexcept {
  _state.fetch_sub(WAITER_INC, std::memory_order_relaxed);
}

inline void AtomicNotifier::commit_wait(size_t w) noexcept {
  uint64_t prev = _state.load(std::memory_order_relaxed);
  while((prev >> EPOCH_SHIFT) == _waiters[w].epoch) {
    _state.wait(prev, std::memory_order_relaxed); 
    prev = _state.load(std::memory_order_relaxed);
  }
  _state.fetch_sub(WAITER_INC, std::memory_order_relaxed);
}


} // namespace taskflow -------------------------------------------------------

