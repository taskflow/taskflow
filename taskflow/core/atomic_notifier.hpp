#if __cplusplus >= TF_CPP20

#pragma once

#include <atomic>
#include <thread>
#include <vector>

namespace tf {

class AtomicNotifierV1 {

  friend class Executor;

  public:
  
  struct Waiter {
    alignas (2*TF_CACHELINE_SIZE) uint32_t epoch;
  };

  AtomicNotifierV1(size_t N) noexcept : _state(0), _waiters(N) {}
  ~AtomicNotifierV1() { assert((_state.load() & WAITER_MASK) == 0); } 

  void notify_one() noexcept;
  void notify_all() noexcept;
  void notify_n(size_t n) noexcept;
  void prepare_wait(Waiter*) noexcept;
  void cancel_wait(Waiter*) noexcept;
  void commit_wait(Waiter*) noexcept;
  size_t size() const noexcept;

 private:

  AtomicNotifierV1(const AtomicNotifierV1&) = delete;
  AtomicNotifierV1(AtomicNotifierV1&&) = delete;
  AtomicNotifierV1& operator=(const AtomicNotifierV1&) = delete;
  AtomicNotifierV1& operator=(AtomicNotifierV1&&) = delete;

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
  static constexpr size_t   EPOCH_SHIFT {32};
  static constexpr uint64_t EPOCH_INC   {uint64_t(1) << EPOCH_SHIFT};
  static constexpr uint64_t WAITER_MASK {EPOCH_INC - 1};
};

inline void AtomicNotifierV1::notify_one() noexcept {
  uint64_t prev = _state.fetch_add(EPOCH_INC, std::memory_order_acq_rel);
  if(TF_UNLIKELY(prev & WAITER_MASK))  { // has waiter (typically unlikely)
    _state.notify_one();
  }
}

inline void AtomicNotifierV1::notify_all() noexcept {
  uint64_t prev = _state.fetch_add(EPOCH_INC, std::memory_order_acq_rel);
  if(TF_UNLIKELY(prev & WAITER_MASK))  { // has waiter (typically unlikely)
    _state.notify_all();
  }
}
  
inline void AtomicNotifierV1::notify_n(size_t n) noexcept {
  if(n >= _waiters.size()) {
    notify_all();
  }
  else {
    for(size_t k=0; k<n; ++k) {
      notify_one();
    }
  }
}

inline size_t AtomicNotifierV1::size() const noexcept {
  return _waiters.size();
}

inline void AtomicNotifierV1::prepare_wait(Waiter* waiter) noexcept {
  uint64_t prev = _state.fetch_add(WAITER_INC, std::memory_order_acq_rel);
  waiter->epoch = (prev >> EPOCH_SHIFT);
}

inline void AtomicNotifierV1::cancel_wait(Waiter*) noexcept {
  // memory_order_relaxed would suffice for correctness, but the faster
  // #waiters gets to 0, the less likely it is that we'll do spurious wakeups
  // (and thus system calls).
  _state.fetch_sub(WAITER_INC, std::memory_order_seq_cst);
}

inline void AtomicNotifierV1::commit_wait(Waiter* waiter) noexcept {
  uint64_t prev = _state.load(std::memory_order_acquire);
  while((prev >> EPOCH_SHIFT) == waiter->epoch) {
    _state.wait(prev, std::memory_order_acquire); 
    prev = _state.load(std::memory_order_acquire);
  }
  // memory_order_relaxed would suffice for correctness, but the faster
  // #waiters gets to 0, the less likely it is that we'll do spurious wakeups
  // (and thus system calls)
  _state.fetch_sub(WAITER_INC, std::memory_order_seq_cst);
}

//-----------------------------------------------------------------------------

class AtomicNotifierV2 {

  friend class Executor;

  public:
  
  struct Waiter {
    alignas (2*TF_CACHELINE_SIZE) uint32_t epoch;
  };

  AtomicNotifierV2(size_t N) noexcept : _state(0), _waiters(N) {}
  ~AtomicNotifierV2() { assert((_state.load() & WAITER_MASK) == 0); } 

  void notify_one() noexcept;
  void notify_all() noexcept;
  void notify_n(size_t n) noexcept;
  void prepare_wait(Waiter*) noexcept;
  void cancel_wait(Waiter*) noexcept;
  void commit_wait(Waiter*) noexcept;
  size_t size() const noexcept;

 private:

  AtomicNotifierV2(const AtomicNotifierV2&) = delete;
  AtomicNotifierV2(AtomicNotifierV2&&) = delete;
  AtomicNotifierV2& operator=(const AtomicNotifierV2&) = delete;
  AtomicNotifierV2& operator=(AtomicNotifierV2&&) = delete;

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

inline void AtomicNotifierV2::notify_one() noexcept {
  std::atomic_thread_fence(std::memory_order_seq_cst);
  //if((_state.load(std::memory_order_acquire) & WAITER_MASK) != 0) {
  //  _state.fetch_add(EPOCH_INC, std::memory_order_relaxed);
  //  _state.notify_one(); 
  //}

  for(uint64_t state = _state.load(std::memory_order_acquire); state & WAITER_MASK;) {
    if(_state.compare_exchange_weak(state, state + EPOCH_INC, std::memory_order_acquire)) {
      _state.notify_one(); 
      break;
    }
  }
}

inline void AtomicNotifierV2::notify_all() noexcept {
  std::atomic_thread_fence(std::memory_order_seq_cst);
  //if((_state.load(std::memory_order_acquire) & WAITER_MASK) != 0) {
  //  _state.fetch_add(EPOCH_INC, std::memory_order_relaxed);
  //  _state.notify_all(); 
  //}
  for(uint64_t state = _state.load(std::memory_order_acquire); state & WAITER_MASK;) {
    if(_state.compare_exchange_weak(state, state + EPOCH_INC, std::memory_order_acquire)) {
      _state.notify_all(); 
      break;
    }
  }
}
  
inline void AtomicNotifierV2::notify_n(size_t n) noexcept {
  if(n >= _waiters.size()) {
    notify_all();
  }
  else {
    for(size_t k=0; k<n; ++k) {
      notify_one();
    }
  }
}

inline size_t AtomicNotifierV2::size() const noexcept {
  return _waiters.size();
}

inline void AtomicNotifierV2::prepare_wait(Waiter* waiter) noexcept {
  auto prev = _state.fetch_add(WAITER_INC, std::memory_order_relaxed);
  waiter->epoch = (prev >> EPOCH_SHIFT);
  std::atomic_thread_fence(std::memory_order_seq_cst);
}

inline void AtomicNotifierV2::cancel_wait(Waiter*) noexcept {
  _state.fetch_sub(WAITER_INC, std::memory_order_seq_cst);
}

inline void AtomicNotifierV2::commit_wait(Waiter* waiter) noexcept {
  uint64_t prev = _state.load(std::memory_order_acquire);
  while((prev >> EPOCH_SHIFT) == waiter->epoch) {
    _state.wait(prev, std::memory_order_acquire); 
    prev = _state.load(std::memory_order_acquire);
  }
  // memory_order_relaxed would suffice for correctness, but the faster
  // #waiters gets to 0, the less likely it is that we'll do spurious wakeups
  // (and thus system calls)
  _state.fetch_sub(WAITER_INC, std::memory_order_seq_cst);
}



} // namespace taskflow -------------------------------------------------------

#endif
