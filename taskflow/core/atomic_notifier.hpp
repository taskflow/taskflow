#ifdef __cpp_lib_atomic_wait

#pragma once

#include <atomic>
#include <thread>
#include <vector>

namespace tf {

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
  void notify_n(size_t n) noexcept;
  void prepare_wait(Waiter*) noexcept;
  void cancel_wait(Waiter*) noexcept;
  void commit_wait(Waiter*) noexcept;
  size_t size() const noexcept;

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

  //static constexpr size_t kEpochOffset = kIsLittleEndian ? 1 : 0;

  // _state stores the epoch in the most significant 32 bits and the
  // waiter count in the least significant 32 bits.
  std::atomic<uint64_t> _state;
  std::vector<Waiter> _waiters;

  static constexpr uint64_t WAITER_INC  {1};
  static constexpr size_t   EPOCH_SHIFT {32};
  static constexpr uint64_t EPOCH_INC   {uint64_t(1) << EPOCH_SHIFT};
  static constexpr uint64_t WAITER_MASK {EPOCH_INC - 1};
};

inline void AtomicNotifier::notify_one() noexcept {
  uint64_t prev = _state.fetch_add(EPOCH_INC, std::memory_order_acq_rel);
  if((prev & WAITER_MASK))  { // has waiter (typically unlikely)
    _state.notify_one();
  }
}

inline void AtomicNotifier::notify_all() noexcept {
  uint64_t prev = _state.fetch_add(EPOCH_INC, std::memory_order_acq_rel);
  if((prev & WAITER_MASK))  { // has waiter (typically unlikely)
    _state.notify_all();
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

inline size_t AtomicNotifier::size() const noexcept {
  return _waiters.size();
}

inline void AtomicNotifier::prepare_wait(Waiter* waiter) noexcept {
  uint64_t prev = _state.fetch_add(WAITER_INC, std::memory_order_acq_rel);
  waiter->epoch = (prev >> EPOCH_SHIFT);
}

inline void AtomicNotifier::cancel_wait(Waiter*) noexcept {
  // memory_order_relaxed would suffice for correctness, but the faster
  // #waiters gets to 0, the less likely it is that we'll do spurious wakeups
  // (and thus system calls).
  _state.fetch_sub(WAITER_INC, std::memory_order_seq_cst);
}

inline void AtomicNotifier::commit_wait(Waiter* waiter) noexcept {
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


} // namespace taskflow

#endif
