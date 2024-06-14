#pragma once

#include <atomic>
#include <thread>

namespace tf {

class Notifier {

 public:

  Notifier() noexcept : _state(0) {}
  ~Notifier() { assert((_state.load() & WAITER_MASK) == 0); } 

  class Key {
    friend class Notifier;
    explicit Key(uint32_t e) noexcept : _epoch(e) {}
    uint32_t _epoch;
  };

  void notify_one() noexcept;
  void notify_all() noexcept;
  Key prepare_wait() noexcept;
  void cancel_wait() noexcept;
  void wait(Key key) noexcept;

 private:

  Notifier(const Notifier&) = delete;
  Notifier(Notifier&&) = delete;
  Notifier& operator=(const Notifier&) = delete;
  Notifier& operator=(Notifier&&) = delete;

  // This requires 64-bit
  static_assert(sizeof(int) == 4, "bad platform");
  static_assert(sizeof(uint32_t) == 4, "bad platform");
  static_assert(sizeof(uint64_t) == 8, "bad platform");
  static_assert(sizeof(std::atomic<uint64_t>) == 8, "bad platform");

  //static constexpr size_t kEpochOffset = kIsLittleEndian ? 1 : 0;

  // _state stores the epoch in the most significant 32 bits and the
  // waiter count in the least significant 32 bits.
  std::atomic<uint64_t> _state;

  static constexpr uint64_t WAITER_INC  {1};
  static constexpr size_t   EPOCH_SHIFT {32};
  static constexpr uint64_t EPOCH_INC   {uint64_t(1) << EPOCH_SHIFT};
  static constexpr uint64_t WAITER_MASK {EPOCH_INC - 1};
};

inline void Notifier::notify_one() noexcept {
  uint64_t prev = _state.fetch_add(EPOCH_INC, std::memory_order_acq_rel);
  if((prev & WAITER_MASK))  { // has waiter (typically unlikely)
    _state.notify_one();
  }
}

inline void Notifier::notify_all() noexcept {
  uint64_t prev = _state.fetch_add(EPOCH_INC, std::memory_order_acq_rel);
  if((prev & WAITER_MASK))  { // has waiter (typically unlikely)
    _state.notify_all();
  }
}

inline Notifier::Key Notifier::prepare_wait() noexcept {
  uint64_t prev = _state.fetch_add(WAITER_INC, std::memory_order_acq_rel);
  return Key(prev >> EPOCH_SHIFT);
}

inline void Notifier::cancel_wait() noexcept {
  // memory_order_relaxed would suffice for correctness, but the faster
  // #waiters gets to 0, the less likely it is that we'll do spurious wakeups
  // (and thus system calls).
  _state.fetch_sub(WAITER_INC, std::memory_order_seq_cst);
}

inline void Notifier::wait(Key key) noexcept {
  uint64_t prev = _state.load(std::memory_order_acquire);
  while((prev >> EPOCH_SHIFT) == key._epoch) {
    _state.wait(prev, std::memory_order_acquire); 
    prev = _state.load(std::memory_order_acquire);
  }
  // memory_order_relaxed would suffice for correctness, but the faster
  // #waiters gets to 0, the less likely it is that we'll do spurious wakeups
  // (and thus system calls)
  _state.fetch_sub(WAITER_INC, std::memory_order_seq_cst);
}


} // namespace taskflow


