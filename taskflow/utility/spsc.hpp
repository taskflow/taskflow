#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <type_traits>
#include "os.hpp"

/**
@file spsc.hpp
@brief lock-free single-producer / single-consumer ring buffer

Designed and implemented by Matthew Busel.
*/

namespace tf {

// ----------------------------------------------------------------------------
// SPSCRing
// ----------------------------------------------------------------------------

/**
@class SPSCRing

@tparam T       element type; must be noexcept move-constructible and
                noexcept move-assignable
@tparam LogSize base-2 logarithm of the internal buffer size;
                capacity = 2^LogSize - 1; LogSize must be in [1, 30]

@brief lock-free single-producer / single-consumer ring buffer

%SPSCRing provides a fixed-capacity, zero-heap-allocation queue safe
for exactly one producer thread and one consumer thread.  Unlike
@ref UnboundedWSQ and @ref BoundedWSQ (which support work-stealing
by multiple thief threads), %SPSCRing is specialised for the strict
1-producer / 1-consumer case and pays no cost for multi-consumer
coordination.

@par Guarantees
- <b>Lock-free</b>: no mutexes, condition variables, or spin-locks.
- <b>Zero allocation</b>: the buffer is stored inline (std::array); the
  total object size is <tt>2^LogSize * sizeof(T) + 128</tt> bytes.
- <b>Cache-friendly</b>: producer index and consumer index live on
  separate cache lines (<tt>alignas(64)</tt>) to eliminate false sharing.
- <b>noexcept</b>: @ref push and @ref pop are unconditionally noexcept
  provided T satisfies the noexcept move requirements.
- <b>Non-blocking</b>: @ref push returns @c false when full; @ref pop
  returns an empty @ref value_type when empty.

@par Capacity
The usable capacity is <tt>2^LogSize - 1</tt>.  One slot is reserved to
distinguish the full state from the empty state without a separate counter.

@par Return-type specialization
Like @ref UnboundedWSQ, the return type of @ref pop is specialized on T:
- For pointer types, @ref pop returns @c T (using @c nullptr as the empty sentinel).
- For non-pointer types, @ref pop returns @c std::optional<T> (using
  @c std::nullopt as the empty sentinel).

@par Typical usage

@code{.cpp}
// shared between producer and consumer threads
tf::SPSCRing<int, 7> ring;   // capacity = 2^7 - 1 = 127

// --- producer thread ---
ring.push(42);               // returns false if full

// --- consumer thread ---
if (auto v = ring.pop()) {
    process(*v);             // std::nullopt / nullptr when empty
}
@endcode

@note Create one %SPSCRing per producer-consumer pair.
      %SPSCRing is not safe for multiple producers or multiple consumers.
*/
template <typename T, size_t LogSize>
class SPSCRing {

  static_assert(LogSize >= 1,
    "tf::SPSCRing: LogSize must be at least 1");
  static_assert(LogSize <= 30,
    "tf::SPSCRing: LogSize must be at most 30");
  static_assert(std::is_nothrow_move_constructible_v<T>,
    "tf::SPSCRing: T must be noexcept move-constructible");
  static_assert(std::is_nothrow_move_assignable_v<T>,
    "tf::SPSCRing: T must be noexcept move-assignable");

  static constexpr size_t SIZE = size_t{1} << LogSize;
  static constexpr size_t MASK = SIZE - 1u;

  public:

  /**
  @brief the return type of @ref pop

  For pointer element types @c T, `value_type` is @c T itself and uses
  @c nullptr to indicate an empty result.  For non-pointer types, it is
  @c std::optional<T>, where @c std::nullopt denotes the absence of a value.

  @code{.cpp}
  static_assert(std::is_same_v<tf::SPSCRing<int,   7>::value_type, std::optional<int>>);
  static_assert(std::is_same_v<tf::SPSCRing<int*,  7>::value_type, int*>);
  @endcode
  */
  using value_type = std::conditional_t<std::is_pointer_v<T>, T, std::optional<T>>;

  /**
  @brief returns the maximum number of elements the ring can hold simultaneously

  The capacity is <tt>2^LogSize - 1</tt>; one slot is reserved for
  full/empty disambiguation.
  */
  [[nodiscard]] static constexpr size_t capacity() noexcept {
    return SIZE - 1u;
  }

  /**
  @brief returns the empty sentinel value appropriate for @ref value_type

  For pointer types this is @c nullptr; for non-pointer types it is
  @c std::nullopt.
  */
  static constexpr value_type empty_value() noexcept {
    if constexpr (std::is_pointer_v<T>) {
      return T{nullptr};
    } else {
      return std::optional<T>{std::nullopt};
    }
  }

  // --------------------------------------------------------------------------
  // Producer API - call from a single producer thread only
  // --------------------------------------------------------------------------

  /**
  @brief tries to enqueue one element

  Moves @p item into the ring.  Returns immediately without blocking.

  @param item  element to enqueue (moved from on success)

  @return @c true on success; @c false if the ring is full (@p item is
          left in a valid but unspecified state)

  @note Call from the <em>producer thread only</em>.
  */
  [[nodiscard]] bool push(T&& item) noexcept {
    const size_t tail = _tail.load(std::memory_order_relaxed);
    const size_t next = (tail + 1u) & MASK;
    if (next == _head.load(std::memory_order_acquire)) {
      return false; // full
    }
    _buf[tail] = std::move(item);
    _tail.store(next, std::memory_order_release);
    return true;
  }

  /**
  @brief tries to enqueue a copy of one element

  Convenience overload for copy-constructible types.

  @param item  element to copy-enqueue
  @return @c true on success; @c false if full
  */
  [[nodiscard]] bool push(const T& item) noexcept(std::is_nothrow_copy_constructible_v<T>) {
    T copy{item};
    return push(std::move(copy));
  }

  // --------------------------------------------------------------------------
  // Consumer API - call from a single consumer thread only
  // --------------------------------------------------------------------------

  /**
  @brief tries to dequeue one element

  Returns immediately without blocking.

  @return @ref value_type containing the dequeued element on success,
          or the empty sentinel (@c std::nullopt / @c nullptr) if empty

  @note Call from the <em>consumer thread only</em>.
  */
  [[nodiscard]] value_type pop() noexcept {
    const size_t head = _head.load(std::memory_order_relaxed);
    if (head == _tail.load(std::memory_order_acquire)) {
      return empty_value(); // empty
    }
    T item{std::move(_buf[head])};
    _head.store((head + 1u) & MASK, std::memory_order_release);
    if constexpr (std::is_pointer_v<T>) {
      return item;
    } else {
      return std::optional<T>{std::move(item)};
    }
  }

  // --------------------------------------------------------------------------
  // Diagnostic queries (approximate - non-atomic snapshot)
  // --------------------------------------------------------------------------

  /**
  @brief returns an approximate element count

  Not guaranteed to be exact under concurrent use.  Suitable for
  monitoring and metrics only.
  */
  [[nodiscard]] size_t size_approx() const noexcept {
    const size_t t = _tail.load(std::memory_order_acquire);
    const size_t h = _head.load(std::memory_order_acquire);
    return (t - h + SIZE) & MASK;
  }

  /**
  @brief returns @c true if the ring appears empty (approximate)
  */
  [[nodiscard]] bool empty_approx() const noexcept {
    return _tail.load(std::memory_order_acquire) ==
           _head.load(std::memory_order_acquire);
  }

  /**
  @brief returns @c true if the ring appears full (approximate)
  */
  [[nodiscard]] bool full_approx() const noexcept {
    const size_t t = _tail.load(std::memory_order_acquire);
    const size_t h = _head.load(std::memory_order_acquire);
    return ((t + 1u) & MASK) == h;
  }

  private:

  alignas(TF_CACHELINE_SIZE) std::atomic<size_t> _tail{0};
  alignas(TF_CACHELINE_SIZE) std::atomic<size_t> _head{0};
  std::array<T, SIZE> _buf{};
};

} // namespace tf
