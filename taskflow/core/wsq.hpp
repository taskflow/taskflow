#pragma once

#include <bit>

#include "../utility/macros.hpp"
#include "../utility/traits.hpp"

/**
@file wsq.hpp
@brief work-stealing queue (wsq) include file
*/

#ifndef TF_DEFAULT_BOUNDED_TASK_QUEUE_LOG_SIZE 
  /**
  @def TF_DEFAULT_BOUNDED_TASK_QUEUE_LOG_SIZE
  
  This macro defines the default size of the bounded task queue in Log2. 
  Bounded task queue is used by each worker.
  By default, the value is set to 8, allowing the queue to hold 256 tasks.
  */
  #define TF_DEFAULT_BOUNDED_TASK_QUEUE_LOG_SIZE 8
#endif

#ifndef TF_DEFAULT_UNBOUNDED_TASK_QUEUE_LOG_SIZE 
  /**
  @def TF_DEFAULT_UNBOUNDED_TASK_QUEUE_LOG_SIZE
  
  This macro defines the default size of the unbounded task queue in Log2.
  Unbounded task queue is used by the executor as an overflow region.
  By default, the value is set to 10, allowing the queue to hold 1024 tasks initially.
  */
  #define TF_DEFAULT_UNBOUNDED_TASK_QUEUE_LOG_SIZE 10
#endif

namespace tf {

// ----------------------------------------------------------------------------
// Work-stealing queue steal protocol sentinels
//
// These free functions define the two sentinel values used by steal operations
// across all work-stealing queue types (BoundedWSQ, UnboundedWSQ). They encode
// the result of a steal attempt into the return value itself, avoiding any
// out-parameter or separate status type.
//
// For pointer types T:
//   wsq_empty_value<T>()     = nullptr  — queue was genuinely empty
//   wsq_contended_value<T>() = 0x1      — queue had work but CAS was lost to
//                                         another thief; caller should retry
//
// The sentinel 0x1 is safe because any real object pointer is aligned to at
// least alignof(T) >= 1, and the OS never maps address 0x1. For void* there
// is no pointee alignment to check, but the same reasoning applies — no
// allocator ever returns address 0x1.
//
// For non-pointer types T, both return std::nullopt since sentinel encoding
// is not possible without a dedicated out-of-band value.
//
// Both queue classes expose these as static member functions (empty_value,
// contended_value) that delegate here, so callers can use either form.
// ----------------------------------------------------------------------------

/**
@brief returns the empty sentinel for work-stealing steal operations

For pointer types @c T, returns @c nullptr. For non-pointer types, returns
@c std::nullopt. A steal operation returning this value means the queue was
genuinely empty at the time of the attempt.
*/
template <typename T>
constexpr auto wsq_empty_value() {
  if constexpr (std::is_pointer_v<T>) {
    return T{nullptr};
  } else {
    return std::optional<T>{std::nullopt};
  }
}

/**
@brief returns the contended sentinel for work-stealing steal operations

For pointer types @c T, returns @c reinterpret_cast<T>(uintptr_t{1}), i.e.,
the pointer address @c 0x1. A steal operation returning this value means the
queue was non-empty but the CAS was lost to another concurrent thief — the
caller should retry the same victim since work is known to exist.

For non-pointer types, returns @c std::nullopt (same as @c wsq_empty_value)
since sentinel encoding is not possible without a dedicated out-of-band value.
*/
template <typename T>
auto wsq_contended_value() {
  if constexpr (std::is_pointer_v<T>) {
    return reinterpret_cast<T>(uintptr_t{1});
  } else {
    return std::optional<T>{std::nullopt};
  }
}

// ----------------------------------------------------------------------------
// Unbounded Work-stealing Queue (WSQ)
// ----------------------------------------------------------------------------


/**
@class: UnboundedWSQ

@tparam T data type (must be a pointer type)

@brief class to create a lock-free unbounded work-stealing queue

This class implements the work-stealing queue described in the paper,
<a href="https://www.di.ens.fr/~zappa/readings/ppopp13.pdf">
Correct and Efficient Work-Stealing for Weak Memory Models</a>.

A work-stealing queue supports a single owner thread that performs push and pop
operations, while multiple concurrent thief threads may steal tasks
from the opposite end of the queue. The implementation is designed to
operate correctly under weak memory models and uses atomic operations
with carefully chosen memory orderings to ensure correctness and
scalability.

@dotfile images/unbounded_wsq.dot

Unlike bounded queues, this queue automatically grows its internal
storage as needed, allowing it to accommodate an arbitrary number of
tasks without a fixed capacity limit.
*/
template <typename T>
class UnboundedWSQ {

  struct Array {

    size_t C;
    size_t M;
    std::atomic<T>* S;

    explicit Array(size_t c) :
      C {c},
      M {c-1},
      S {new std::atomic<T>[C]} {
    }

    ~Array() {
      delete [] S;
    }

    size_t capacity() const noexcept {
      return C;
    }

    void push(int64_t i, T o) noexcept {
      S[i & M].store(o, std::memory_order_relaxed);
    }

    T pop(int64_t i) noexcept {
      return S[i & M].load(std::memory_order_relaxed);
    }

    Array* resize(int64_t b, int64_t t) {
      Array* ptr = new Array(2*C);
      for(int64_t i=t; i!=b; ++i) {
        ptr->push(i, pop(i));
      }
      return ptr;
    }

    Array* resize(int64_t b, int64_t t, size_t N) {
      // assert(N>0);
      Array* ptr = new Array(std::bit_ceil(C + N));
      for(int64_t i=t; i!=b; ++i) {
        ptr->push(i, pop(i));
      }
      return ptr;
    }

  };

  alignas(TF_CACHELINE_SIZE) std::atomic<int64_t> _top;
  alignas(TF_CACHELINE_SIZE) std::atomic<int64_t> _bottom;
  std::atomic<Array*> _array;
  std::vector<Array*> _garbage;

  public:
  
  /**
  @brief the return type of queue operations
  
  `value_type` represents the type returned by `pop` and `steal` operations.
  For pointer element types `T`, it is `T` itself and uses `nullptr` to
  indicate an empty result. For non-pointer types, it is `std::optional<T>`,
  where `std::nullopt` denotes the absence of a value.

  @code{.cpp}
  static_assert(std::is_same_v<tf::UnboundedWSQ<int>::value_type, std::optional<int>>);
  static_assert(std::is_same_v<tf::UnboundedWSQ<int*>::value_type, nullptr);
  @endcode
  
  This design avoids the overhead of `std::optional` for pointer types
  while providing a uniform empty-result semantics.
  */
  using value_type = std::conditional_t<std::is_pointer_v<T>, T, std::optional<T>>;

  /**
  @brief constructs the queue with the given size in the base-2 logarithm

  @param LogSize the base-2 logarithm of the queue size

  @code{.cpp}
  tf::UnboundedWSQ<int> wsq(10);  
  assert(wsq.capacity() == 1024);
  @endcode
  */
  explicit UnboundedWSQ(int64_t LogSize = TF_DEFAULT_UNBOUNDED_TASK_QUEUE_LOG_SIZE);

  /**
  @brief destructs the queue
  */
  ~UnboundedWSQ();

  /**
  @brief queries if the queue is empty at the time of this call
  
  @code{.cpp}
  tf::UnboundedWSQ<int> wsq(10);  
  assert(wsq.empty() == true);
  wsq.push(1);
  assert(wsq.empty() == false);
  @endcode
  */
  bool empty() const noexcept;

  /**
  @brief queries the number of items at the time of this call
  
  @code{.cpp}
  tf::UnboundedWSQ<int> wsq(10);  
  assert(wsq.size() == 0);
  wsq.push(1);
  assert(wsq.size() == 1);
  @endcode
  */
  size_t size() const noexcept;

  /**
  @brief queries the capacity of the queue
  
  @code{.cpp}
  tf::UnboundedWSQ<int> wsq(10);  
  assert(wsq.capacity() == 1024);
  for(int i=0; i<1025; i++){  // insert more than 1024 ints to trigger resizing
    wsq.push(i);
  }
  assert(wsq.capacity() == 2048);
  assert(wsq.size() == 1025);
  @endcode
  */
  size_t capacity() const noexcept;
  
  /**
  @brief inserts an item to the queue

  @param item the item to push to the queue
  
  This method pushes one item into the queue.
  The operation can trigger the queue to resize its capacity if more space is required.
  
  @code{.cpp}
  tf::UnboundedWSQ<int> wsq(10);  
  assert(wsq.capacity() == 1024);
  for(int i=0; i<1025; i++) {   // insert more than 1024 items to trigger resizing
    wsq.push(i);
  }
  assert(wsq.capacity() == 2048);
  assert(wsq.size() == 1025);
  @endcode
  
  Only the owner thread can insert an item to the queue.
  */
  void push(T item);
  
  /**
  @brief tries to insert a batch of items into the queue

  @tparam I input iterator type
  @param first iterator to the first item in the batch
  @param N number of items to insert beginning at @p first

  This method pushes up to @p N items from the range `[first, first + N)` into the queue. 
  The operation can trigger the queue to resize its capacity 
  if more space is required.
  The iterator `first` is updated in place and will point to the next uninserted element after the call.

  Bulk insertion is often faster than inserting elements one by one because it requires fewer atomic operations.

  @code{.cpp}
  tf::UnboundedWSQ<int> wsq(10);  
  assert(wsq.capacity() == 1024);
  std::vector<int> vec(1025, 1);
  auto first = vec.data();
  wsq.bulk_push(first, vec.size()); 
  assert(wsq.capacity() == 2048);
  assert(wsq.size() == vec.size());
  assert(std::distance(vec.begin(), first) == 1025);
  @endcode
  
  Only the owner thread can insert an item to the queue.
  */
  template <typename I>
  void bulk_push(I& first, size_t N);

  /**
  @brief pops out an item from the queue

  This method pops an item from the queue.
  If the queue is empty, empty_value() is returned.
  The elements popped out from the queue follow a last-in-first-out (LIFO) order.
  
  @code{.cpp}
  tf::UnboundedWSQ<int> wsq(10);  
  wsq.push(1);
  wsq.push(2);
  wsq.push(3);
  assert(wsq.pop().value() = 3);
  assert(wsq.pop().value() = 2);
  assert(wsq.pop().value() = 1);
  assert(wsq.pop() == std::nullopt);
  @endcode
  
  Only the owner thread can pop out an item from the queue.
  */
  value_type pop();

  /**
  @brief steals an item from the queue

  Any threads can try to steal an item from the queue.
  The return can be an empty_value() if this operation failed.
  The elements stolen from the queue follow a first-in-first-out (FIFO) order.
  
  @code{.cpp}
  tf::UnboundedWSQ<int> wsq(10);  
  wsq.push(1);
  wsq.push(2);
  wsq.push(3);
  assert(wsq.steal().value() = 1);
  assert(wsq.steal().value() = 2);
  assert(wsq.steal().value() = 3);
  assert(wsq.steal() == std::nullopt);
  @endcode

  Multiple threads can simultaneously steal items from the queue.
  */
  value_type steal();

  /**
  @brief attempts to steal an item from the queue with three-state feedback

  @return one of three sentinel-encoded values (for pointer types @c T):
    - a valid pointer: item successfully stolen (CAS succeeded, queue was non-empty)
    - @c contended_value(): queue was non-empty but the CAS was lost to another
                            concurrent thief; the caller should retry the same
                            victim immediately since work is known to exist
    - @c empty_value() (nullptr): queue was genuinely empty; the caller should
                                  move on to a different victim

  For non-pointer types @c T, the return is `std::optional<T>` and only two
  states are possible (value present or @c std::nullopt); contention cannot
  be distinguished from emptiness and the method behaves identically to @c steal.

  @par Sentinel safety for pointer types

  The contended sentinel is `reinterpret_cast<T>(uintptr_t{1})`, i.e., the
  pointer value @c 0x1. This is guaranteed never to be a valid object pointer
  because the C++ standard and every major platform ABI require that any live
  object is aligned to at least @c alignof(T) bytes. For pointer element types
  used with this queue (e.g., @c Node*), @c alignof(Node) is at least 8 (and
  in Taskflow's case 64, due to @c alignas(TF_CACHELINE_SIZE) on @c Node
  members). Therefore the low bits of any real pointer are always zero, making
  @c 0x1 an impossible address. For @c void* there is no pointee type to check,
  but the same reasoning applies — no allocator ever returns address @c 0x1.

  @par Caller usage pattern

  @code{.cpp}
  tf::UnboundedWSQ<Node*> wsq;
  Node* result = wsq.steal_with_feedback();

  if(result == wsq.empty_value()) {
    // queue was empty — move on to another victim
  }
  else if(result == wsq.contended_value()) {
    // lost CAS race — queue has work, retry this victim
  }
  else {
    // result is a valid stolen item — use it
  }
  @endcode

  Multiple threads can simultaneously call this method.
  */
  value_type steal_with_feedback();
  
  /**
  @brief returns the empty sentinel value for the queue element type
  
  This function provides a type-appropriate empty value used to indicate
  that a pop or steal operation failed. For pointer types, the empty value
  is `nullptr` of type `T`; for non-pointer types, it is `std::nullopt` of type `std::optional<T>`.
  
  The function is implemented as a `constexpr` helper to avoid additional
  storage, runtime overhead, or code duplication across queue operations.
  
  @return an empty `value_type` representing the absence of an element.
  */
  static constexpr auto empty_value()     { return wsq_empty_value<T>();     }

  /**
  @brief returns the contended sentinel value for pointer element types

  When @c steal_with_feedback returns this value, the queue was non-empty
  but the CAS was lost to another concurrent thief. The caller should retry
  the same victim immediately since work is known to exist.

  Delegates to @c wsq_contended_value<T>(). See that function for a full
  explanation of the sentinel value and its safety guarantees.
  */
  static auto contended_value()           { return wsq_contended_value<T>(); }

  private:

  Array* _resize_array(Array* a, int64_t b, int64_t t);
  Array* _resize_array(Array* a, int64_t b, int64_t t, size_t N);
};

// Constructor
template <typename T>
UnboundedWSQ<T>::UnboundedWSQ(int64_t LogSize) {
  _top.store(0, std::memory_order_relaxed);
  _bottom.store(0, std::memory_order_relaxed);
  _array.store(new Array{(size_t{1} << LogSize)}, std::memory_order_relaxed);
  _garbage.reserve(32);
}

// Destructor
template <typename T>
UnboundedWSQ<T>::~UnboundedWSQ() {
  for(auto a : _garbage) {
    delete a;
  }
  delete _array.load();
}

// Function: empty
template <typename T>
bool UnboundedWSQ<T>::empty() const noexcept {
  int64_t t = _top.load(std::memory_order_relaxed);
  int64_t b = _bottom.load(std::memory_order_relaxed);
  return (b <= t);
}

// Function: size
template <typename T>
size_t UnboundedWSQ<T>::size() const noexcept {
  int64_t t = _top.load(std::memory_order_relaxed);
  int64_t b = _bottom.load(std::memory_order_relaxed);
  return static_cast<size_t>(b >= t ? b - t : 0);
}

// Function: push
template <typename T>
void UnboundedWSQ<T>::push(T o) {

  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_acquire);
  Array* a = _array.load(std::memory_order_relaxed);

  // queue is full with one additional item (b-t+1)
  if (a->capacity() < static_cast<size_t>(b - t + 1)) [[unlikely]] {
    a = _resize_array(a, b, t);
  }

  a->push(b, o);
  std::atomic_thread_fence(std::memory_order_release);

  // original paper uses relaxed here but tsa complains
  _bottom.store(b + 1, std::memory_order_release);
}

// Function: bulk_push
template <typename T>
template <typename I>
void UnboundedWSQ<T>::bulk_push(I& first, size_t N) {

  if(N == 0) return;

  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_acquire);
  Array* a = _array.load(std::memory_order_relaxed);

  // queue is full with N additional items
  if ( (b - t + N) > a->capacity() ) [[unlikely]] {
    a = _resize_array(a, b, t, N);
  }

  for(size_t i=0; i<N; ++i) {
    a->push(b++, *first++);
  }
  std::atomic_thread_fence(std::memory_order_release);

  // original paper uses relaxed here but tsa complains
  _bottom.store(b, std::memory_order_release);
}

// Function: pop
template <typename T>
typename UnboundedWSQ<T>::value_type 
UnboundedWSQ<T>::pop() {

  int64_t b = _bottom.load(std::memory_order_relaxed) - 1;
  Array* a = _array.load(std::memory_order_relaxed);
  _bottom.store(b, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t t = _top.load(std::memory_order_relaxed);

  //T item {nullptr};
  auto item = empty_value();

  if(t <= b) {
    item = a->pop(b);
    if(t == b) {
      // the last item just got stolen
      if(!_top.compare_exchange_strong(t, t+1, std::memory_order_seq_cst,
                                               std::memory_order_relaxed)) {
        //item = nullptr;
        item = empty_value();
      }
      _bottom.store(b + 1, std::memory_order_relaxed);
    }
  }
  else {
    _bottom.store(b + 1, std::memory_order_relaxed);
  }

  return item;
}

// Function: steal
template <typename T>
typename UnboundedWSQ<T>::value_type 
UnboundedWSQ<T>::steal() {
  
  int64_t t = _top.load(std::memory_order_acquire);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = _bottom.load(std::memory_order_acquire);

  //T item {nullptr};
  auto item = empty_value();

  if(t < b) {
    Array* a = _array.load(std::memory_order_consume);
    item = a->pop(t);
    if(!_top.compare_exchange_strong(t, t+1,
                                     std::memory_order_seq_cst,
                                     std::memory_order_relaxed)) {
      //return nullptr;
      return empty_value();
    }
  }

  return item;
}

// Function: steal_with_feedback
// Returns a stolen item, contended_value(), or empty_value() — see declaration.
template <typename T>
typename UnboundedWSQ<T>::value_type
UnboundedWSQ<T>::steal_with_feedback() {

  int64_t t = _top.load(std::memory_order_acquire);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = _bottom.load(std::memory_order_acquire);

  if(t < b) {
    // queue is non-empty: load the candidate item and attempt the CAS
    Array* a = _array.load(std::memory_order_consume);
    auto item = a->pop(t);
    if(!_top.compare_exchange_strong(t, t+1,
                                     std::memory_order_seq_cst,
                                     std::memory_order_relaxed)) {
      // CAS lost to another thief — queue had work but we didn't get it.
      // Return contended_value() so the caller knows to retry this victim.
      return contended_value();
    }
    return item;
  }

  // bottom <= top: queue is genuinely empty
  return empty_value();
}

// Function: capacity
template <typename T>
size_t UnboundedWSQ<T>::capacity() const noexcept {
  return _array.load(std::memory_order_relaxed)->capacity();
}

template <typename T>
typename UnboundedWSQ<T>::Array*
UnboundedWSQ<T>::_resize_array(Array* a, int64_t b, int64_t t) {
  Array* tmp = a->resize(b, t);
  _garbage.push_back(a);
  // Note: the original paper using relaxed causes t-san to complain
  _array.store(tmp, std::memory_order_release);
  return tmp;
}

template <typename T>
typename UnboundedWSQ<T>::Array*
UnboundedWSQ<T>::_resize_array(Array* a, int64_t b, int64_t t, size_t N) {
  Array* tmp = a->resize(b, t, N);
  _garbage.push_back(a);
  // Note: the original paper using relaxed causes t-san to complain
  _array.store(tmp, std::memory_order_release);
  return tmp;
}

// ----------------------------------------------------------------------------
// Bounded Work-stealing Queue (WSQ)
// ----------------------------------------------------------------------------

/**
@class: BoundedWSQ

@tparam T data type
@tparam LogSize the base-2 logarithm of the queue size

@brief class to create a lock-free bounded work-stealing queue

This class implements the work-stealing queue described in the paper,
<a href="https://www.di.ens.fr/~zappa/readings/ppopp13.pdf">
Correct and Efficient Work-Stealing for Weak Memory Models</a>.

A work-stealing queue supports a single owner thread that performs push and pop
operations, while multiple concurrent thief threads may steal tasks
from the opposite end of the queue. The implementation is designed to
operate correctly under weak memory models and uses atomic operations
with carefully chosen memory orderings to ensure correctness and
scalability.

@dotfile images/bounded_wsq.dot

The queue has a fixed capacity determined at construction time and
does not grow dynamically. When the queue is full, push operations
may fail or require external handling.
*/
template <typename T, size_t LogSize = TF_DEFAULT_BOUNDED_TASK_QUEUE_LOG_SIZE>
class BoundedWSQ {
  
  constexpr static size_t BufferSize = size_t{1} << LogSize;
  constexpr static size_t BufferMask = (BufferSize - 1);

  static_assert((BufferSize >= 2) && ((BufferSize & (BufferSize - 1)) == 0));

  alignas(TF_CACHELINE_SIZE) std::atomic<int64_t> _top {0};
  alignas(TF_CACHELINE_SIZE) std::atomic<int64_t> _bottom {0};
  alignas(TF_CACHELINE_SIZE) std::atomic<T> _buffer[BufferSize];

  public:
  
  /**
  @brief the return type of queue operations
  
  `value_type` represents the type returned by `pop` and `steal` operations.
  For pointer element types `T`, it is `T` itself and uses `nullptr` to
  indicate an empty result. For non-pointer types, it is `std::optional<T>`,
  where `std::nullopt` denotes the absence of a value.
  
  @code{.cpp}
  static_assert(std::is_same_v<tf::UnboundedWSQ<int>::value_type, std::optional<int>>);
  static_assert(std::is_same_v<tf::UnboundedWSQ<int*>::value_type, nullptr);
  @endcode
  
  This design avoids the overhead of `std::optional` for pointer types
  while providing a uniform empty-result semantics.
  */
  using value_type = std::conditional_t<std::is_pointer_v<T>, T, std::optional<T>>;
    
  /**
  @brief constructs the queue with a given capacity
  
  @code{.cpp}
  tf::BoundedWSQ<int, 10> wsq;  
  static_assert(wsq.capacity() == 1024);
  @endcode
  */
  BoundedWSQ() = default;

  /**
  @brief destructs the queue
  */
  ~BoundedWSQ() = default;
  
  /**
  @brief queries if the queue is empty at the time of this call
  
  @code{.cpp}
  tf::BoundedWSQ<int, 10> wsq;  
  assert(wsq.empty() == true);
  wsq.push(1);
  assert(wsq.empty() == false);
  @endcode
  */
  bool empty() const noexcept;
  
  /**
  @brief queries the number of items at the time of this call
  
  @code{.cpp}
  tf::BoundedWSQ<int, 10> wsq;  
  assert(wsq.size() == 0);
  wsq.push(1);
  assert(wsq.size() == 1);
  @endcode
  */
  size_t size() const noexcept;

  /**
  @brief queries the capacity of the queue
  
  The capacity of a bounded work-stealing queue is decided at compile time.

  @code{.cpp}
  tf::BoundedWSQ<int, 10> wsq;
  static_assert(wsq.capacity() == 1024);
  @endcode
  */
  constexpr size_t capacity() const;
  
  /**
  @brief tries to insert an item to the queue

  @tparam O data type 
  @param item the item to perfect-forward to the queue
  @return `true` if the insertion succeed or `false` (queue is full)
  
  This method attempts to push one item into the queue.
  If the operation succeed, it returns `true` or `false` otherwise.
  
  @code{.cpp}
  tf::BoundedWSQ<int, 10> wsq;  
  static_assert(wsq.capacity() == 1024);
  for(int i=0; i<1024; i++) {
    assert(wsq.try_push(i) == true);
  }
  assert(wsq.size() == 1024);
  assert(wsq.try_push(0) == false);
  @endcode

  Only the owner thread can insert an item to the queue. 
  */
  template <typename O>
  bool try_push(O&& item);
  
  /**
  @brief tries to insert a batch of items into the queue

  @tparam I input iterator type
  @param first iterator to the first item in the batch
  @param N number of items to insert beginning at @p first
  @return the number of items successfully inserted

  This method attempts to push up to @p N items from the range
  `[first, first + N)` into the queue. Insertion stops early if the
  queue becomes full. 
  The iterator `first` is updated in place and will point to the next uninserted element after the call.

  Bulk insertion is often faster than inserting elements one by one because it requires fewer atomic operations.
  
  @code{.cpp}
  tf::BoundedWSQ<int, 10> wsq;  
  static_assert(wsq.capacity() == 1024);
  std::vector<int> vec(1030, 1);
  auto first = vec.begin();
  assert(wsq.try_bulk_push(first, vec.size()) == wsq.capacity());
  assert(std::distance(vec.begin(), first) == wsq.capacity());
  @endcode

  Only the owner thread can insert items into the queue.
  */
  template <typename I>
  size_t try_bulk_push(I& first, size_t N);
  
  /**
  @brief pops out an item from the queue

  The method pops an item out of the queue based on a last-in-first-out (LIFO) order.
  The return can be an empty_value() if this operation failed (empty queue).

  @code{.cpp}
  tf::BoundedWSQ<int, 10> wsq;  
  wsq.push(1);
  wsq.push(2);
  wsq.push(3);
  assert(wsq.pop().value() = 3);
  assert(wsq.pop().value() = 2);
  assert(wsq.pop().value() = 1);
  assert(wsq.pop() == std::nullopt);
  @endcode

  Only the owner thread can pop out an item from the queue. 
  */
  value_type pop();
  
  /**
  @brief steals an item from the queue

  Any threads can try to steal an item from the queue.
  The return can be an empty_value() if this operation failed.
  The elements stolen from the queue follow a first-in-first-out (FIFO) order.
  
  @code{.cpp}
  tf::BoundedWSQ<int, 10> wsq;  
  wsq.push(1);
  wsq.push(2);
  wsq.push(3);
  assert(wsq.steal().value() = 1);
  assert(wsq.steal().value() = 2);
  assert(wsq.steal().value() = 3);
  assert(wsq.steal() == std::nullopt);
  @endcode
  
  Multiple threads can simultaneously steal items from the queue.
  */
  value_type steal();

  /**
  @brief attempts to steal an item from the queue with three-state feedback

  @return one of three sentinel-encoded values (for pointer types @c T):
    - a valid pointer: item successfully stolen (CAS succeeded, queue was non-empty)
    - @c contended_value(): queue was non-empty but the CAS was lost to another
                            concurrent thief; the caller should retry the same
                            victim immediately since work is known to exist
    - @c empty_value() (nullptr): queue was genuinely empty; the caller should
                                  move on to a different victim

  For non-pointer types @c T, the return is `std::optional<T>` and only two
  states are possible (value present or @c std::nullopt); contention cannot
  be distinguished from emptiness and the method behaves identically to @c steal.

  @par Sentinel safety for pointer types

  The contended sentinel is `reinterpret_cast<T>(uintptr_t{1})`, i.e., the
  pointer value @c 0x1. This is guaranteed never to be a valid object pointer
  because the C++ standard and every major platform ABI require that any live
  object is aligned to at least @c alignof(T) bytes. For pointer element types
  used with this queue (e.g., @c Node*), @c alignof(Node) is at least 8 (and
  in Taskflow's case 64, due to @c alignas(TF_CACHELINE_SIZE) on @c Node
  members). Therefore the low bits of any real pointer are always zero, making
  @c 0x1 an impossible address. For @c void* there is no pointee type to check,
  but the same reasoning applies — no allocator ever returns address @c 0x1.

  @par Caller usage pattern

  @code{.cpp}
  tf::BoundedWSQ<Node*> wsq;
  Node* result = wsq.steal_with_feedback();

  if(result == wsq.empty_value()) {
    // queue was empty — move on to another victim
  }
  else if(result == wsq.contended_value()) {
    // lost CAS race — queue has work, retry this victim
  }
  else {
    // result is a valid stolen item — use it
  }
  @endcode

  Multiple threads can simultaneously call this method.
  */
  value_type steal_with_feedback();

  /**
  @brief returns the empty sentinel value for the queue element type
  
  This function provides a type-appropriate empty value used to indicate
  that a pop or steal operation failed. For pointer types, the empty value
  is `nullptr` of type `T`; for non-pointer types, it is `std::nullopt` of type `std::optional<T>`.
  
  The function is implemented as a `constexpr` helper to avoid additional
  storage, runtime overhead, or code duplication across queue operations.
  
  @return an empty `value_type` representing the absence of an element.
  */
  static constexpr auto empty_value()     { return wsq_empty_value<T>();     }

  /**
  @brief returns the contended sentinel value for pointer element types

  When @c steal_with_feedback returns this value, the queue was non-empty
  but the CAS was lost to another concurrent thief. The caller should retry
  the same victim immediately since work is known to exist.

  Delegates to @c wsq_contended_value<T>(). See that function for a full
  explanation of the sentinel value and its safety guarantees.
  */
  static auto contended_value()           { return wsq_contended_value<T>(); }
};

// Function: empty
template <typename T, size_t LogSize>
bool BoundedWSQ<T, LogSize>::empty() const noexcept {
  int64_t t = _top.load(std::memory_order_relaxed);
  int64_t b = _bottom.load(std::memory_order_relaxed);
  return b <= t;
}

// Function: size
template <typename T, size_t LogSize>
size_t BoundedWSQ<T, LogSize>::size() const noexcept {
  int64_t t = _top.load(std::memory_order_relaxed);
  int64_t b = _bottom.load(std::memory_order_relaxed);
  return static_cast<size_t>(b >= t ? b - t : 0);
}

// Function: try_push
template <typename T, size_t LogSize>
template <typename O>
bool BoundedWSQ<T, LogSize>::try_push(O&& o) {

  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_acquire);

  // queue is full with one additional item (b-t+1)
  if(static_cast<size_t>(b - t + 1) > BufferSize) [[unlikely]] {
    return false;
  }
  
  _buffer[b & BufferMask].store(std::forward<O>(o), std::memory_order_relaxed);

  std::atomic_thread_fence(std::memory_order_release);
  
  // original paper uses relaxed here but tsa complains
  _bottom.store(b + 1, std::memory_order_release);

  return true;
}

// Function: try_bulk_push
template <typename T, size_t LogSize>
template <typename I>
size_t BoundedWSQ<T, LogSize>::try_bulk_push(I& first, size_t N) {

  if(N == 0) return 0;

  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_acquire);

  size_t r = BufferSize - (b - t);  // remaining capacity
  size_t n = std::min(N, r);        // number of pushable elements

  if(n > 0) {
    // push n elements into the queue
    for(size_t i=0; i<n; ++i) {
      _buffer[b++ & BufferMask].store(*first++, std::memory_order_relaxed);
    }
    std::atomic_thread_fence(std::memory_order_release);
    // original paper uses relaxed here but tsa complains
    _bottom.store(b, std::memory_order_release);
  }
  
  return n;
}

// Function: pop
template <typename T, size_t LogSize>
typename BoundedWSQ<T, LogSize>::value_type 
BoundedWSQ<T, LogSize>::pop() {

  int64_t b = _bottom.load(std::memory_order_relaxed) - 1;
  _bottom.store(b, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t t = _top.load(std::memory_order_relaxed);

  //T item {nullptr};
  auto item = empty_value();

  if(t <= b) {
    item = _buffer[b & BufferMask].load(std::memory_order_relaxed);
    if(t == b) {
      // the last item just got stolen
      if(!_top.compare_exchange_strong(t, t+1, 
                                       std::memory_order_seq_cst, 
                                       std::memory_order_relaxed)) {
        //item = nullptr;
        item = empty_value();
      }
      _bottom.store(b + 1, std::memory_order_relaxed);
    }
  }
  else {
    _bottom.store(b + 1, std::memory_order_relaxed);
  }

  return item;
}

// Function: steal
template <typename T, size_t LogSize>
typename BoundedWSQ<T, LogSize>::value_type 
BoundedWSQ<T, LogSize>::steal() {
  int64_t t = _top.load(std::memory_order_acquire);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = _bottom.load(std::memory_order_acquire);
  
  //T item{nullptr};
  auto item = empty_value();

  if(t < b) {
    item = _buffer[t & BufferMask].load(std::memory_order_relaxed);
    if(!_top.compare_exchange_strong(t, t+1,
                                     std::memory_order_seq_cst,
                                     std::memory_order_relaxed)) {
      //return nullptr;
      return empty_value();
    }
  }

  return item;
}

// Function: steal_with_feedback
// Returns a stolen item, contended_value(), or empty_value() — see declaration.
template <typename T, size_t LogSize>
typename BoundedWSQ<T, LogSize>::value_type
BoundedWSQ<T, LogSize>::steal_with_feedback() {

  int64_t t = _top.load(std::memory_order_acquire);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = _bottom.load(std::memory_order_acquire);

  if(t < b) {
    // queue is non-empty: load the candidate item and attempt the CAS
    auto item = _buffer[t & BufferMask].load(std::memory_order_relaxed);
    if(!_top.compare_exchange_strong(t, t+1,
                                     std::memory_order_seq_cst,
                                     std::memory_order_relaxed)) {
      // CAS lost to another thief — queue had work but we didn't get it.
      // Return contended_value() so the caller knows to retry this victim.
      return contended_value();
    }
    return item;
  }

  // bottom <= top: queue is genuinely empty
  return empty_value();
}

// Function: capacity
template <typename T, size_t LogSize>
constexpr size_t BoundedWSQ<T, LogSize>::capacity() const {
  return BufferSize;
}


}  // end of namespace tf -----------------------------------------------------
