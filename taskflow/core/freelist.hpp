#pragma once

#include "../utility/os.hpp"
#include <atomic>

namespace tf {

/**
@file freelist.hpp
@brief atomic intrusive stack include file
*/

/**
@class AtomicIntrusiveStack

@tparam T         pointer type of the node (e.g., @c Node*)
@tparam MemberPtr pointer-to-member that identifies the intrusive link field
                  within the node type (e.g., @c &Node::next)

@brief class to create a lock-free, ABA-safe intrusive stack

An intrusive stack embeds the link pointer directly inside each node rather
than allocating separate list nodes. The caller selects which field serves as
the link by supplying @c MemberPtr. While a node is held in the stack that
field is exclusively owned by the stack and must not be accessed by the caller.

The field identified by @c MemberPtr must be of type @c T (a pointer to the
same node type) so the stack can chain nodes through it:

@code{.cpp}
struct Node {
  Node* next {nullptr};  // used as the intrusive link
  int   value{0};
};
tf::AtomicIntrusiveStack<Node*, &Node::next> stack;
@endcode

All methods are safe to call concurrently from multiple threads.
@c push and @c bulk_push are lock-free. @c pop is lock-free.

@par ABA safety

Each CAS operation pairs the pointer with a monotonically-incrementing version
tag stored in a 128-bit @c TaggedPointer atomic. This requires
@c CMPXCHG16B support on x86-64. A 64-bit tagged-pointer alternative for
platforms without 128-bit CAS is available in the commented-out section at
the bottom of this file.


*/
template <typename T, auto MemberPtr>
class AtomicIntrusiveStack {

  // NodeType is the pointee (e.g. Node from Node*)
  using NodeType = std::remove_pointer_t<T>;

  static_assert(
    std::is_same_v<T, std::remove_reference_t<
      decltype(std::declval<NodeType>().*MemberPtr)
    >>,
    "MemberPtr must point to a field of type T within the node"
  );

  // helper: access the intrusive link field of a node
  static T& _link(T node) { return node->*MemberPtr; }

  struct TaggedPointer {
    T      ptr;
    size_t tag;
  };

  alignas(TF_CACHELINE_SIZE) std::atomic<TaggedPointer> _head;

  public:

  /**
  @brief constructs an empty stack

  @code{.cpp}
  struct Node { Node* next {nullptr}; };
  tf::AtomicIntrusiveStack<Node*, &Node::next> stack;
  assert(stack.empty() == true);
  @endcode
  */
  AtomicIntrusiveStack() {
    _head.store({nullptr, 0}, std::memory_order_relaxed);
  }

  /**
  @brief queries whether the stack is empty at the time of this call

  The result is a snapshot and may be stale by the time the caller acts on
  it in a concurrent setting.

  @code{.cpp}
  struct Node { Node* next {nullptr}; };
  tf::AtomicIntrusiveStack<Node*, &Node::next> stack;

  assert(stack.empty() == true);
  Node n;
  stack.push(&n);
  assert(stack.empty() == false);
  stack.pop();
  assert(stack.empty() == true);
  @endcode
  */
  bool empty() const {
    return _head.load(std::memory_order_relaxed).ptr == nullptr;
  }

  /**
  @brief pushes a node onto the top of the stack

  @param node pointer to the node to push; must not be @c nullptr

  Sets the node's link field to the current head and atomically swaps the
  head to @c node. The version tag is incremented on every successful CAS
  to prevent ABA. Safe to call concurrently with other @c push, @c bulk_push,
  and @c pop calls.

  @code{.cpp}
  struct Node { Node* next {nullptr}; int value {0}; };
  tf::AtomicIntrusiveStack<Node*, &Node::next> stack;

  Node a{nullptr, 1}, b{nullptr, 2};
  stack.push(&a);
  stack.push(&b);
  // stack order top-to-bottom: b -> a
  assert(stack.pop() == &b);
  assert(stack.pop() == &a);
  assert(stack.pop() == nullptr);
  @endcode
  */
  void push(T node) {

    TaggedPointer curr = _head.load(std::memory_order_relaxed);

    while(true) {
      _link(node) = curr.ptr;
      TaggedPointer next = {node, curr.tag + 1};

      if(_head.compare_exchange_weak(curr, next,
                                     std::memory_order_release,
                                     std::memory_order_relaxed)) {
        break;
      }
      // on failure curr is reloaded by compare_exchange_weak
    }
  }

  /**
  @brief removes and returns the top node, or @c nullptr if the stack is empty

  @return pointer to the popped node, or @c nullptr if the stack was empty

  Atomically swaps the head to the next node in the chain. The popped node's
  link field is cleared to @c nullptr before it is returned, so the caller
  receives a clean node with no dangling stack pointer. Safe to call
  concurrently with other @c push, @c bulk_push, and @c pop calls.

  @code{.cpp}
  struct Node { Node* next {nullptr}; int value {0}; };
  tf::AtomicIntrusiveStack<Node*, &Node::next> stack;

  // pop from empty stack returns nullptr
  assert(stack.pop() == nullptr);

  Node a{nullptr, 1}, b{nullptr, 2};
  stack.push(&a);
  stack.push(&b);

  Node* n = stack.pop();
  assert(n == &b);

  assert(stack.pop() == &a);
  assert(stack.pop() == nullptr);
  @endcode
  */
  T pop() {

    TaggedPointer curr = _head.load(std::memory_order_acquire);

    while(curr.ptr != nullptr) {
      TaggedPointer next = {_link(curr.ptr), curr.tag + 1};

      if(_head.compare_exchange_weak(curr, next,
                                     std::memory_order_release,
                                     std::memory_order_acquire)) {
        // This can cause thread sanitizer to report error since this assignment
        // has not guarantee to finish before the return while the other thread
        // may call push that modify curr.ptr causing data race...
        return curr.ptr;
      }
      // on failure curr is reloaded
    }
    return nullptr;
  }
};

// ----------------------------------------------------------------------------
// AtomicIntrusiveStack — 64-bit tagged-pointer alternative
//
// Uses the upper 16 bits of a 64-bit uintptr_t as a version tag, avoiding
// the need for 128-bit CAS. Relies on x86-64 canonical address space where
// only the lower 48 bits are used for actual addresses.
//
// Parameterized identically to the 128-bit version above.
// ----------------------------------------------------------------------------

/*
template <typename T, auto MemberPtr>
class AtomicIntrusiveStack {

  using NodeType = std::remove_pointer_t<T>;

  static_assert(
    std::is_same_v<T, std::remove_reference_t<
      decltype(std::declval<NodeType>().*MemberPtr)
    >>,
    "MemberPtr must point to a field of type T within the node"
  );

  static T& _link(T node) { return node->*MemberPtr; }

  static constexpr uintptr_t ADDR_MASK = (uintptr_t{1} << 48) - 1;
  static constexpr uintptr_t TAG_INC   = (uintptr_t{1} << 48);

  T _unpack(uintptr_t val) const {
    return reinterpret_cast<T>(val & ADDR_MASK);
  }

  uintptr_t _pack(T ptr, uintptr_t old_val) const {
    uintptr_t next_tag = (old_val & ~ADDR_MASK) + TAG_INC;
    return next_tag | (reinterpret_cast<uintptr_t>(ptr) & ADDR_MASK);
  }

  alignas(TF_CACHELINE_SIZE) std::atomic<uintptr_t> _head;

public:

  AtomicIntrusiveStack() : _head(0) {}

  bool empty() const {
    return _unpack(_head.load(std::memory_order_relaxed)) == nullptr;
  }

  void push(T node) {
    uintptr_t curr = _head.load(std::memory_order_relaxed);
    while(true) {
      _link(node) = _unpack(curr);
      uintptr_t next = _pack(node, curr);
      if(_head.compare_exchange_weak(curr, next,
                                     std::memory_order_release,
                                     std::memory_order_relaxed)) {
        break;
      }
    }
  }

  template <typename I>
  void bulk_push(I first, size_t N) {
    if(N == 0) return;
    for(size_t i = 0; i < N-1; ++i) {
      _link(first[i]) = first[i+1];
    }
    _link(first[N-1]) = nullptr;
    uintptr_t curr = _head.load(std::memory_order_relaxed);
    while(true) {
      _link(first[N-1]) = _unpack(curr);
      uintptr_t next = _pack(first[0], curr);
      if(_head.compare_exchange_weak(curr, next,
                                     std::memory_order_release,
                                     std::memory_order_relaxed)) {
        break;
      }
    }
  }

  T pop() {
    uintptr_t curr = _head.load(std::memory_order_acquire);
    while(true) {
      T curr_ptr = _unpack(curr);
      if(curr_ptr == nullptr) return nullptr;
      uintptr_t next = _pack(_link(curr_ptr), curr);
      if(_head.compare_exchange_weak(curr, next,
                                     std::memory_order_release,
                                     std::memory_order_acquire)) {
        _link(curr_ptr) = nullptr;
        return curr_ptr;
      }
    }
  }
};
*/

}  // end of namespace tf. ----------------------------------------------------
