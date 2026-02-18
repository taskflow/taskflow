#pragma once

namespace tf {

// ----------------------------------------------------------------------------
// Lock-free stack
// ----------------------------------------------------------------------------

template <typename T>
concept AtomicIntrusiveStackValue = requires(std::remove_pointer_t<T> n) { n.next; };

template <AtomicIntrusiveStackValue T>
class AtomicIntrusiveStack {

  struct TaggedPointer {
    T ptr;
    size_t tag;
  };

  alignas(TF_CACHELINE_SIZE) std::atomic<TaggedPointer> head;

  public:

  AtomicIntrusiveStack() {
    head.store({nullptr, 0});
  }

  bool empty() const {
    return head.load(std::memory_order_relaxed).ptr == nullptr;
  }

  void push(T node) {

    TaggedPointer curr_head = head.load(std::memory_order_relaxed);

    while (true) {
      node->next = curr_head.ptr;
      // Create a new tagged pointer with an incremented tag
      TaggedPointer next_head = {node, curr_head.tag + 1};

      if (head.compare_exchange_weak(curr_head, next_head,
                                     std::memory_order_release,
                                     std::memory_order_relaxed)) {
        break;
      }
      // On failure, curr_head is updated with the latest value
    }
  }
  
  template <typename I>
  void bulk_push(I first, size_t N) {

    if(N == 0) {
      return;
    }
    
    for(size_t i=0; i<N-1; ++i) {
      first[i]->next = first[i+1];
    }
    first[N-1]->next = nullptr;

    TaggedPointer curr_head = head.load(std::memory_order_relaxed);

    while (true) {
      first[N-1]->next = curr_head.ptr;
      // Create a new tagged pointer with an incremented tag
      TaggedPointer next_head = {first[0], curr_head.tag + 1};

      if (head.compare_exchange_weak(curr_head, next_head,
                                     std::memory_order_release,
                                     std::memory_order_relaxed)) {
        break;
      }
      // On failure, curr_head is updated with the latest value
    }
  }

  T steal() {

    TaggedPointer curr_head = head.load(std::memory_order_acquire);

    while (curr_head.ptr != nullptr) {
      TaggedPointer next_head = {curr_head.ptr->next, curr_head.tag + 1};

      if (head.compare_exchange_weak(curr_head, next_head,
                                     std::memory_order_release,
                                     std::memory_order_acquire)) {
        curr_head.ptr->next = nullptr;
        return curr_head.ptr;
      }
    }
    return nullptr;
  }
};

/*
template <AtomicIntrusiveStackValue T>
class AtomicIntrusiveStack {

  // Mask for the lower 48 bits (the actual memory address)
  static constexpr uintptr_t ADDR_MASK = (1ULL << 48) - 1;
  // The bit where the tag starts
  static constexpr uintptr_t TAG_INC = (1ULL << 48);

  // Helper to extract the real pointer and strip the tag
  T unpack(uintptr_t val) const {
    return reinterpret_cast<T>(val & ADDR_MASK);
  }

  // Helper to increment the tag and pack a new pointer
  uintptr_t pack(T ptr, uintptr_t old_val) const {
    uintptr_t next_tag = (old_val & ~ADDR_MASK) + TAG_INC;
    return next_tag | (reinterpret_cast<uintptr_t>(ptr) & ADDR_MASK);
  }
  
  // Single 64-bit atomic - guaranteed lock-free on 64-bit systems
  alignas(TF_CACHELINE_SIZE) std::atomic<uintptr_t> head;

public:

  AtomicIntrusiveStack() : head(0) {}

  bool empty() const {
    return unpack(head.load(std::memory_order_relaxed)) == nullptr;
  }

  void push(T node) {
    uintptr_t curr = head.load(std::memory_order_relaxed);
    while (true) {
      node->next = unpack(curr);
      uintptr_t next = pack(node, curr);

      if (head.compare_exchange_weak(curr, next,
                                     std::memory_order_release,
                                     std::memory_order_relaxed)) {
        break;
      }
    }
  }

  template <typename I>
  void bulk_push(I first, size_t N) {

    if (N == 0) {
      return;
    }

    // Connect the nodes locally (no atomics needed here)
    for (size_t i = 0; i < N - 1; ++i) {
      first[i]->next = first[i + 1];
    }
    first[N-1]->next = nullptr;

    uintptr_t curr = head.load(std::memory_order_relaxed);
    while (true) {
      // Point the end of the bulk chain to the current head
      first[N - 1]->next = unpack(curr);
      // Make the start of the chain the new head
      uintptr_t next = pack(first[0], curr);

      if (head.compare_exchange_weak(curr, next,
                                     std::memory_order_release,
                                     std::memory_order_relaxed)) {
        break;
      }
    }
  }

  T steal() {
    uintptr_t curr = head.load(std::memory_order_acquire);
    while (true) {
      auto curr_ptr = unpack(curr);
      if (curr_ptr == nullptr) return nullptr;

      // Danger: ABA would happen here if we didn't have tags!
      auto next_node = curr_ptr->next;
      uintptr_t next = pack(next_node, curr);

      if (head.compare_exchange_weak(curr, next,
                                     std::memory_order_release,
                                     std::memory_order_acquire)) {
        curr_ptr->next = nullptr;
        return curr_ptr;
      }
    }
  }
};
*/

}  // end of namespace tf. ----------------------------------------------------
