#pragma once

namespace tf {

template <typename T>
class IntrusiveForwardList {

  public:

  // --- Iterator Subclass ---
  class Iterator {

    public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = T;
    using pointer           = T*;
    using reference         = T&;

    explicit Iterator(T* ptr) : _ptr(ptr) {}

    reference operator*() const { return *_ptr; }
    pointer operator->() { return _ptr; }

    Iterator& operator++() {
      if (_ptr) _ptr = _ptr->next;
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator==(const Iterator& other) const { return _ptr == other._ptr; }
    bool operator!=(const Iterator& other) const { return _ptr != other._ptr; }

  private:
    T* _ptr;
  };

  IntrusiveForwardList() = default;

  IntrusiveForwardList(const IntrusiveForwardList&) = delete;
  IntrusiveForwardList& operator=(const IntrusiveForwardList&) = delete;

  IntrusiveForwardList(IntrusiveForwardList&& other) noexcept 
    : _head(other._head) {
    other._head = nullptr;
  }

  // 4. Move Assignment
  IntrusiveForwardList& operator=(IntrusiveForwardList&& other) noexcept {
    if (this != &other) {
      // In a real std::forward_list, we might clear() here, 
      // but since we don't own the nodes, we just overwrite.
      _head = other._head;
      other._head = nullptr;
    }
    return *this;
  }

  // Element Access
  T& front() { return *_head; }
  const T& front() const { return *_head; }

  // Iterators
  Iterator begin() { return Iterator(_head); }
  Iterator end()   { return Iterator(nullptr); }

  // Capacity
  bool empty() const { return _head == nullptr; }

  // Modifiers
  void push_front(T* node) {
    node->next = _head;
    _head = node;
  }

  void pop_front() {
    if (_head) {
      T* old_head = _head;
      _head = _head->next;
      old_head->next = nullptr;
    }
  }

  void clear() {
    while (!empty()) pop_front();
  }

  // std::forward_list style insertion after a specific node
  void insert_after(T* prev_node, T* new_node) {
    if (!prev_node || !new_node) return;
    new_node->next = prev_node->next;
    prev_node->next = new_node;
  }

  private:

  T* _head {nullptr};
};

}  // end of namespace tf -------------------------------------------------------------------------
