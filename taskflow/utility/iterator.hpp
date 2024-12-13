#pragma once

#include <cstddef>
#include <type_traits>

namespace tf {

template <typename B, typename E, typename S>
constexpr std::enable_if_t<std::is_integral<std::decay_t<B>>::value && 
                           std::is_integral<std::decay_t<E>>::value && 
                           std::is_integral<std::decay_t<S>>::value, bool>
is_range_invalid(B beg, E end, S step) {
  return ((step == 0 && beg != end) ||
          (beg < end && step <=  0) ||  // positive range
          (beg > end && step >=  0));   // negative range
}

template <typename B, typename E, typename S>
constexpr std::enable_if_t<std::is_integral<std::decay_t<B>>::value && 
                           std::is_integral<std::decay_t<E>>::value && 
                           std::is_integral<std::decay_t<S>>::value, size_t>
distance(B beg, E end, S step) {
  return (end - beg + step + (step > 0 ? -1 : 1)) / step;
}
  

// ----------------------------------------------------------------------------
// iterator definition
// ----------------------------------------------------------------------------

/**
@brief RandomAccessIterator
*/
template<typename T>
class RandomAccessIterator {

  public:

  using iterator_category = std::random_access_iterator_tag; // RandomAccessIterator category
  using value_type = T;                                     // Value type
  using difference_type = std::ptrdiff_t;                   // Difference type
  using pointer = T*;                                       // Pointer type
  using reference = T&;                                     // Reference type

  // Constructor
  RandomAccessIterator(T* ptr) : _ptr(ptr) {}

  // Dereference operator
  reference operator*() const { return *_ptr; }

  // Pointer access operator
  pointer operator->() const { return _ptr; }

  // Pre-increment
  RandomAccessIterator& operator++() {
    ++_ptr;
    return *this;
  }

  // Post-increment
  RandomAccessIterator operator++(int) {
    RandomAccessIterator temp = *this;
    ++_ptr;
    return temp;
  }

  // Pre-decrement
  RandomAccessIterator& operator--() {
    --_ptr;
    return *this;
  }

  // Post-decrement
  RandomAccessIterator operator--(int) {
    RandomAccessIterator temp = *this;
    --_ptr;
    return temp;
  }

  // Addition operator
  RandomAccessIterator operator+(difference_type n) const {
    return RandomAccessIterator(_ptr + n);
  }

  // Subtraction operator
  RandomAccessIterator operator-(difference_type n) const {
    return RandomAccessIterator(_ptr - n);
  }

  // Difference operator
  difference_type operator-(const RandomAccessIterator& other) const {
    return _ptr - other._ptr;
  }

  // Compound addition
  RandomAccessIterator& operator+=(difference_type n) {
    _ptr += n;
    return *this;
  }

  // Compound subtraction
  RandomAccessIterator& operator-=(difference_type n) {
    _ptr -= n;
    return *this;
  }

  // Subscript operator
  reference operator[](difference_type n) const {
    return _ptr[n];
  }

  // Equality comparison
  bool operator == (const RandomAccessIterator& other) const {
    return _ptr == other._ptr;
  }

  // Inequality comparison
  bool operator != (const RandomAccessIterator& other) const {
    return _ptr != other._ptr;
  }

  // Less-than comparison
  bool operator < (const RandomAccessIterator& other) const {
    return _ptr < other._ptr;
  }

  // Greater-than comparison
  bool operator > (const RandomAccessIterator& other) const {
    return _ptr > other._ptr;
  }

  // Less-than-or-equal comparison
  bool operator <= (const RandomAccessIterator& other) const {
    return _ptr <= other._ptr;
  }

  // Greater-than-or-equal comparison
  bool operator >= (const RandomAccessIterator& other) const {
    return _ptr >= other._ptr;
  }

  private:

  T* _ptr; // Pointer to the underlying element
};

template<typename T>
class ConstantRandomAccessIterator {
  
  public:

  using iterator_category = std::random_access_iterator_tag; // Iterator category
  using value_type = T;                                     // Value type
  using difference_type = std::ptrdiff_t;                   // Difference type
  using pointer = const T*;                                 // Pointer type
  using reference = const T&;                               // Reference type

  // Constructor
  ConstantRandomAccessIterator(const T* ptr) : _ptr(ptr) {}

  // Dereference operator
  reference operator*() const { return *_ptr; }

  // Pointer access operator
  pointer operator->() const { return _ptr; }

  // Pre-increment
  ConstantRandomAccessIterator& operator++() {
    ++_ptr;
    return *this;
  }

  // Post-increment
  ConstantRandomAccessIterator operator++(int) {
    ConstantRandomAccessIterator temp = *this;
    ++_ptr;
    return temp;
  }

  // Pre-decrement
  ConstantRandomAccessIterator& operator--() {
    --_ptr;
    return *this;
  }

  // Post-decrement
  ConstantRandomAccessIterator operator--(int) {
    ConstantRandomAccessIterator temp = *this;
    --_ptr;
    return temp;
  }

  // Addition operator
  ConstantRandomAccessIterator operator+(difference_type n) const {
    return ConstantRandomAccessIterator(_ptr + n);
  }

  // Subtraction operator
  ConstantRandomAccessIterator operator-(difference_type n) const {
    return ConstantRandomAccessIterator(_ptr - n);
  }

  // Difference operator
  difference_type operator-(const ConstantRandomAccessIterator& other) const {
    return _ptr - other._ptr;
  }

  // Compound addition
  ConstantRandomAccessIterator& operator+=(difference_type n) {
    _ptr += n;
    return *this;
  }

  // Compound subtraction
  ConstantRandomAccessIterator& operator-=(difference_type n) {
    _ptr -= n;
    return *this;
  }

  // Subscript operator
  reference operator[](difference_type n) const {
    return _ptr[n];
  }

  // Equality comparison
  bool operator==(const ConstantRandomAccessIterator& other) const {
    return _ptr == other._ptr;
  }

  // Inequality comparison
  bool operator!=(const ConstantRandomAccessIterator& other) const {
    return _ptr != other._ptr;
  }

  // Less-than comparison
  bool operator<(const ConstantRandomAccessIterator& other) const {
    return _ptr < other._ptr;
  }

  // Greater-than comparison
  bool operator>(const ConstantRandomAccessIterator& other) const {
    return _ptr > other._ptr;
  }

  // Less-than-or-equal comparison
  bool operator<=(const ConstantRandomAccessIterator& other) const {
    return _ptr <= other._ptr;
  }

  // Greater-than-or-equal comparison
  bool operator>=(const ConstantRandomAccessIterator& other) const {
    return _ptr >= other._ptr;
  }

  private:

  const T* _ptr; // Pointer to the underlying element (const)
};


}  // end of namespace tf -----------------------------------------------------
