#pragma once

#include <cstring>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <limits>

namespace tf {

// Class: PassiveVector
// A vector storing only passive data structure (PDS) or POD data type.
template <typename T, typename A = std::allocator<T>>
class PassiveVector {

  static_assert(
    std::is_trivial<T>::value && std::is_standard_layout<T>::value, 
    "must be a plain old data type"
  );

  public:

    typedef T                                     value_type;
    typedef T &                                   reference;
    typedef const T &                             const_reference;
    typedef T *                                   pointer;
    typedef const T *                             const_pointer;
    typedef T *                                   iterator;
    typedef const T *                             const_iterator;
    typedef std::reverse_iterator<iterator>       reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
    typedef ptrdiff_t                             difference_type;
    typedef size_t                                size_type;

    // DONE
    PassiveVector() = default;
    
    PassiveVector(PassiveVector&& rhs) noexcept : 
      _b {rhs._b}, _e {rhs._e}, _c {rhs._c} {
      rhs._b = nullptr;
      rhs._e = nullptr;
      rhs._c = nullptr;
    }

    PassiveVector(const PassiveVector& rhs) {
      _enlarge(rhs.capacity());
      std::memcpy(_b, rhs._b, sizeof(T) * rhs.size()); 
      _e = _b + rhs.size();
    }
    
    PassiveVector(size_type n) {
      _enlarge(_next_capacity(n));
      _e = _b + n;
    }
    
    ~PassiveVector() {
      if(_b) {
        _allocator.deallocate(_b, capacity());
      }
    }
    
    void resize(size_type N) {
      if(N > capacity()) {
        _enlarge(_next_capacity(N));
      }
      _e = _b + N;
    }
    
    void reserve(size_type C) {
      if(C > capacity()) {
        _enlarge(_next_capacity(C)); 
      }
    }
    
    void push_back(const T& item) {
      if(size() == capacity()) {
        _enlarge(_next_capacity());
      }
      *_e++ = item; 
    }

    void push_back(T&& item) {
      if(size() == capacity()) {
        _enlarge(_next_capacity());
      }
      *_e++ = std::move(item);
    }

    void pop_back() {
      --_e;
    }

    void clear() {
      _e = _b;
    }
    
    bool operator == (const PassiveVector& rhs) const {
      if(size() != rhs.size()) {
        return false;
      }
      return std::memcmp(_b, rhs._b, size() * sizeof(T)) == 0;
    }

    PassiveVector& operator = (PassiveVector&& rhs) noexcept {
      
      if(this == &rhs) {  // unlikely optimization
        return *this;
      }

      if(_b) {
        _allocator.deallocate(_b, capacity());
      }
      _b = rhs._b;
      _e = rhs._e;
      _c = rhs._c;
      rhs._b = nullptr;
      rhs._e = nullptr;
      rhs._c = nullptr;

      return *this;
    }

    bool empty() const noexcept { return _b == _e; }

    size_type size() const noexcept { return size_type(_e - _b); }
    size_type capacity() const noexcept { return size_type(_c - _b); }
    size_type max_size() const { return (std::numeric_limits<size_type>::max)(); }
    
    iterator begin() noexcept                         { return _b; }
    const_iterator begin() const noexcept             { return _b; }
    const_iterator cbegin() const noexcept            { return _b; }
    iterator end() noexcept                           { return _e; }
    const_iterator end() const noexcept               { return _e; }
    const_iterator cend() const noexcept              { return _e; }

    reverse_iterator rbegin() noexcept                { return _e; }
    const_reverse_iterator crbegin() const noexcept   { return _e; }
    reverse_iterator rend() noexcept                  { return _b; }
    const_reverse_iterator crend() const noexcept     { return _b; }

    reference operator [] (size_type idx)             { return _b[idx]; }
    const_reference operator [] (size_type idx) const { return _b[idx]; }

    reference at(size_type pos) {
      if(pos >= size()) {  // TODO: unlikely optimization
        throw std::out_of_range("accessed position is out of range");
      }
      return this->operator[](pos);
    }

    const_reference at(size_type pos) const {
      if(pos >= size()) {  // TODO: unlikely optimization
        throw std::out_of_range("accessed position is out of range");
      }
      return this->operator[](pos);
    }
    
    reference front()             { return *_b;      }
    const_reference front() const { return *_b;      }
    reference back()              { return _e[-1]; }
    const_reference back() const  { return _e[-1]; }
  
    pointer data() noexcept             { return _b; }
    const_pointer data() const noexcept { return _b; }
    
  private:
    
    T* _b {nullptr};
    T* _e {nullptr};
    T* _c {nullptr};

    A _allocator;

    size_type _next_capacity() const {
      if(capacity() == 0) {
        return std::max(64 / sizeof(T), size_type(1));
      }
      if(capacity() > 4096 * 32 / sizeof(T)) {
        return capacity() * 2;
      }
      return(capacity() * 3 + 1) / 2;
    }

    size_type _next_capacity(size_type n) const {
      if(n == 0) {
        return std::max(64 / sizeof(T), size_type(1));
      }
      if(n > 4096 * 32 / sizeof(T)) {
        return n * 2;
      }
      return (n * 3 + 1) / 2;
    }
    
    void _enlarge(size_type new_c) {

      auto new_b = _allocator.allocate(new_c);
      auto old_n = size();
      
      if(_b) {
        std::memcpy(new_b, _b, sizeof(T) * old_n);
        _allocator.deallocate(_b, capacity());
      }
      
      _b = new_b;
      _e = _b + old_n;
      _c = _b + new_c;
    }
};


}  // end of namespace tf. ----------------------------------------------------

