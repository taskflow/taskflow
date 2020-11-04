#pragma once

#include <cstring>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <limits>

namespace tf {

// Class: PassiveVector
// A vector storing only passive data structure (PDS) or POD data type.
template <typename T, size_t S = 4, typename A = std::allocator<T>>
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
    
    PassiveVector() noexcept :  
      _data {reinterpret_cast<pointer>(_stack)},
      _num  {0},
      _cap  {S} {
    }

    explicit PassiveVector(size_type n) : _num {n} {
      
      // need to place on heap
      if(n > S) {
        _cap  = n << 2;
        _data = _allocator.allocate(_cap);
      }
      // stack
      else {
        _cap = S;
        _data = reinterpret_cast<pointer>(_stack);
      }

    }

    PassiveVector(const PassiveVector& rhs) : _num {rhs._num} {

      // heap
      if(rhs._num > S) {
        _cap = rhs._cap;
        _data = _allocator.allocate(rhs._cap);
      }
      else {
        _cap = S;
        _data = reinterpret_cast<pointer>(_stack);
      }

      std::memcpy(_data, rhs._data, _num * sizeof(T));
    }

    PassiveVector(PassiveVector&& rhs) : _num {rhs._num} {

      // rhs is in the stack
      if(rhs.in_stack()) {
        _cap  = S;
        _data = reinterpret_cast<pointer>(_stack);
        std::memcpy(_stack, rhs._stack, rhs._num*sizeof(T));
      }
      // rhs is in the heap
      else {
        _cap = rhs._cap;
        _data = rhs._data;
        rhs._data = reinterpret_cast<pointer>(rhs._stack);
        rhs._cap  = S;
      }

      rhs._num = 0;
    }

    ~PassiveVector() {
      if(!in_stack()) {
        _allocator.deallocate(_data, _cap);
      }
    }

    iterator begin() noexcept                         { return _data;        }
    const_iterator begin() const noexcept             { return _data;        }
    const_iterator cbegin() const noexcept            { return _data;        }
    iterator end() noexcept                           { return _data + _num; }
    const_iterator end() const noexcept               { return _data + _num; }
    const_iterator cend() const noexcept              { return _data + _num; }

    reverse_iterator rbegin() noexcept                { return _data + _num; }
    const_reverse_iterator crbegin() const noexcept   { return _data + _num; }
    reverse_iterator rend() noexcept                  { return _data;        }
    const_reverse_iterator crend() const noexcept     { return _data;        }

    reference operator [] (size_type idx)             { return _data[idx];   }
    const_reference operator [] (size_type idx) const { return _data[idx];   }

    reference at(size_type pos) {
      if(pos >= _num) {
        throw std::out_of_range("accessed position is out of range");
      }
      return this->operator[](pos);
    }

    const_reference at(size_type pos) const {
      if(pos >= _num) {
        throw std::out_of_range("accessed position is out of range");
      }
      return this->operator[](pos);
    }


    reference front()             { return _data[0];      }
    const_reference front() const { return _data[0];      }
    reference back()              { return _data[_num-1]; }
    const_reference back() const  { return _data[_num-1]; }
  
    pointer data() noexcept             { return _data; }
    const_pointer data() const noexcept { return _data; }

    void push_back(const T& item) {
      if(_num == _cap) {
        _enlarge(_cap << 1);
      }
      _data[_num++] = item; 
    }

    void push_back(T&& item) {
      if(_num == _cap) {
        _enlarge(_cap << 1);
      }
      _data[_num++] = item;
    }

    void pop_back() {
      if(_num > 0) {
        --_num;
      }
    }

    void clear() {
      _num = 0;
    }

    void resize(size_type N) {
      if(N > _cap) {
        _enlarge(N<<1);
      }
      _num = N;
    }

    void reserve(size_type C) {
      if(C > _cap) {
        _enlarge(C);     
      }
    }
   
    bool empty() const    { return _num == 0; }
    bool in_stack() const { return _data == reinterpret_cast<const_pointer>(_stack); }

    size_type size() const     { return _num; }
    size_type capacity() const { return _cap; }
    size_type max_size() const { return (std::numeric_limits<size_type>::max)(); }

    bool operator == (const PassiveVector& rhs) const {
      if(_num != rhs._num) {
        return false;
      }
      return std::memcmp(_data, rhs._data, _num * sizeof(T)) == 0;
    }
    
  private:
    
    char _stack[S*sizeof(T)];
    
    T* _data;
    
    size_type _num;
    size_type _cap;

    A _allocator;
    
    void _enlarge(size_type new_cap) {

      auto new_data = _allocator.allocate(new_cap);

      std::memcpy(new_data, _data, sizeof(T) * _num);

      if(!in_stack()) {
        _allocator.deallocate(_data, _cap);
      }
      
      _cap  = new_cap;
      _data = new_data;
    }
};


}  // end of namespace tf. ----------------------------------------------------

