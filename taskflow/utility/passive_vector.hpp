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

  static_assert(std::is_pod_v<T>, "must be a passive data structure type");

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
    
    PassiveVector() noexcept = default;

    explicit PassiveVector(size_type n) {

      if(n > S) {
        _cap  = n << 2;
        _heap = _allocator.allocate(_cap);
      }

      _num = n;
    }

    PassiveVector(const PassiveVector& rhs) {

      _num = rhs._num;

      // heap
      if(rhs._num > _cap) {
        _cap = rhs._cap;
        _heap = _allocator.allocate(_cap);
      }

      std::memcpy(_heap, rhs._heap, _num * sizeof(T));
    }

    ~PassiveVector() {
      if(!in_stack()) {
        _allocator.deallocate(_heap, _cap);
      }
    }

    iterator begin() noexcept                         { return _heap;        }
    const_iterator begin() const noexcept             { return _heap;        }
    const_iterator cbegin() const noexcept            { return _heap;        }
    iterator end() noexcept                           { return _heap + _num; }
    const_iterator end() const noexcept               { return _heap + _num; }
    const_iterator cend() const noexcept              { return _heap + _num; }

    reverse_iterator rbegin() noexcept                { return _heap + _num; }
    const_reverse_iterator crbegin() const noexcept   { return _heap + _num; }
    reverse_iterator rend() noexcept                  { return _heap;        }
    const_reverse_iterator crend() const noexcept     { return _heap;        }

    reference operator [] (size_type idx)             { return _heap[idx];   }
    const_reference operator [] (size_type idx) const { return _heap[idx];   }

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


    reference front()             { return _heap[0];      }
    const_reference front() const { return _heap[0];      }
    reference back()              { return _heap[_num-1]; }
    const_reference back() const  { return _heap[_num-1]; }
  
    pointer data() noexcept             { return _heap; }
    const_pointer data() const noexcept { return _heap; }

    void push_back(const T& item) {
      if(_num == _cap) {
        _enlarge();
      }
      _heap[_num++] = item; 
    }

    void push_back(T&& item) {
      if(_num == _cap) {
        _enlarge();
      }
      _heap[_num++] = item;
    }

    void pop_back() {
      if(_num > 0) {
        --_num;
      }
    }

    void clear() {
      _num = 0;
    }
   
    bool empty() const    { return _num == 0; }
    bool in_stack() const { return _heap == reinterpret_cast<const_pointer>(_stack); }

    size_type size() const     { return _num; }
    size_type capacity() const { return _cap; }
    size_type max_size() const { return std::numeric_limits<size_type>::max(); }
    
  private:
    
    std::byte _stack[S*sizeof(T)];
    
    T* _heap {reinterpret_cast<pointer>(_stack)};
    
    size_type _num {0};
    size_type _cap {S};

    A _allocator;
    
    void _enlarge() {

      auto new_cap  = _cap << 2;
      auto new_heap = _allocator.allocate(new_cap);

      std::memcpy(new_heap, _heap, sizeof(T) * _num);

      if(!in_stack()) {
        _allocator.deallocate(_heap, _cap);
      }
      
      _cap  = new_cap;
      _heap = new_heap;
    }
};


}  // end of namespace tf. ----------------------------------------------------

