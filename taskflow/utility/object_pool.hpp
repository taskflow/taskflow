// 2019/06/13 - created by Tsung-Wei Huang
//  - implemented an object pool class

#pragma once

#include <vector>

namespace tf {

// Class: ObjectPool
template <typename T>
class ObjectPool {

  public:

    ObjectPool() = default;
    ObjectPool(const ObjectPool& other) = delete;
    ObjectPool(ObjectPool&& other);

    ~ObjectPool();

    ObjectPool& operator = (const ObjectPool& other) = delete;
    ObjectPool& operator = (ObjectPool&& other);
  
    template <typename... ArgsT>
    T* get(ArgsT&&... args);

    void recycle(T* obj);

    size_t size() const;

  private:
    
    std::vector<T*> _free_list;
};

// Move constructor
template <typename T>
ObjectPool<T>::ObjectPool(ObjectPool&& other) :
  _free_list {std::move(other._free_list)} {
}

// Destructor
template <typename T>
ObjectPool<T>::~ObjectPool() {
  for(T* ptr : _free_list) {
    std::free(ptr);
  }
}

// Move assignment
template <typename T>
ObjectPool<T>& ObjectPool<T>::operator = (ObjectPool&& other) {
  _free_list = std::move(other._free_list);
  return *this;
}

// Function: size
template <typename T>
size_t ObjectPool<T>::size() const {
  return _free_list.size();
}

// Function: get
template <typename T>
template <typename... ArgsT>
T* ObjectPool<T>::get(ArgsT&&... args) {

  T* ptr;

  if(_free_list.empty()) {
    ptr = static_cast<T*>(std::malloc(sizeof(T)));
  }
  else {
    ptr = _free_list.back();
    _free_list.pop_back();
  }
    
  ::new(ptr) T(std::forward<ArgsT>(args)...);

  return ptr;
}

// Procedure: recycle
template <typename T>
void ObjectPool<T>::recycle(T* obj) {
  obj->~T();
  _free_list.push_back(obj);
}

// ----------------------------------------------------------------------------

// Function: per_thread_object_pool
template <typename T>
ObjectPool<T>& per_thread_object_pool() {
  thread_local ObjectPool<T> op;
  return op;
}

}  // end of namespace tf -----------------------------------------------------





