// 2019/07/10 - modified by Tsung-Wei Huang
//  - replace raw pointer with smart pointer
//
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

    ~ObjectPool() = default;

    ObjectPool& operator = (const ObjectPool& other) = delete;
    ObjectPool& operator = (ObjectPool&& other);
  
    size_t size() const;

    void release(std::unique_ptr<T>&& obj);

    template <typename... ArgsT>
    std::unique_ptr<T> acquire(ArgsT&&... args);

  private:
    
    std::vector<std::unique_ptr<T>> _stack;
};

// Move constructor
template <typename T>
ObjectPool<T>::ObjectPool(ObjectPool&& other) :
  _stack {std::move(other._stack)} {
}

// Move assignment
template <typename T>
ObjectPool<T>& ObjectPool<T>::operator = (ObjectPool&& other) {
  _stack = std::move(other._stack);
  return *this;
}

// Function: size
template <typename T>
size_t ObjectPool<T>::size() const {
  return _stack.size();
}

// Function: acquire
template <typename T>
template <typename... ArgsT>
std::unique_ptr<T> ObjectPool<T>::acquire(ArgsT&&... args) {
  if(_stack.empty()) {
    return std::make_unique<T>(std::forward<ArgsT>(args)...);
  }
  else {
    auto ptr = std::move(_stack.back());
    ptr->animate(std::forward<ArgsT>(args)...);
    _stack.pop_back();
    return ptr;
  }
}

// Procedure: release
template <typename T>
void ObjectPool<T>::release(std::unique_ptr<T>&& obj) {
  obj->recycle();
  _stack.push_back(std::move(obj));
}

}  // end of namespace tf -----------------------------------------------------





