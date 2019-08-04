// 2019/05/15 - created by Tsung-Wei Huang
//  - isolated from the original workstealing executor

#pragma once

#include <atomic>
#include <vector>
#include <optional>

namespace tf {

/**
@class: WorkStealingQueue

@tparam T data type

@brief Lock-free unbounded single-producer multiple-consumer queue.

This class implements the work stealing queue described in the paper, 
"Dynamic Circular Work-stealing Deque," SPAA, 2015.
Only the queue owner can perform pop and push operations,
while others can steal data from the queue.

PPoPP implementation paper
"Correct and Efficient Work-Stealing for Weak Memory Models"
https://www.di.ens.fr/~zappa/readings/ppopp13.pdf
*/
template <typename T>
class WorkStealingQueue {

  //constexpr static int64_t cacheline_size = 64;

  //using storage_type = std::aligned_storage_t<sizeof(T), cacheline_size>;

  struct Array {

    int64_t C;
    int64_t M;
    //storage_type* S;
    T* S;

    explicit Array(int64_t c) : 
      C {c},
      M {c-1},
      //S {new storage_type[C]} {
      S {new T[static_cast<size_t>(C)]} {
      //for(int64_t i=0; i<C; ++i) {
      //  ::new (std::addressof(S[i])) T();
      //}
    }

    ~Array() {
      //for(int64_t i=0; i<C; ++i) {
      //  reinterpret_cast<T*>(std::addressof(S[i]))->~T();
      //}
      delete [] S;
    }

    int64_t capacity() const noexcept {
      return C;
    }
    
    template <typename O>
    void push(int64_t i, O&& o) noexcept {
      //T* ptr = reinterpret_cast<T*>(std::addressof(S[i & M]));
      //*ptr = std::forward<O>(o); 
      S[i & M] = std::forward<O>(o);
    }

    T pop(int64_t i) noexcept {
      //return *reinterpret_cast<T*>(std::addressof(S[i & M]));
      return S[i & M];
    }

    Array* resize(int64_t b, int64_t t) {
      Array* ptr = new Array {2*C};
      for(int64_t i=t; i!=b; ++i) {
        ptr->push(i, pop(i));
      }
      return ptr;
    }

  };

  std::atomic<int64_t> _top;
  std::atomic<int64_t> _bottom;
  std::atomic<Array*> _array;
  std::vector<Array*> _garbage;
  //char _padding[cacheline_size];

  public:
    
    /**
    @brief constructs the queue with a given capacity

    @param capacity the capacity of the queue (must be power of 2)
    */
    explicit WorkStealingQueue(int64_t capacity = 1024);

    /**
    @brief destructs the queue
    */
    ~WorkStealingQueue();
    
    /**
    @brief queries if the queue is empty at the time of this call
    */
    bool empty() const noexcept;
    
    /**
    @brief queries the number of items at the time of this call
    */
    size_t size() const noexcept;

    /**
    @brief queries the capacity of the queue
    */
    int64_t capacity() const noexcept;
    
    /**
    @brief inserts an item to the queue

    Only the owner thread can insert an item to the queue. 
    The operation can trigger the queue to resize its capacity 
    if more space is required.

    @tparam O data type 

    @param item the item to perfect-forward to the queue
    */
    template <typename O>
    void push(O&& item);
    
    /**
    @brief pops out an item from the queue

    Only the owner thread can pop out an item from the queue. 
    The return can be a @std_nullopt if this operation failed (empty queue).
    */
    std::optional<T> pop();
    
    /**
    @brief steals an item from the queue

    Any threads can try to steal an item from the queue.
    The return can be a @std_nullopt if this operation failed (not necessary empty).
    */
    std::optional<T> steal();
};

// Constructor
template <typename T>
WorkStealingQueue<T>::WorkStealingQueue(int64_t c) {
  assert(c && (!(c & (c-1))));
  _top.store(0, std::memory_order_relaxed);
  _bottom.store(0, std::memory_order_relaxed);
  _array.store(new Array{c}, std::memory_order_relaxed);
  _garbage.reserve(32);
}

// Destructor
template <typename T>
WorkStealingQueue<T>::~WorkStealingQueue() {
  for(auto a : _garbage) {
    delete a;
  }
  delete _array.load();
}
  
// Function: empty
template <typename T>
bool WorkStealingQueue<T>::empty() const noexcept {
  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_relaxed);
  return b <= t;
}

// Function: size
template <typename T>
size_t WorkStealingQueue<T>::size() const noexcept {
  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_relaxed);
  return static_cast<size_t>(b >= t ? b - t : 0);
}

// Function: push
template <typename T>
template <typename O>
void WorkStealingQueue<T>::push(O&& o) {
  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_acquire);
  Array* a = _array.load(std::memory_order_relaxed);

  // queue is full
  if(a->capacity() - 1 < (b - t)) {
    Array* tmp = a->resize(b, t);
    _garbage.push_back(a);
    std::swap(a, tmp);
    _array.store(a, std::memory_order_relaxed);
  }

  a->push(b, std::forward<O>(o));
  std::atomic_thread_fence(std::memory_order_release);
  _bottom.store(b + 1, std::memory_order_relaxed);
}

// Function: pop
template <typename T>
std::optional<T> WorkStealingQueue<T>::pop() {
  int64_t b = _bottom.load(std::memory_order_relaxed) - 1;
  Array* a = _array.load(std::memory_order_relaxed);
  _bottom.store(b, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t t = _top.load(std::memory_order_relaxed);

  std::optional<T> item;

  if(t <= b) {
    item = a->pop(b);
    if(t == b) {
      // the last item just got stolen
      if(!_top.compare_exchange_strong(t, t+1, 
                                       std::memory_order_seq_cst, 
                                       std::memory_order_relaxed)) {
        item = std::nullopt;
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
std::optional<T> WorkStealingQueue<T>::steal() {
  int64_t t = _top.load(std::memory_order_acquire);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = _bottom.load(std::memory_order_acquire);
  
  std::optional<T> item;

  if(t < b) {
    Array* a = _array.load(std::memory_order_consume);
    item = a->pop(t);
    if(!_top.compare_exchange_strong(t, t+1,
                                     std::memory_order_seq_cst,
                                     std::memory_order_relaxed)) {
      return std::nullopt;
    }
  }

  return item;
}

// Function: capacity
template <typename T>
int64_t WorkStealingQueue<T>::capacity() const noexcept {
  return _array.load(std::memory_order_relaxed)->capacity();
}

}  // end of namespace tf -----------------------------------------------------
