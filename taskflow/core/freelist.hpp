#pragma once

#include "tsq.hpp"

namespace tf {


/**
@class: FreelistTaskQueue

@tparam T data type (must be a pointer type)

@brief class to create a lock-free unbounded work-stealing queue

This class implements the work-stealing queue described in the paper,
<a href="https://www.di.ens.fr/~zappa/readings/ppopp13.pdf">Correct and Efficient Work-Stealing for Weak Memory Models</a>.

Only the queue owner can perform pop and push operations,
while others can steal data from the queue simultaneously.

*/
template <typename T>
class FreelistTaskQueue {
  
  static_assert(std::is_pointer_v<T>, "T must be a pointer type");

  struct Array {

    int64_t C;
    int64_t M;
    std::atomic<T>* S;

    explicit Array(int64_t c) :
      C {c},
      M {c-1},
      S {new std::atomic<T>[static_cast<size_t>(C)]} {
    }

    ~Array() {
      delete [] S;
    }

    int64_t capacity() const noexcept {
      return C;
    }

    void push(int64_t i, T o) noexcept {
      S[i & M].store(o, std::memory_order_relaxed);
    }

    T pop(int64_t i) noexcept {
      return S[i & M].load(std::memory_order_relaxed);
    }

    Array* resize(int64_t b, int64_t t) {
      Array* ptr = new Array {2*C};
      for(int64_t i=t; i!=b; ++i) {
        ptr->push(i, pop(i));
      }
      return ptr;
    }

  };

  // Doubling the alignment by 2 seems to generate the most
  // decent performance.
  alignas(2*TF_CACHELINE_SIZE) std::atomic<int64_t> _top;
  alignas(2*TF_CACHELINE_SIZE) std::atomic<int64_t> _bottom;
  std::atomic<Array*> _array;
  std::vector<Array*> _garbage;

  public:

  /**
  @brief constructs the queue with the given size in the base-2 logarithm

  @param LogSize the base-2 logarithm of the queue size
  */
  explicit FreelistTaskQueue(int64_t LogSize = TF_DEFAULT_UNBOUNDED_TASK_QUEUE_LOG_SIZE);

  /**
  @brief destructs the queue
  */
  ~FreelistTaskQueue();

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

  @param item the item to push to the queue
  
  Only the owner thread can insert an item to the queue.
  The operation can trigger the queue to resize its capacity
  if more space is required.
  */
  void push(T item);

  /**
  @brief steals an item from the queue

  Any threads can try to steal an item from the queue.
  The return can be a @c nullptr if this operation failed (not necessary empty).
  */
  T steal();

  /**
  @brief attempts to steal a task with a hint mechanism
  
  @param num_empty_steals a reference to a counter tracking consecutive empty steal attempts
  
  This function tries to steal a task from the queue. If the steal attempt
  is successful, the stolen task is returned. 
  Additionally, if the queue is empty, the provided counter `num_empty_steals` is incremented;
  otherwise, `num_empty_steals` is reset to zero.

  */
  T steal_with_hint(size_t& num_empty_steals);

  private:

  Array* resize_array(Array* a, int64_t b, int64_t t);
};

// Constructor
template <typename T>
FreelistTaskQueue<T>::FreelistTaskQueue(int64_t LogSize) {
  _top.store(0, std::memory_order_relaxed);
  _bottom.store(0, std::memory_order_relaxed);
  _array.store(new Array{(int64_t{1} << LogSize)}, std::memory_order_relaxed);
  _garbage.reserve(32);
}

// Destructor
template <typename T>
FreelistTaskQueue<T>::~FreelistTaskQueue() {
  for(auto a : _garbage) {
    delete a;
  }
  delete _array.load();
}

// Function: empty
template <typename T>
bool FreelistTaskQueue<T>::empty() const noexcept {
  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_relaxed);
  return (b <= t);
}

// Function: size
template <typename T>
size_t FreelistTaskQueue<T>::size() const noexcept {
  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_relaxed);
  return static_cast<size_t>(b >= t ? b - t : 0);
}

// Function: push
template <typename T>
void FreelistTaskQueue<T>::push(T o) {

  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_acquire);
  Array* a = _array.load(std::memory_order_relaxed);

  // queue is full with one additional item (b-t+1)
  if TF_UNLIKELY(a->capacity() - 1 < (b - t)) {
    a = resize_array(a, b, t);
  }

  a->push(b, o);
  std::atomic_thread_fence(std::memory_order_release);

  // original paper uses relaxed here but tsa complains
  _bottom.store(b + 1, std::memory_order_release);
}

// Function: steal
template <typename T>
T FreelistTaskQueue<T>::steal() {
  
  int64_t t = _top.load(std::memory_order_relaxed);
  //std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = _bottom.load(std::memory_order_acquire);

  T item {nullptr};

  if(t < b) {
    Array* a = _array.load(std::memory_order_consume);
    item = a->pop(t);
    if(!_top.compare_exchange_strong(t, t+1,
                                     std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {
      return nullptr;
    }
  }

  return item;
}

// Function: steal
template <typename T>
T FreelistTaskQueue<T>::steal_with_hint(size_t& num_empty_steals) {
  
  int64_t t = _top.load(std::memory_order_relaxed);
  //std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = _bottom.load(std::memory_order_acquire);

  T item {nullptr};

  if(t < b) {
    num_empty_steals = 0;
    Array* a = _array.load(std::memory_order_consume);
    item = a->pop(t);
    if(!_top.compare_exchange_strong(t, t+1,
                                     std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {
      return nullptr;
    }
  }
  ++num_empty_steals;
  return item;
}

// Function: capacity
template <typename T>
int64_t FreelistTaskQueue<T>::capacity() const noexcept {
  return _array.load(std::memory_order_relaxed)->capacity();
}

template <typename T>
typename FreelistTaskQueue<T>::Array*
FreelistTaskQueue<T>::resize_array(Array* a, int64_t b, int64_t t) {

  //Array* tmp = a->resize(b, t);
  //_garbage.push_back(a);
  //std::swap(a, tmp);
  //_array.store(a, std::memory_order_release);
  //// Note: the original paper using relaxed causes t-san to complain
  ////_array.store(a, std::memory_order_relaxed);
  //return a;
  

  Array* tmp = a->resize(b, t);
  _garbage.push_back(a);
  _array.store(tmp, std::memory_order_release);
  // Note: the original paper using relaxed causes t-san to complain
  //_array.store(a, std::memory_order_relaxed);
  return tmp;
}

// ----------------------------------------------------------------------------

/**
@private
*/
template <typename T>
class Freelist {

  friend class Executor;

  public:
  
  struct Bucket {
    std::mutex mutex;
    FreelistTaskQueue<T> queue;
  };

  TF_FORCE_INLINE Freelist(size_t N) : _buckets(N < 4 ? 1 : floor_log2(N)) {}

  //TF_FORCE_INLINE void push(size_t w, T item) {
  //  std::scoped_lock lock(_buckets[w].mutex);
  //  _buckets[w].queue.push(item);  
  //}

  TF_FORCE_INLINE void push(T item) {
    auto b = reinterpret_cast<uintptr_t>(item) % _buckets.size();
    std::scoped_lock lock(_buckets[b].mutex);
    _buckets[b].queue.push(item);
  }

  TF_FORCE_INLINE T steal(size_t w) {
    return _buckets[w].queue.steal();
  }
  
  TF_FORCE_INLINE T steal_with_hint(size_t w, size_t& num_empty_steals) {
    return _buckets[w].queue.steal_with_hint(num_empty_steals);
  }

  TF_FORCE_INLINE size_t size() const {
    return _buckets.size();
  }

  TF_FORCE_INLINE bool empty(size_t& which) const {
    for(which=0; which<_buckets.size(); ++which) {
      if(!_buckets[which].queue.empty()) {
        return false;
      }
    }
    return true;
  }

  private:
  
  std::vector<Bucket> _buckets;
};


}  // end of namespace tf -----------------------------------------------------
