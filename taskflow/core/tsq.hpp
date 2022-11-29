#pragma once

#include "../utility/macros.hpp"
#include "../utility/traits.hpp"

/**
@file tsq.hpp
@brief task queue include file
*/

namespace tf {


// ----------------------------------------------------------------------------
// Task Types
// ----------------------------------------------------------------------------

/**
@enum TaskPriority

@brief enumeration of all task priority values

A priority is an enumerated value of type @c unsigned.
Currently, %Taskflow defines three priority levels, 
@c HIGH, @c NORMAL, and @c LOW, starting from 0, 1, to 2.
That is, the lower the value, the higher the priority.

*/
enum class TaskPriority : unsigned {
  /** @brief value of the highest priority (i.e., 0)  */
  HIGH = 0,
  /** @brief value of the normal priority (i.e., 1)  */
  NORMAL = 1,
  /** @brief value of the lowest priority (i.e., 2) */
  LOW = 2,
  /** @brief conventional value for iterating priority values */
  MAX = 3
};



// ----------------------------------------------------------------------------
// Task Queue
// ----------------------------------------------------------------------------


/**
@class: TaskQueue

@tparam T data type (must be a pointer type)
@tparam MAX_PRIORITY maximum level of the priority 

@brief class to create a lock-free unbounded single-producer multiple-consumer queue

This class implements the work-stealing queue described in the paper,
<a href="https://www.di.ens.fr/~zappa/readings/ppopp13.pdf">Correct and Efficient Work-Stealing for Weak Memory Models</a>,
and extends it to include priority.

Only the queue owner can perform pop and push operations,
while others can steal data from the queue simultaneously.
Priority starts from zero (highest priority) to the template value 
`MAX_PRIORITY-1` (lowest priority).
All operations are associated with priority values to indicate
the corresponding queues to which an operation is applied.

The default template value, `MAX_PRIORITY`, is `TaskPriority::MAX` 
which applies only three priority levels to the task queue.

@code{.cpp}
auto [A, B, C, D, E] = taskflow.emplace(
  [] () { },
  [&] () { 
    std::cout << "Task B: " << counter++ << '\n';  // 0
  },
  [&] () { 
    std::cout << "Task C: " << counter++ << '\n';  // 2
  },
  [&] () { 
    std::cout << "Task D: " << counter++ << '\n';  // 1
  },
  [] () { }
);

A.precede(B, C, D); 
E.succeed(B, C, D);
  
B.priority(tf::TaskPriority::HIGH);
C.priority(tf::TaskPriority::LOW);
D.priority(tf::TaskPriority::NORMAL);
  
executor.run(taskflow).wait();
@endcode

In the above example, we have a task graph of five tasks,
@c A, @c B, @c C, @c D, and @c E, in which @c B, @c C, and @c D
can run in simultaneously when @c A finishes.
Since we only uses one worker thread in the executor, 
we can deterministically run @c B first, then @c D, and @c C
in order of their priority values.
The output is as follows:

@code{.shell-session}
Task B: 0
Task D: 1
Task C: 2
@endcode

*/
template <typename T, unsigned MAX_PRIORITY = static_cast<unsigned>(TaskPriority::MAX)>
class TaskQueue {
  
  static_assert(MAX_PRIORITY > 0, "MAX_PRIORITY must be at least one");
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
  CachelineAligned<std::atomic<int64_t>> _top[MAX_PRIORITY];
  CachelineAligned<std::atomic<int64_t>> _bottom[MAX_PRIORITY];
  std::atomic<Array*> _array[MAX_PRIORITY];
  std::vector<Array*> _garbage[MAX_PRIORITY];

  //std::atomic<T> _cache {nullptr};

  public:

    /**
    @brief constructs the queue with a given capacity

    @param capacity the capacity of the queue (must be power of 2)
    */
    explicit TaskQueue(int64_t capacity = 512);

    /**
    @brief destructs the queue
    */
    ~TaskQueue();

    /**
    @brief queries if the queue is empty at the time of this call
    */
    bool empty() const noexcept;

    /**
    @brief queries if the queue is empty at a specific priority value
    */
    bool empty(unsigned priority) const noexcept;

    /**
    @brief queries the number of items at the time of this call
    */
    size_t size() const noexcept;

    /**
    @brief queries the number of items with the given priority
           at the time of this call
    */
    size_t size(unsigned priority) const noexcept;

    /**
    @brief queries the capacity of the queue
    */
    int64_t capacity() const noexcept;
    
    /**
    @brief queries the capacity of the queue at a specific priority value
    */
    int64_t capacity(unsigned priority) const noexcept;

    /**
    @brief inserts an item to the queue

    @param item the item to push to the queue
    @param priority priority value of the item to push (default = 0)
    
    Only the owner thread can insert an item to the queue.
    The operation can trigger the queue to resize its capacity
    if more space is required.
    */
    TF_FORCE_INLINE void push(T item, unsigned priority);

    /**
    @brief pops out an item from the queue

    Only the owner thread can pop out an item from the queue.
    The return can be a @c nullptr if this operation failed (empty queue).
    */
    T pop();

    /**
    @brief pops out an item with a specific priority value from the queue

    @param priority priority of the item to pop

    Only the owner thread can pop out an item from the queue.
    The return can be a @c nullptr if this operation failed (empty queue).
    */
    TF_FORCE_INLINE T pop(unsigned priority);

    /**
    @brief steals an item from the queue

    Any threads can try to steal an item from the queue.
    The return can be a @c nullptr if this operation failed (not necessary empty).
    */
    T steal();

    /**
    @brief steals an item with a specific priority value from the queue

    @param priority priority of the item to steal

    Any threads can try to steal an item from the queue.
    The return can be a @c nullptr if this operation failed (not necessary empty).
    */
    T steal(unsigned priority);

  private:
    TF_NO_INLINE Array* resize_array(Array* a, unsigned p, std::int64_t b, std::int64_t t);
};

// Constructor
template <typename T, unsigned MAX_PRIORITY>
TaskQueue<T, MAX_PRIORITY>::TaskQueue(int64_t c) {
  assert(c && (!(c & (c-1))));
  unroll<0, MAX_PRIORITY, 1>([&](auto p){
    _top[p].data.store(0, std::memory_order_relaxed);
    _bottom[p].data.store(0, std::memory_order_relaxed);
    _array[p].store(new Array{c}, std::memory_order_relaxed);
    _garbage[p].reserve(32);
  });
}

// Destructor
template <typename T, unsigned MAX_PRIORITY>
TaskQueue<T, MAX_PRIORITY>::~TaskQueue() {
  unroll<0, MAX_PRIORITY, 1>([&](auto p){
    for(auto a : _garbage[p]) {
      delete a;
    }
    delete _array[p].load();
  });
}

// Function: empty
template <typename T, unsigned MAX_PRIORITY>
bool TaskQueue<T, MAX_PRIORITY>::empty() const noexcept {
  for(unsigned i=0; i<MAX_PRIORITY; i++) {
    if(!empty(i)) {
      return false;
    }
  }
  return true;
}

// Function: empty
template <typename T, unsigned MAX_PRIORITY>
bool TaskQueue<T, MAX_PRIORITY>::empty(unsigned p) const noexcept {
  int64_t b = _bottom[p].data.load(std::memory_order_relaxed);
  int64_t t = _top[p].data.load(std::memory_order_relaxed);
  return (b <= t);
}

// Function: size
template <typename T, unsigned MAX_PRIORITY>
size_t TaskQueue<T, MAX_PRIORITY>::size() const noexcept {
  size_t s;
  unroll<0, MAX_PRIORITY, 1>([&](auto i) { s = i ? size(i) + s : size(i); });
  return s;
}

// Function: size
template <typename T, unsigned MAX_PRIORITY>
size_t TaskQueue<T, MAX_PRIORITY>::size(unsigned p) const noexcept {
  int64_t b = _bottom[p].data.load(std::memory_order_relaxed);
  int64_t t = _top[p].data.load(std::memory_order_relaxed);
  return static_cast<size_t>(b >= t ? b - t : 0);
}

// Function: push
template <typename T, unsigned MAX_PRIORITY>
TF_FORCE_INLINE void TaskQueue<T, MAX_PRIORITY>::push(T o, unsigned p) {

  int64_t b = _bottom[p].data.load(std::memory_order_relaxed);
  int64_t t = _top[p].data.load(std::memory_order_acquire);
  Array* a = _array[p].load(std::memory_order_relaxed);

  // queue is full
  if(a->capacity() - 1 < (b - t)) {
    a = resize_array(a, p, b, t);
  }

  a->push(b, o);
  std::atomic_thread_fence(std::memory_order_release);
  _bottom[p].data.store(b + 1, std::memory_order_relaxed);
}

// Function: pop
template <typename T, unsigned MAX_PRIORITY>
T TaskQueue<T, MAX_PRIORITY>::pop() {
  for(unsigned i=0; i<MAX_PRIORITY; i++) {
    if(auto t = pop(i); t) {
      return t;
    }
  }
  return nullptr;
}

// Function: pop
template <typename T, unsigned MAX_PRIORITY>
TF_FORCE_INLINE T TaskQueue<T, MAX_PRIORITY>::pop(unsigned p) {

  int64_t b = _bottom[p].data.load(std::memory_order_relaxed) - 1;
  Array* a = _array[p].load(std::memory_order_relaxed);
  _bottom[p].data.store(b, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t t = _top[p].data.load(std::memory_order_relaxed);

  T item {nullptr};

  if(t <= b) {
    item = a->pop(b);
    if(t == b) {
      // the last item just got stolen
      if(!_top[p].data.compare_exchange_strong(t, t+1,
                                               std::memory_order_seq_cst,
                                               std::memory_order_relaxed)) {
        item = nullptr;
      }
      _bottom[p].data.store(b + 1, std::memory_order_relaxed);
    }
  }
  else {
    _bottom[p].data.store(b + 1, std::memory_order_relaxed);
  }

  return item;
}

// Function: steal
template <typename T, unsigned MAX_PRIORITY>
T TaskQueue<T, MAX_PRIORITY>::steal() {
  for(unsigned i=0; i<MAX_PRIORITY; i++) {
    if(auto t = steal(i); t) {
      return t;
    }
  }
  return nullptr;
}

// Function: steal
template <typename T, unsigned MAX_PRIORITY>
T TaskQueue<T, MAX_PRIORITY>::steal(unsigned p) {
  
  int64_t t = _top[p].data.load(std::memory_order_acquire);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = _bottom[p].data.load(std::memory_order_acquire);

  T item {nullptr};

  if(t < b) {
    Array* a = _array[p].load(std::memory_order_consume);
    item = a->pop(t);
    if(!_top[p].data.compare_exchange_strong(t, t+1,
                                             std::memory_order_seq_cst,
                                             std::memory_order_relaxed)) {
      return nullptr;
    }
  }

  return item;
}

// Function: capacity
template <typename T, unsigned MAX_PRIORITY>
int64_t TaskQueue<T, MAX_PRIORITY>::capacity() const noexcept {
  size_t s;
  unroll<0, MAX_PRIORITY, 1>([&](auto i) { 
    s = i ? capacity(i) + s : capacity(i); 
  });
  return s;
}

// Function: capacity
template <typename T, unsigned MAX_PRIORITY>
int64_t TaskQueue<T, MAX_PRIORITY>::capacity(unsigned p) const noexcept {
  return _array[p].load(std::memory_order_relaxed)->capacity();
}

template <typename T, unsigned MAX_PRIORITY>
TF_NO_INLINE typename TaskQueue<T, MAX_PRIORITY>::Array*
  TaskQueue<T, MAX_PRIORITY>::resize_array(Array* a, unsigned p, std::int64_t b, std::int64_t t) {

  Array* tmp = a->resize(b, t);
  _garbage[p].push_back(a);
  std::swap(a, tmp);
  _array[p].store(a, std::memory_order_release);
  // Note: the original paper using relaxed causes t-san to complain
  //_array.store(a, std::memory_order_relaxed);
  return a;
}


}  // end of namespace tf -----------------------------------------------------
