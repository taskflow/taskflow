// 2018/12/24 - modified by Tsung-Wei Huang
//   - refined the work balancing strategy 
//
// 2018/12/06 - modified by Tsung-Wei Huang
//   - refactored the code
//   - added load balancing strategy
//   - removed the storage alignment in WorkStealingQueue
//
// 2018/12/03 - created by Tsung-Wei Huang
//   - added WorkStealingQueue class

#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <memory>
#include <cassert>
#include <deque>
#include <optional>
#include <thread>
#include <algorithm>
#include <set>
#include <numeric>

namespace tf {

/**
@class: WorkStealingQueue

@tparam T data type

@brief Lock-free unbounded single-producer multiple-consumer queue.

This class implements the work stealing queue described in the paper, 
"Dynamic Circular Work-stealing Deque," SPAA, 2015.
Only the queue owner can perform pop and push operations,
while others can steal data from the queue.
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

    Array(int64_t c) : 
      C {c},
      M {c-1},
      //S {new storage_type[C]} {
      S {new T[C]} {
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
    WorkStealingQueue(int64_t capacity = 4096);

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
    int64_t size() const noexcept;

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
    The return can be a @std_nullopt if this operation failed (not necessary empty).
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
  _array.store((new Array{c}), std::memory_order_relaxed);
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
int64_t WorkStealingQueue<T>::size() const noexcept {
  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_relaxed);
  return b - t;
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

// ----------------------------------------------------------------------------


/** 
@class: WorkStealingThreadpool

@brief Executor that implements an efficient work stealing algorithm.

@tparam Closure closure type

*/
template <typename Closure>
class WorkStealingThreadpool {
    
  struct Worker {
    std::condition_variable cv;
    WorkStealingQueue<Closure> queue;
    std::optional<Closure> cache;
    bool exit  {false};
    bool ready {false};
    uint64_t seed;
    unsigned last_victim;
  };

  public:
    
    /**
    @brief constructs the executor with a given number of worker threads

    @param N the number of worker threads
    */
    explicit WorkStealingThreadpool(unsigned N);

    /**
    @brief destructs the executor

    Destructing the executor will immediately force all worker threads to stop.
    The executor does not guarantee all tasks to finish upon destruction.
    */
    ~WorkStealingThreadpool();
    
    /**
    @brief queries the number of worker threads
    */
    size_t num_workers() const;
    
    /**
    @brief queries if the caller is the owner of the executor
    */
    bool is_owner() const;
    
    /**
    @brief constructs the closure in place in the executor

    @tparam ArgsT... argument parameter pack

    @param args... arguments to forward to the constructor of the closure
    */
    template <typename... ArgsT>
    void emplace(ArgsT&&... args);
    
    /**
    @brief moves a batch of closures to the executor

    @param closures a vector of closures to move
    */
    void batch(std::vector<Closure>&& closures);

  private:
    
    const std::thread::id _owner {std::this_thread::get_id()};

    mutable std::mutex _mutex;

    std::vector<Worker> _workers;
    std::vector<Worker*> _idlers;
    std::vector<std::thread> _threads;

    std::unordered_map<std::thread::id, size_t> _worker_maps;

    WorkStealingQueue<Closure> _queue;

    void _spawn(unsigned);
    void _shutdown();
    void _balance_load(unsigned);

    unsigned _randomize(uint64_t&) const;
    unsigned _fast_modulo(uint32_t, uint32_t) const;

    std::optional<Closure> _steal(unsigned);
};

// Constructor
template <typename Closure>
WorkStealingThreadpool<Closure>::WorkStealingThreadpool(unsigned N) : _workers {N} {
  _worker_maps.reserve(N);
  _spawn(N);
}

// Destructor
template <typename Closure>
WorkStealingThreadpool<Closure>::~WorkStealingThreadpool() {
  _shutdown();
}

// Procedure: _shutdown
template <typename Closure>
void WorkStealingThreadpool<Closure>::_shutdown(){

  assert(is_owner());

  {
    std::scoped_lock lock(_mutex);
    for(auto& w : _workers){
      w.exit = true;
      w.cv.notify_one();
    }
  } 

  for(auto& t : _threads){
    t.join();
  } 

  //_threads.clear();  
  //_workers.clear();
  //_worker_maps.clear();
}

// Function: _randomize
template <typename Closure>
unsigned WorkStealingThreadpool<Closure>::_randomize(uint64_t& state) const {
  uint64_t current = state;
  state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
  // Generate the random output (using the PCG-XSH-RS scheme)
  return static_cast<unsigned>((current ^ (current >> 22)) >> (22 + (current >> 61)));
}

// http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
template <typename Closure>
unsigned WorkStealingThreadpool<Closure>::_fast_modulo(uint32_t x, uint32_t N) const {
  return ((uint64_t) x * (uint64_t) N) >> 32;
}

// Procedure: _spawn
template <typename Closure>
void WorkStealingThreadpool<Closure>::_spawn(unsigned N) {
  
  // Lock to synchronize all workers before creating _worker_mapss
  std::scoped_lock lock(_mutex);
  
  for(unsigned i=0; i<N; ++i) {
    _threads.emplace_back([this, i, N] () -> void {

      std::optional<Closure> t;
      Worker& w = (_workers[i]);
      w.last_victim = (i + 1) % N;
      w.seed = i + 1;

      std::unique_lock lock(_mutex, std::defer_lock);

      while(!w.exit) {
        
        // pop from my own queue
        if(t = w.queue.pop(); !t) {
          // steal from others
          t = _steal(i);
        }
        
        // no tasks
        if(!t) {
          if(lock.try_lock()) {  // avoid contention
            if(_queue.empty()) {
              w.ready = false;
              _idlers.push_back(&w);
              while(!w.ready && !w.exit) {
                w.cv.wait(lock);
              }
            }
            lock.unlock();
          }

          if(w.cache) {
            std::swap(t, w.cache);
          }
        }

        while(t) {
          (*t)();
          if(w.cache) {
            t = std::move(w.cache);
            w.cache = std::nullopt;
          }
          else {
            t = std::nullopt;
          }
        }

        // balance load
        _balance_load(i);

      } // End of while ------------------------------------------------------ 

    });     

    _worker_maps.insert({_threads.back().get_id(), i});
  }
}

// Function: is_owner
template <typename Closure>
bool WorkStealingThreadpool<Closure>::is_owner() const {
  return std::this_thread::get_id() == _owner;
}

// Function: num_workers
template <typename Closure>
size_t WorkStealingThreadpool<Closure>::num_workers() const { 
  return _threads.size();  
}

// Procedure: _balance_load
template <typename Closure>
void WorkStealingThreadpool<Closure>::_balance_load(unsigned me) {

  auto n = _workers[me].queue.size();

  // return if no idler - this might not be the right value
  // but it doesn't affect the correctness
  if(_idlers.empty() || n <= 4) {
    return;
  }
  
  // try with probability 1/n
  if(_fast_modulo(_randomize(_workers[me].seed), n) == 0u) {
    // wake up my partner to help balance
    if(_mutex.try_lock()) {
      if(!_idlers.empty()) {
        Worker* w = _idlers.back();
        _idlers.pop_back();
        w->ready = true;
        w->cv.notify_one();
        w->last_victim = me;
      }
      _mutex.unlock();
    }
  }
}

// Function: _steal
template <typename Closure>
std::optional<Closure> WorkStealingThreadpool<Closure>::_steal(unsigned thief) {

  std::optional<Closure> task;
  
  for(int round=0; round<1024; ++round) {

    // try getting a task from the centralized queue
    if(task = _queue.steal(); task) {
      return task;
    }

    // try stealing a task from other workers
    unsigned victim = _workers[thief].last_victim;

    for(unsigned i=0; i<_workers.size(); i++){

      if(victim != thief) {
        if(task = _workers[victim].queue.steal(); task){
          _workers[thief].last_victim = victim;
          return task;
        }
      }

      if(++victim; victim == _workers.size()){
        victim = 0;
      }
    }

    // nothing happens this round
    std::this_thread::yield();
  }
  
  return std::nullopt; 
}

// Procedure: emplace
template <typename Closure>
template <typename... ArgsT>
void WorkStealingThreadpool<Closure>::emplace(ArgsT&&... args){

  //no worker thread available
  if(num_workers() == 0){
    Closure{std::forward<ArgsT>(args)...}();
    return;
  }

  // caller is not the owner
  if(auto tid = std::this_thread::get_id(); tid != _owner){

    // the caller is the worker of the threadpool
    if(auto itr = _worker_maps.find(tid); itr != _worker_maps.end()){

      unsigned me = itr->second;

      // dfs speculation
      if(!_workers[me].cache){
        _workers[me].cache.emplace(std::forward<ArgsT>(args)...);
      }
      // bfs load balancing
      else {
        _workers[me].queue.push(Closure{std::forward<ArgsT>(args)...});
      }
      return;
    }
  }

  std::scoped_lock lock(_mutex);
  
  if(_idlers.empty()){
    _queue.push(Closure{std::forward<ArgsT>(args)...});
  } 
  else{
    Worker* w = _idlers.back();
    _idlers.pop_back();
    w->ready = true;
    w->cache.emplace(std::forward<ArgsT>(args)...);
    w->cv.notify_one();   
  }
}

// Procedure: batch
template <typename Closure>
void WorkStealingThreadpool<Closure>::batch(std::vector<Closure>&& tasks) {

  if(tasks.empty()) {
    return;
  }

  //no worker thread available
  if(num_workers() == 0){
    for(auto &t: tasks){
      t();
    }
    return;
  }
  
  // caller is not the owner
  if(auto tid = std::this_thread::get_id(); tid != _owner){

    // the caller is the worker of the threadpool
    if(auto itr = _worker_maps.find(tid); itr != _worker_maps.end()){

      unsigned me = itr->second;

      size_t i = 0;

      if(!_workers[me].cache) {
        _workers[me].cache = std::move(tasks[i++]);
      }

      for(; i<tasks.size(); ++i) {
        _workers[me].queue.push(std::move(tasks[i]));
      }

      return;
    }
  }

  std::scoped_lock lock(_mutex);

  size_t N = std::min(tasks.size(), _idlers.size());

  for(size_t k=N; k<tasks.size(); ++k) {
    _queue.push(std::move(tasks[k]));
  }

  for(size_t i=0; i<N; ++i) {
    Worker* w = _idlers.back();
    _idlers.pop_back();
    w->ready = true;
    w->cache = std::move(tasks[i]);
    w->cv.notify_one();   
  }
} 

};  // end of namespace tf. ---------------------------------------------------





