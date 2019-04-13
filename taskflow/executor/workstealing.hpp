// 2019/04/11 - modified by Tsung-Wei Huang
//   - renamed to executor
//
// 2019/03/30 - modified by Tsung-Wei Huang
//   - added consensus sleep-loop stragety 
//
// 2019/03/21 - modified by Tsung-Wei Huang
//   - removed notifier
//   - implemented a new scheduling strategy
//     (a thread will sleep as long as all threads meet the constraints)
// 
// 2019/02/15 - modified by Tsung-Wei Huang
//   - batch to take reference not move
//
// 2019/02/10 - modified by Tsung-Wei Huang
//   - modified WorkStealingExecutor with notifier
//   - modified the stealing loop
//   - improved the performance
//
// 2019/01/03 - modified by Tsung-Wei Huang
//   - updated the load balancing strategy
//
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
#include <deque>
#include <optional>
#include <thread>
#include <algorithm>
#include <set>
#include <numeric>
#include <cassert>
#include "notifier.hpp"

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
@class: WorkStealingExecutor

@brief Executor that implements an efficient work stealing algorithm.

@tparam Closure closure type
*/
template <typename Closure>
class WorkStealingExecutor {
    
  struct Worker {
    WorkStealingQueue<Closure> queue;
    std::optional<Closure> cache;
  };
    
  struct PerThread {
    WorkStealingExecutor* pool {nullptr}; 
    int worker_id {-1};
    uint64_t seed {std::hash<std::thread::id>()(std::this_thread::get_id())};
  };

  struct Consensus {
    const unsigned threshold;
    std::atomic<unsigned> count;
    Consensus(unsigned N) : threshold {N/2}, count {N} {}
    void consent() { ++count; }
    void dissent() { --count; }
    operator bool () const { return count > threshold; }
  };
  
  public:
    
    /**
    @brief constructs the executor with a given number of worker threads

    @param N the number of worker threads
    */
    explicit WorkStealingExecutor(unsigned N);

    /**
    @brief destructs the executor

    Destructing the executor will immediately force all worker threads to stop.
    The executor does not guarantee all tasks to finish upon destruction.
    */
    ~WorkStealingExecutor();
    
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

    @param closures a vector of closures
    */
    void batch(std::vector<Closure>& closures);

  private:
    
    const std::thread::id _owner {std::this_thread::get_id()};

    std::mutex _mutex;

    std::vector<Worker> _workers;
    std::vector<Notifier::Waiter> _waiters;
    std::vector<unsigned> _coprimes;
    std::vector<std::thread> _threads;

    WorkStealingQueue<Closure> _queue;
    
    std::atomic<unsigned> _num_idlers {0};
    std::atomic<bool> _done {false};

    Consensus _consensus;
    Notifier _notifier;
    
    void _spawn(unsigned);
    void _balance_load(unsigned);

    unsigned _randomize(uint64_t&) const;
    unsigned _fast_modulo(unsigned, unsigned) const;
    unsigned _find_victim(unsigned) const;
    
    PerThread& _per_thread() const;

    std::optional<Closure> _steal(unsigned);

    bool _wait_for_tasks(unsigned, std::optional<Closure>&);
};

// Constructor
template <typename Closure>
WorkStealingExecutor<Closure>::WorkStealingExecutor(unsigned N) : 
  _workers   {N},
  _waiters   {N},
  _consensus {N},
  _notifier  {_waiters} {
  
  for(unsigned i = 1; i <= N; i++) {
    unsigned a = i;
    unsigned b = N;
    // If GCD(a, b) == 1, then a and b are coprimes.
    if(std::gcd(a, b) == 1) {
      _coprimes.push_back(i);
    }
  }

  _spawn(N);
}

// Destructor
template <typename Closure>
WorkStealingExecutor<Closure>::~WorkStealingExecutor() {

  _done = true;
  _notifier.notify(true);
  
  for(auto& t : _threads){
    t.join();
  } 

}

// Function: _per_thread
template <typename Closure>
typename WorkStealingExecutor<Closure>::PerThread& 
WorkStealingExecutor<Closure>::_per_thread() const {
  thread_local PerThread pt;
  return pt;
}

// Function: _randomize
template <typename Closure>
unsigned WorkStealingExecutor<Closure>::_randomize(uint64_t& state) const {
  uint64_t current = state;
  state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
  // Generate the random output (using the PCG-XSH-RS scheme)
  return static_cast<unsigned>((current ^ (current >> 22)) >> (22 + (current >> 61)));
}

// http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
template <typename Closure>
unsigned WorkStealingExecutor<Closure>::_fast_modulo(unsigned x, unsigned N) const {
  return ((uint64_t) x * (uint64_t) N) >> 32;
}

// Procedure: _spawn
template <typename Closure>
void WorkStealingExecutor<Closure>::_spawn(unsigned N) {
  
  // Lock to synchronize all workers before creating _worker_mapss
  for(unsigned i=0; i<N; ++i) {
    _threads.emplace_back([this, i, N] () -> void {

      PerThread& pt = _per_thread();  
      pt.pool = this;
      pt.worker_id = i;
    
      auto& worker = _workers[i];

      std::optional<Closure> t;
      
      bool active {true};
      _consensus.dissent();
      
      // must use 1 as condition instead of !done
      while(1) {
        
        // execute the tasks.
        run_task:

        assert(_consensus.count <= N && 0 <= _consensus.count);

        if(!active && t) {
          active = true;
          _consensus.dissent();
        }

        // exploit
        while(t) {
          (*t)();
          if(worker.cache) {
            t = std::move(worker.cache);
            worker.cache = std::nullopt;
          }
          else {
            t = worker.queue.pop();
          }
        }

        // explore
        while (1) {
          if(auto victim = _find_victim(i); victim != N) {
            t = victim == i ? _queue.steal() : _workers[victim].queue.steal();
            if(t) {
              goto run_task;
            }
          }
          else break;
        }
        
        if(active) {
          active = false;
          _consensus.consent();
        }

        // wait for tasks
        if(_consensus) {
          if(_wait_for_tasks(i, t) == false) {
            break;
          }
        }
      }

      if(active) {
        active = false;
        _consensus.consent();
      }
      
    });     
  }
}

// Function: is_owner
template <typename Closure>
bool WorkStealingExecutor<Closure>::is_owner() const {
  return std::this_thread::get_id() == _owner;
}

// Function: num_workers
template <typename Closure>
size_t WorkStealingExecutor<Closure>::num_workers() const { 
  return _workers.size();  
}

// Procedure: _balance_load
template <typename Closure>
void WorkStealingExecutor<Closure>::_balance_load(unsigned me) {

  auto n = _workers[me].queue.size();

  // return if no idler - this might not be the right value
  // but it doesn't affect the correctness
  if(n <= 4) {
    return;
  }

  // try with probability 1/n
  if(_fast_modulo(_randomize(_workers[me].seed), n) == 0) {
    _notifier.notify(false);
  }
}

// Function: _non_empty_queue
template <typename Closure>
unsigned WorkStealingExecutor<Closure>::_find_victim(unsigned thief) const {

  assert(_workers[thief].queue.empty());
  
  auto &pt = _per_thread();
  auto rnd = _randomize(pt.seed);
  auto inc = _coprimes[_fast_modulo(rnd, _coprimes.size())];
  auto vtm = _fast_modulo(rnd, _workers.size());

  // try stealing a task from other workers
  for(unsigned i=0; i<_workers.size(); ++i){

    if((thief == vtm && !_queue.empty()) ||
       (thief != vtm && !_workers[vtm].queue.empty())) {
      return vtm;
    }

    if(vtm += inc; vtm >= _workers.size()) {
      vtm -= _workers.size();
    }
  }

  return _workers.size();
}

// Function: _steal
template <typename Closure>
std::optional<Closure> WorkStealingExecutor<Closure>::_steal(unsigned thief) {
  
  assert(_workers[thief].queue.empty());
  
  auto &pt = _per_thread();
  auto rnd = _randomize(pt.seed);
  auto inc = _coprimes[_fast_modulo(rnd, _coprimes.size())];
  auto vtm = _fast_modulo(rnd, _workers.size());
  
  std::optional<Closure> task;

  // try stealing a task from other workers
  for(unsigned i=0; i<_workers.size(); ++i){

    task = vtm == thief ? _queue.steal() : _workers[vtm].queue.steal();

    if(task) {
      _workers[thief].last_vtm = vtm;
      return task;
    }

    if(vtm += inc; vtm >= _workers.size()) {
      vtm -= _workers.size();
    }
  }

  return std::nullopt; 
}

// Function: _wait_for_tasks
template <typename Closure>
bool WorkStealingExecutor<Closure>::_wait_for_tasks(
  unsigned i, 
  std::optional<Closure>& t
) {

  assert(!t);

  _notifier.prepare_wait(&_waiters[i]);
  
  // check again.
  if(auto victim = _find_victim(i); victim != _workers.size()) {
    _notifier.cancel_wait(&_waiters[i]);
    t = _workers[victim].queue.steal();
    return true;
  }

  if(auto I = ++_num_idlers; _done && I == _workers.size()) {
    _notifier.cancel_wait(&_waiters[i]);
    if(_find_victim(i) != _workers.size()) {
      --_num_idlers;
      return true;
    }
    _notifier.notify(true);
    return false;
  }

  _notifier.commit_wait(&_waiters[i]);
  --_num_idlers;

  return true;
}

// Procedure: emplace
template <typename Closure>
template <typename... ArgsT>
void WorkStealingExecutor<Closure>::emplace(ArgsT&&... args){
  
  //no worker thread available
  if(num_workers() == 0){
    Closure{std::forward<ArgsT>(args)...}();
    return;
  }

  auto& pt = _per_thread();
  
  // caller is a worker to this pool
  if(pt.pool == this) {
    if(!_workers[pt.worker_id].cache) {
      _workers[pt.worker_id].cache.emplace(std::forward<ArgsT>(args)...);
      return;
    }
    else {
      _workers[pt.worker_id].queue.push(Closure{std::forward<ArgsT>(args)...});
    }
  }
  // other threads
  else {
    std::scoped_lock lock(_mutex);
    _queue.push(Closure{std::forward<ArgsT>(args)...});
  }

  _notifier.notify(false);
}

// Procedure: batch
template <typename Closure>
void WorkStealingExecutor<Closure>::batch(std::vector<Closure>& tasks) {

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

  auto& pt = _per_thread();

  if(pt.pool == this) {
    
    size_t i = 0;

    if(!_workers[pt.worker_id].cache) {
      _workers[pt.worker_id].cache = std::move(tasks[i++]);
    }

    for(; i<tasks.size(); ++i) {
      _workers[pt.worker_id].queue.push(std::move(tasks[i]));
      _notifier.notify(false);
    }

    return;
  }
  
  {
    std::scoped_lock lock(_mutex);

    for(size_t k=0; k<tasks.size(); ++k) {
      _queue.push(std::move(tasks[k]));
      _notifier.notify(false);
    }
  }

} 

}  // end of namespace tf. ---------------------------------------------------





