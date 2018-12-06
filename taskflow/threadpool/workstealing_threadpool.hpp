// 2018/12/05 - modified by Tsung-Wei Huang
//   - refactored the code
//   - replaced idler storage with lock-free queue
//   - added load balancing heuristics
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

// Class: WorkStealingQueue
// Unbounded work stealing queue implementation based on the following paper:
// David Chase and Yossi Lev. Dynamic circular work-stealing deque.
// In SPAA ’05: Proceedings of the seventeenth annual ACM symposium
// on Parallelism in algorithms and architectures, pages 21–28,
// New York, NY, USA, 2005. ACM.
template <typename T>
class WorkStealingQueue {

  constexpr static int64_t cacheline_size = 64;

  using storage_type = std::aligned_storage_t<sizeof(T), cacheline_size>;

  struct Array {

    int64_t C;
    storage_type* S;

    Array(int64_t c) : 
      C {c},
      S {new storage_type[C]} {
      for(int64_t i=0; i<C; ++i) {
        ::new (std::addressof(S[i])) T();
      }
    }

    ~Array() {
      for(int64_t i=0; i<C; ++i) {
        reinterpret_cast<T*>(std::addressof(S[i]))->~T();
      }
      delete [] S;
    }

    int64_t capacity() const noexcept {
      return C;
    }
    
    template <typename O>
    void push(int64_t i, O&& o) noexcept {
      T* ptr = reinterpret_cast<T*>(std::addressof(S[i & (C-1)]));
      *ptr = std::forward<O>(o); 
    }

    T pop(int64_t i) noexcept {
      return *reinterpret_cast<T*>(std::addressof(S[i & (C-1)]));
    }

    Array* resize(int64_t b, int64_t t) {
      Array* ptr = new Array {2*C};
      for(int64_t i=t; i!=b; ++i) {
        ptr->push(i, pop(i));
      }
      return ptr;
    }

  };

  alignas(cacheline_size) std::atomic<int64_t> _top;
  alignas(cacheline_size) std::atomic<int64_t> _bottom;
  alignas(cacheline_size) std::atomic<Array*> _array;
  std::vector<Array*> _garbage;

  public:

    WorkStealingQueue(int64_t = 4096);
    ~WorkStealingQueue();

    bool empty() const noexcept;

    int64_t size() const noexcept;
    int64_t capacity() const noexcept;
    
    template <typename O>
    void push(O&&);

    std::optional<T> pop();
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


// Class: WorkStealingThreadpool
template <typename Closure>
class WorkStealingThreadpool {

  struct Worker {
    std::condition_variable cv;
    WorkStealingQueue<Closure> queue;
    std::optional<Closure> cache;
    bool exit  {false};
    bool ready {false};
    uint64_t seed;
    unsigned victim_hint;
  };

  public:

    WorkStealingThreadpool(unsigned);
    ~WorkStealingThreadpool();
    
    size_t num_tasks() const;
    size_t num_workers() const;

    bool is_owner() const;

    template <typename... ArgsT>
    void emplace(ArgsT&&...);

    void batch(std::vector<Closure>&&);

  private:
    
    const std::thread::id _owner  {std::this_thread::get_id()};
    const int load_balancing_factor {4};

    mutable std::mutex _mutex;

    std::vector<Worker> _workers;
    std::vector<std::thread> _threads;

    std::unordered_map<std::thread::id, unsigned> _worker_maps;

    WorkStealingQueue<Worker*> _idlers;
    WorkStealingQueue<Closure> _queue;

    void _spawn(unsigned);
    void _shutdown();
    void _balance_load(unsigned);
    
    unsigned _next_power_of_2(unsigned) const;
    unsigned _randomize(uint64_t&) const;
    unsigned _fast_modulo(uint32_t, uint32_t) const;

    std::optional<Closure> _steal(unsigned);
};

// Constructor
template <typename Closure>
WorkStealingThreadpool<Closure>::WorkStealingThreadpool(unsigned N) : 
  _workers {N},
  _idlers  {_next_power_of_2(std::max(2u, N))} {
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
}

// Function: _randomize
// Generate the random output (using the PCG-XSH-RS scheme)
template <typename Closure>
unsigned WorkStealingThreadpool<Closure>::_randomize(uint64_t& state) const {
  uint64_t current = state;
  state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
  return static_cast<unsigned>((current ^ (current >> 22)) >> (22 + (current >> 61)));
}

// Function: _fast_modulo
// Perfrom fast modulo operation (might be biased but it's ok for our heuristics)
// http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
template <typename Closure>
unsigned WorkStealingThreadpool<Closure>::_fast_modulo(uint32_t x, uint32_t N) const {
  return ((uint64_t) x * (uint64_t) N) >> 32;
}

// Function: _next_power_of_2
template <typename Closure>
unsigned WorkStealingThreadpool<Closure>::_next_power_of_2(unsigned n) const {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
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
      w.victim_hint = (i + 1) % N;
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
          lock.lock();
          if(_queue.empty()) {
            w.ready = false;
            _idlers.push(&w);
            while(!w.ready && !w.exit) {
              w.cv.wait(lock);
            }
          }
          lock.unlock();

          if(w.cache) {
            std::swap(t, w.cache);
          }
        }

        while(t) {
          (*t)();
          if(w.cache) {
            t = std::move(*w.cache);
            w.cache = std::nullopt;
          }
          else {
            t = w.queue.pop();
          }
        }
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

// Procedure: balance_load
template <typename Closure>
void WorkStealingThreadpool<Closure>::_balance_load(unsigned me) {

  int factor = load_balancing_factor;
  
  while(_workers[me].queue.size() > factor) {
    if(auto idler = _idlers.steal(); idler) {
      (*idler)->ready = true;
      (*idler)->victim_hint = me;
      (*idler)->cache = _workers[me].queue.steal();
      (*idler)->cv.notify_one();
      factor += load_balancing_factor;
    }
    else break;
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
    unsigned victim = _workers[thief].victim_hint;

    for(unsigned i=0; i<_workers.size(); i++){

      if(victim != thief) {
        if(task = _workers[victim].queue.steal(); task){
          _workers[thief].victim_hint = victim;
          return task;
        }
      }

      victim += 1;
      if(victim == _workers.size()){
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

      if(_workers[me].cache) {
        _workers[me].queue.push(Closure{std::forward<ArgsT>(args)...});
      }
      else {
        _workers[me].cache.emplace(std::forward<ArgsT>(args)...);
      }
      
      // load balancing
      _balance_load(me);
      
      return;
    }
  }

  if(auto idler = _idlers.steal(); idler) {
    (*idler)->ready = true;
    (*idler)->cache.emplace(std::forward<ArgsT>(args)...);
    (*idler)->cv.notify_one(); 
  }
  else {
    std::scoped_lock lock(_mutex);
    _queue.push(Closure{std::forward<ArgsT>(args)...});
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
      
      // load balancing 
      _balance_load(me);

      return;
    }
  }
  
  {
    std::scoped_lock lock(_mutex);
    
    for(auto& task : tasks) {
      _queue.push(std::move(task));
    }
  }

  while(!_queue.empty()) {
    if(auto idler = _idlers.steal(); idler) {
      (*idler)->ready = true;
      (*idler)->cache = _queue.steal();
      (*idler)->cv.notify_one();
    }
    else break;
  }
} 

};  // end of namespace tf. ---------------------------------------------------





