// 2018/10/05 - modified by Chun-Xun
//   - adopted the new threadpool model   
//
// 2018/09/21 - modified by Tsung-Wei and Chun-Xun
//   - refactored the code
//  
// 2018/09/12 - created by Tsung-Wei Huang and Chun-Xun Lin
//
// Implemented PrivatizedThreadpool using the data structre inspired
// Eigen CXX/Threadpool.

// TODO
// - double check whether we can use std::forward<Args>(args)... in enqueue/dequeue
// - can we replace lock with CAS
// - refactored the WorkQueue class ...
// - add more example to threadpool and use std::future to mimic the control flow
// - atomic add problem (extremely slow)

#pragma once

#include <iostream>
#include <functional>
#include <vector>
#include <mutex>
#include <thread>
#include <stdexcept>
#include <condition_variable>
#include <memory>
#include <future>
#include <optional>
#include <unordered_set>
#include <unordered_map>

#include "move_on_copy.hpp"

namespace tf {

// ========
// Guannan's implementation
// ========

//-------------RunQueue-----------------
template <typename T, unsigned N>
class RunQueue {

  static_assert((N & (N - 1)) == 0, "N must be power of two");
  static_assert(N > 2, "N must be larger than two");
  
  constexpr static unsigned IDX_MASK = N - 1;
  constexpr static unsigned POS_MASK = (N << 1) - 1;

  struct Entry {
    std::atomic<uint8_t> state;
    T w;
  };

  enum : uint8_t {
    EMPTY,
    BUSY,
    READY
  };

  public:

    RunQueue();
    
    bool push_back(T&&);
    bool push_front(T&&);

    bool push_front(T&);
    bool pop_front(std::optional<T>&);
    bool push_back(T&);
    bool pop_back(std::optional<T>&);
    bool empty() const;

  private:

    std::mutex _mutex;

    std::atomic<unsigned> _front;
    std::atomic<unsigned> _back;

    Entry _array[N];
};

// Constructor    
template <typename T, unsigned N>
RunQueue<T, N>::RunQueue() {

  _front.store(0, std::memory_order_relaxed);
  _back.store(0, std::memory_order_relaxed);

  for(unsigned i=0; i<N; ++i) {
    _array[i].state.store(EMPTY, std::memory_order_relaxed);
  }
}

// insert item w to the beginning of the queue and return true if inserted 
// or false otherwise.
// this function can only be called by the owner thread.
template <typename T, unsigned N>
bool RunQueue<T, N>::push_front(T& w) {

  auto front = _front.load(std::memory_order_relaxed);
  auto& item = _array[front & IDX_MASK];
  auto state = item.state.load(std::memory_order_relaxed);
  
  if(state != EMPTY || 
    !item.state.compare_exchange_strong(state, BUSY, std::memory_order_acquire)) {
    return false;
  }

  _front.store((front + 1) & POS_MASK, std::memory_order_relaxed);
  item.w = std::move(w);
  item.state.store(READY, std::memory_order_release);

  return true;
}

template <typename T, unsigned N>
bool RunQueue<T, N>::push_front(T&& w) {
  return push_front(w);
}

// pop the first item out of the queue and store it to w
template <typename T, unsigned N>
bool RunQueue<T, N>::pop_front(std::optional<T>& w) {

  if(empty()) {
    return false;
  }

  auto front = _front.load(std::memory_order_relaxed);
  auto& item = _array[(front - 1) & IDX_MASK];
  auto state = item.state.load(std::memory_order_relaxed);

  if(state != READY || 
    !item.state.compare_exchange_strong(state, BUSY, std::memory_order_acquire)) {
    return false;
  }
  
  _front.store((front - 1) & POS_MASK, std::memory_order_relaxed);
  w = std::move(item.w); 
  item.state.store(EMPTY, std::memory_order_release);

  return true;
}

// add an item at the end of the queue
template <typename T, unsigned N>
bool RunQueue<T, N>::push_back(T& w) {

  std::scoped_lock lock(_mutex);

  auto back  = _back.load(std::memory_order_relaxed);
  auto& item = _array[(back - 1) & IDX_MASK];
  auto state = item.state.load(std::memory_order_relaxed);

  if(state != EMPTY ||
    !item.state.compare_exchange_strong(state, BUSY, std::memory_order_acquire)) {
    return false;
  }
  
  _back.store((back - 1) & POS_MASK, std::memory_order_relaxed);
  item.w = std::move(w);
  item.state.store(READY, std::memory_order_release);

  return true;
}

// add an item at the end of the queue
template <typename T, unsigned N>
bool RunQueue<T, N>::push_back(T&& w) {
  return push_back(w);
}

// pop_back removes and returns the last elements in the queue.
// Can fail spuriously.
template <typename T, unsigned N>
bool RunQueue<T, N>::pop_back(std::optional<T>& w) {

  if(empty()) {
    return false;
  }

  std::unique_lock lock(_mutex, std::try_to_lock);

  if (!lock) {
    return false;
  }

  auto back  = _back.load(std::memory_order_relaxed);
  auto& item = _array[back & IDX_MASK];
  auto state = item.state.load(std::memory_order_relaxed);

  if (state != READY ||
     !item.state.compare_exchange_strong(state, BUSY, std::memory_order_acquire)) {
    return false;
  }

  w = std::move(item.w);
  _back.store((back + 1) & POS_MASK, std::memory_order_relaxed);
  item.state.store(EMPTY, std::memory_order_release);

  return true;
}

// EMPTY tests whether container is empty.
// Can be called by any thread at any time.
template <typename T, unsigned N>
bool RunQueue<T, N>::empty() const { 
  return _front.load(std::memory_order_relaxed) ==
         _back.load(std::memory_order_relaxed);
}


//------------------------------------------------------------
//------------------Class:: PrivatizedThreadpool-------------

template <typename Task>
class PrivatizedThreadpool {

  struct Worker{
    std::mutex mu;
    std::condition_variable cv;
    bool ready {true};
    RunQueue<Task, 1024> queue;
    bool exit {false};
    std::optional<Task> task {std::nullopt};
  };

  struct PerThread {
    constexpr PerThread() : pool(nullptr), rand(0), thread_id(-1) { }
    PrivatizedThreadpool* pool;  // Parent pool, or null for normal threads.
    uint64_t rand;  // Random generator state.
    int thread_id;  // Worker thread index in pool.
  };


  public:

    PrivatizedThreadpool(unsigned);
    ~PrivatizedThreadpool();

    size_t num_tasks() const;
    size_t num_workers() const;

    bool is_owner() const;

    template <typename... ArgsT>
    void emplace(ArgsT&&...);

  private:

    //std::mutex _cout_mutex; //temporary mutex to protect cout
    
    //-------private members
    const std::thread::id _owner {std::this_thread::get_id()};

    mutable std::mutex _mutex;
    std::vector<Task> _tasks;

    std::vector<std::thread> _threads;
    std::vector<Worker> _workers;
   
    std::atomic<unsigned> _num_idlers {0};
    // TODO: remove this?
    unsigned _worker_num {0}; //try to avoid repetitive calls of size()
    std::atomic<size_t> _next_queue {0}; 
    
    // TODO: can be removed?
    bool _done {false};
    std::condition_variable _complete;

    //-------private functions  
    std::optional<size_t> _nonempty_worker_queue() const;

    bool _steal(std::optional<Task>&);

    void _shutdown();
    void _spawn(unsigned);

    // TODO: prefer reference 
    inline PerThread* GetPerThread() {
      thread_local PerThread per_thread_;
      PerThread* pt = &per_thread_;
      return pt;
    }
    
    // TODO: need another protocol to replace this
    inline unsigned Rand(uint64_t* state) {
      uint64_t current = *state;
      // Update the internal state
      *state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
      // Generate the random output (using the PCG-XSH-RS scheme)
      return static_cast<unsigned>((current ^ (current >> 22)) >> (22 + (current >> 61)));
    }

};  // class PrivatizedThreadpool. --------------------------------------

// Function: _nonempty_worker_queue
template <typename Task>
std::optional<size_t> PrivatizedThreadpool<Task>::_nonempty_worker_queue() const {
  for(size_t i=0;i <_workers.size(); ++i){
    if(!_workers[i].queue.empty()){
      return i;
    }
  }
  return {};
}


// Function: _steal 
template <typename Task>
bool PrivatizedThreadpool<Task>::_steal(std::optional<Task>& w){

  PerThread* pt = GetPerThread();
  const auto queue_num = _worker_num;
  unsigned r = Rand(&pt->rand);
  unsigned victim = r % queue_num;
  
  for(size_t i=0; i<queue_num; i++){
    if(_workers[victim].queue.pop_back(w)){
      return true;
    }
    victim += 1;
    if(victim >= queue_num){
      victim -= queue_num;
    }
  }

  {
    std::scoped_lock lock(_mutex);
    if(!_tasks.empty()){
      w = std::move(_tasks.back());
      _tasks.pop_back();
      return true;
    }
  }

  return false;
}

// Constructor
template <typename Task>
PrivatizedThreadpool<Task>::PrivatizedThreadpool(unsigned N): _workers {N}, _worker_num(N) {
  _spawn(N);
}

// Destructor
template <typename Task>
PrivatizedThreadpool<Task>::~PrivatizedThreadpool(){
  _shutdown();
}

// Function: is_owner
template <typename Task>
bool PrivatizedThreadpool<Task>::is_owner() const {
  return std::this_thread::get_id() == _owner;
}

// Function: num_tasks
template <typename Task>
size_t PrivatizedThreadpool<Task>::num_tasks() const { 
  return _tasks.size(); 
}

// Function: num_workers
template <typename Task>
size_t PrivatizedThreadpool<Task>::num_workers() const { 
  return _worker_num;  
}

// Function: shutdown
template <typename Task>
void PrivatizedThreadpool<Task>::_shutdown(){

  assert(is_owner());
  
  // TODO: here can be removed
  //std::cout << "Shutting down..." << std::endl;
  {
    std::unique_lock lock(_mutex);
    _done = true;
    while(_num_idlers != _worker_num){
      _complete.wait(lock);
    }
  }
  
  //std::cout << "Freeing all workers..." << std::endl;
  for(auto& w: _workers){ 
    std::scoped_lock worker_lock(w.mu);//need a lock here, tested by unittest 
    w.ready = true;
    w.exit = true;
    w.cv.notify_one();
  }

  for(auto& t : _threads){
    t.join();
  }

  _threads.clear();
  _workers.clear();
  //std::cout << "Threadpool terminated..." << std::endl;
}


// Function: _spawn 
template <typename Task>
void PrivatizedThreadpool<Task>::_spawn(unsigned N) {
  //assume constructor has intialized the worker vector 
 
  assert(is_owner());

  for(size_t i=0; i<N; i++){
    _threads.emplace_back([this, i=i]() -> void{
      
      PerThread* pt = GetPerThread();
      pt->pool = this;
      pt->rand = std::hash<std::thread::id>()(std::this_thread::get_id());
      pt->thread_id = i;

      std::optional<Task> t;
      Worker& w = _workers[i];

      while(!w.exit){
        //fail to get task from private queue or steal task from other queue
        if(!w.queue.pop_front(t) && !_steal(t)){ 
    
          //owner may insert tasks to private queue after the check 
          if(++_num_idlers == num_workers()){
            if(auto ret = _nonempty_worker_queue(); ret.has_value()){
              --_num_idlers;
              _workers[*ret].cv.notify_one(); //pesimistic
              continue;
            }
            
            // TODO: may be removed?
            {
              //need to guarantee that the owner is waiting on shutdown       
              std::scoped_lock lock(_mutex); 
              if(_done){
              _complete.notify_one(); //notify owner to wrap up
              }
            }
          }
    
          {
            std::unique_lock worker_lock(w.mu);
            
            if(w.queue.empty() && _tasks.empty()){
              w.ready = false;
            }        

            while(!w.exit && !w.ready){
              w.cv.wait(worker_lock);
            }
            w.ready = true;
          } //implicit desctruction of worker lock
          t = std::move(w.task);
          w.task = std::nullopt; 
          --_num_idlers;           
        }

        while(t.has_value()){ 
          (*t)();
          if(w.task){
            t = std::move(w.task);
            w.task = std::nullopt;
          }
          else{
            t = std::nullopt;
          }  
        } 

      }

      //std::cout << "Exit " << i <<std::endl;


    });
  }

}

template <typename Task>
template <typename... ArgsT>
void PrivatizedThreadpool<Task>::emplace(ArgsT&&... args){

  //no worker thread available
  if(num_workers() == 0){
    Task{std::forward<ArgsT>(args)...}();
    return;
  }

  //std::cout << "Inserting task" << std::endl;

  Task t {std::forward<ArgsT>(args)...};

  PerThread* pt = GetPerThread();
  size_t target = _next_queue; //round robin
  bool ret = false;

  //workers
  if (pt->pool == this) {
    target = pt->thread_id;
    Worker& w = _workers[target];   
    if(!w.task.has_value()){ //directly insert if it owns the queue
      w.task = std::move(t);
      return;
    }
    else{
      if(auto ret = w.queue.push_front(std::move(t)); ret) {
        return;
      }
      else{
        std::scoped_lock lock(_mutex);
        _tasks.push_back(std::move(t));
        return;
      }
    }
  } 
  //non workers
  else {

    // TODO: modulo operation can be slow here?
    target = (++_next_queue) % _worker_num; 
    Worker& w = _workers[target];
    _next_queue = target;
    

    {//avoid racing read
      std::scoped_lock worker_lock(w.mu);

      if(w.ready){ //worker busy
        ret = w.queue.push_back(std::move(t));
        if(!ret){
          std::scoped_lock lock(_mutex);
          _tasks.push_back(std::move(t));
        }    
      }
      else{
        w.task = std::move(t);
        w.ready = true;
        w.cv.notify_one();
      }
    }
    return;
  }
}  



// ==========
// Chun-Xun's implementation
// ==========
/*
// ----------------------------------------------------------------------------  
// Privatized queue of worker. The lock-free queue is inspired by 
// http://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue
template<typename T, size_t buffer_size>
class PrivatizedTaskQueue {

  public:

  // TODO replaced the new with std::array
  PrivatizedTaskQueue()
    : buffer_(new cell_t [buffer_size])
    , buffer_mask_(buffer_size - 1)
  {
    for (size_t i = 0; i != buffer_size; i += 1)
      buffer_[i].sequence_.store(i, std::memory_order_relaxed);
    enqueue_pos_.store(0, std::memory_order_relaxed);
    dequeue_pos_.store(0, std::memory_order_relaxed);
  }

  ~PrivatizedTaskQueue(){
    delete [] buffer_;
  }

  bool enqueue(T& data){
    cell_t* cell;
    size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    for (;;)
    {
      cell = &buffer_[pos & buffer_mask_];
      size_t seq = 
        cell->sequence_.load(std::memory_order_acquire);
      intptr_t dif = (intptr_t)seq - (intptr_t)pos;
      if (dif == 0)
      {
        //if (enqueue_pos_.compare_exchange_weak 
        if (enqueue_pos_.compare_exchange_strong
            (pos, pos + 1, std::memory_order_relaxed))
          break;
      }
      else if (dif < 0)
        return false;
      else
        pos = enqueue_pos_.load(std::memory_order_relaxed);
    }
    cell->data_ = std::move(data);
    cell->sequence_.store(pos + 1, std::memory_order_release);
    return true;
  }

  bool dequeue(std::optional<T>& data){
    cell_t* cell;
    size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
    for (;;)
    {
      cell = &buffer_[pos & buffer_mask_];
      size_t seq = 
        cell->sequence_.load(std::memory_order_acquire);
      intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);
      if (dif == 0)
      {
        //if (dequeue_pos_.compare_exchange_weak 
        if (dequeue_pos_.compare_exchange_strong
            (pos, pos + 1, std::memory_order_relaxed))
          break;
      }
      else if (dif < 0)
        return false;
      else
        pos = dequeue_pos_.load(std::memory_order_relaxed);
    }
    data = std::move(cell->data_);
    cell->sequence_.store
      (pos + buffer_mask_ + 1, std::memory_order_release);
    return true;
  }

  bool empty() const {
    return enqueue_pos_.load(std::memory_order_relaxed) == 
           dequeue_pos_.load(std::memory_order_relaxed);
  };

  private:
  
  struct cell_t {
    std::atomic<size_t>   sequence_;
    T                     data_;
  };

  static size_t const     cacheline_size = 64;
  typedef char            cacheline_pad_t [cacheline_size];

  cacheline_pad_t         pad0_;
  cell_t* const           buffer_;
  size_t const            buffer_mask_;
  cacheline_pad_t         pad1_;
  std::atomic<size_t>     enqueue_pos_;
  cacheline_pad_t         pad2_;
  std::atomic<size_t>     dequeue_pos_;
  cacheline_pad_t         pad3_;
}; 


// ----------------------------------------------------------------------------

// Class: PrivatizedThreadpool
template <typename Task>
class PrivatizedThreadpool {

  struct Worker{
    std::condition_variable cv;
    PrivatizedTaskQueue<Task, 1024> queue;
    bool exit {false};
    std::optional<Task> cache;
  };

  public:

    PrivatizedThreadpool(unsigned);
    ~PrivatizedThreadpool();

    size_t num_tasks() const;
    size_t num_workers() const;

    bool is_owner() const;

    template <typename... ArgsT>
    void emplace(ArgsT&&...);

  private:
    
    const std::thread::id _owner {std::this_thread::get_id()};

    mutable std::mutex _mutex;

    std::vector<Task> _tasks;
    std::vector<std::thread> _threads;
    std::vector<Worker> _workers;
    
    std::unordered_map<std::thread::id, size_t> _worker_maps;

    size_t _num_idlers {0};
    size_t _next_queue {0};
    
    std::optional<size_t> _nonempty_worker_queue() const;

    void _xorshift32(uint32_t&);
    bool _steal(std::optional<Task>&, uint32_t&);

    void _shutdown();
    void _spawn(unsigned);

};  // class PrivatizedThreadpool. --------------------------------------


// Function: _nonempty_worker_queue
template <typename Task>
std::optional<size_t> PrivatizedThreadpool<Task>::_nonempty_worker_queue() const {
  for(size_t i=0;i <_workers.size(); ++i){
    if(!_workers[i].queue.empty()){
      return i;
    }
  }
  return {};
}

// Function: _xorshift32
template <typename Task>
void PrivatizedThreadpool<Task>::_xorshift32(uint32_t& x){
  // x must be non zero: https://en.wikipedia.org/wiki/Xorshift
  // Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" 
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
}

// Function: _steal 
template <typename Task>
bool PrivatizedThreadpool<Task>::_steal(std::optional<Task>& w, uint32_t& seed){
  _xorshift32(seed);
  const auto queue_num = _workers.size();
  auto victim = seed % queue_num;
  for(size_t i=0; i<queue_num; i++){
    if(_workers[victim].queue.dequeue(w)){
      return true;
    }
    victim += 1;
    if(victim >= queue_num){
      victim -= queue_num;
    }
  }
  return false;
}

// Constructor
template <typename Task>
PrivatizedThreadpool<Task>::PrivatizedThreadpool(unsigned N): _workers {N} {
  _spawn(N);
}

// Destructor
template <typename Task>
PrivatizedThreadpool<Task>::~PrivatizedThreadpool(){
  _shutdown();
}

// Function: is_owner
template <typename Task>
bool PrivatizedThreadpool<Task>::is_owner() const {
  return std::this_thread::get_id() == _owner;
}

// Function: num_tasks
template <typename Task>
size_t PrivatizedThreadpool<Task>::num_tasks() const { 
  return _tasks.size(); 
}

// Function: num_workers
template <typename Task>
size_t PrivatizedThreadpool<Task>::num_workers() const { 
  return _threads.size();  
}

// Function: shutdown
template <typename Task>
void PrivatizedThreadpool<Task>::_shutdown(){

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

  _threads.clear();  
  _workers.clear();
  _worker_maps.clear();
}

// Function: _spawn 
template <typename Task>
void PrivatizedThreadpool<Task>::_spawn(unsigned N) {

  assert(is_owner());

  // Lock to synchronize all workers before creating _worker_mapss
  std::scoped_lock lock(_mutex);

  for(size_t i=0; i<N; ++i){
    _threads.emplace_back([this, i=i]() -> void {

      std::optional<Task> t;
      Worker& w = (_workers[i]);
      uint32_t seed = i+1;
      std::unique_lock lock(_mutex, std::defer_lock);

      while(!w.exit){
        if(!w.queue.dequeue(t) && !_steal(t, seed)) {
          lock.lock();
          if(!_tasks.empty()) {
            t = std::move(_tasks.back());
            _tasks.pop_back();
          }
          else{
            if(++_num_idlers == num_workers()){
              // Last active thread checks if all queues are empty
              if(auto ret = _nonempty_worker_queue(); ret.has_value()){
                // if the nonempty queue is mine, continue to process tasks in queue
                if(*ret == i){
                  --_num_idlers;
                  lock.unlock();
                  continue;
                }
                // If any queue is not empty, notify the worker to process the tasks 
                _workers[*ret].cv.notify_one();
              }
            } 

            while(!w.exit && w.queue.empty() && _tasks.empty()){
              w.cv.wait(lock);
            }
            --_num_idlers;
          } 
          lock.unlock();
        } // End of pop_front 

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
      } // End of while ------------------------------------------------------
    });     

    _worker_maps.insert({_threads.back().get_id(), i});
  } // End of For ---------------------------------------------------------------------------------
}


template <typename Task>
template <typename... ArgsT>
void PrivatizedThreadpool<Task>::emplace(ArgsT&&... args){

  //no worker thread available
  if(num_workers() == 0){
    Task{std::forward<ArgsT>(args)...}();
    return;
  }

  Task t {std::forward<ArgsT>(args)...};
  
  // caller is not the owner
  if(auto tid = std::this_thread::get_id(); tid != _owner){
    // the caller is the worker of the threadpool
    if(auto itr = _worker_maps.find(tid); itr != _worker_maps.end()){
      if(!_workers[itr->second].cache.has_value()){
        _workers[itr->second].cache = std::move(t);
        return ;
      }
      if(!_workers[itr->second].queue.enqueue(t)){
        std::scoped_lock lock(_mutex);       
        _tasks.push_back(std::move(t));
      }
      return ;
    }
  }

  // owner thread or other threads
  auto id = (++_next_queue) % _workers.size();

  if(!_workers[id].queue.enqueue(t)){
    std::scoped_lock lock(_mutex);
    _tasks.push_back(std::move(t));
  }
  else{
    // Lock to make sure the worker will be notified
    std::scoped_lock lock(_mutex);
  }
  _workers[id].cv.notify_one();
} 

*/

};  // namespace tf -----------------------------------------------------------




