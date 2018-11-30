// 2018/11/28 - modified by Chun-Xun Lin
// 
// Added the method batch to insert a vector of tasks.
//
// 2018/10/10 - modified by Tsung-Wei
//   - merged with Guannan's implementation
//   - merged with Chun-Xun's implementation
// For now we will just leave privatized threadpool an independent interests.
// The speed curve is not reliable and can fluctuate a lot.
//
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
#include <array>

namespace tf {

/*// ========
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

template <typename Closure>
class PrivatizedThreadpool {

  struct Worker{
    std::mutex mutex;
    std::condition_variable cv;
    bool ready {true};
    bool exit {false};
    std::optional<Closure> task {std::nullopt};
    RunQueue<Closure, 1024> queue;
  };

  struct PerThread {
    PrivatizedThreadpool* pool {nullptr};
    uint64_t rand {0};
    int thread_id {-1};
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
    std::vector<Closure> _tasks;
    std::vector<std::thread> _threads;
    std::vector<Worker> _workers;
   
    std::atomic<unsigned> _num_idlers {0};
    std::atomic<size_t> _next_queue {0}; 
    
    std::optional<size_t> _nonempty_worker_queue() const;
    std::optional<Closure> _try_dequeue();

    bool _steal(std::optional<Closure>&);

    void _shutdown();
    void _spawn(unsigned);

    PerThread* _per_thread() const;
    
    unsigned _randomize(uint64_t*) const;

};  // class PrivatizedThreadpool. --------------------------------------

// Function: _nonempty_worker_queue
template <typename Closure>
std::optional<size_t> PrivatizedThreadpool<Closure>::_nonempty_worker_queue() const {
  for(size_t i=0;i <_workers.size(); ++i){
    if(!_workers[i].queue.empty()){
      return i;
    }
  }
  return {};
}
    
// Function: _per_thread
template <typename Closure>
typename PrivatizedThreadpool<Closure>::PerThread* 
PrivatizedThreadpool<Closure>::_per_thread() const {
  thread_local PerThread per_thread_;
  PerThread* pt = &per_thread_;
  return pt;
}

// Function: _randomize
template <typename Closure>
unsigned PrivatizedThreadpool<Closure>::_randomize(uint64_t* state) const {
  uint64_t current = *state;
  *state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
  return static_cast<unsigned>((current ^ (current >> 22)) >> (22 + (current >> 61)));
}

// Function: _steal 
template <typename Closure>
bool PrivatizedThreadpool<Closure>::_steal(std::optional<Closure>& w){

  PerThread* pt = _per_thread();
  const auto queue_num = _workers.size();
  unsigned r = _randomize(&pt->rand);
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
template <typename Closure>
PrivatizedThreadpool<Closure>::PrivatizedThreadpool(unsigned N): _workers {N} {
  _spawn(N);
}

// Destructor
template <typename Closure>
PrivatizedThreadpool<Closure>::~PrivatizedThreadpool(){
  _shutdown();
}

// Function: is_owner
template <typename Closure>
bool PrivatizedThreadpool<Closure>::is_owner() const {
  return std::this_thread::get_id() == _owner;
}

// Function: num_tasks
template <typename Closure>
size_t PrivatizedThreadpool<Closure>::num_tasks() const { 
  return _tasks.size(); 
}

// Function: num_workers
template <typename Closure>
size_t PrivatizedThreadpool<Closure>::num_workers() const { 
  return _workers.size();
}

// Function: shutdown
template <typename Closure>
void PrivatizedThreadpool<Closure>::_shutdown(){

  assert(is_owner());
  
  for(auto& w: _workers){ 
    std::scoped_lock worker_lock(w.mutex);//need a lock here, tested by unittest 
    w.ready = true;
    w.exit = true;
    w.cv.notify_one();
  }

  for(auto& t : _threads){
    t.join();
  }

  _threads.clear();
  _workers.clear();
}


// Function: _spawn 
template <typename Closure>
void PrivatizedThreadpool<Closure>::_spawn(unsigned N) {
 
  assert(is_owner());

  for(size_t i=0; i<N; i++){
    _threads.emplace_back([this, i=i]() -> void{
      
      PerThread* pt = _per_thread();
      pt->pool = this;
      pt->rand = std::hash<std::thread::id>()(std::this_thread::get_id());
      pt->thread_id = i;

      std::optional<Closure> t;
      Worker& w = _workers[i];

      while(!w.exit){
        //fail to get task from private queue or steal task from other queue
        if(!w.queue.pop_front(t) && !_steal(t)){ 
    
          // owner may insert tasks to private queue after the check 
          if(++_num_idlers == num_workers()){
            if(auto ret = _nonempty_worker_queue(); ret){
              --_num_idlers;
              _workers[*ret].cv.notify_one(); //pesimistic
              continue;
            }
          }
    
          {
            std::unique_lock worker_lock(w.mutex);
            
            if(w.queue.empty() && _tasks.empty()){
              w.ready = false;
            }        
            
            // TODO: problematic
            while(!w.exit && !w.ready){
              w.cv.wait(worker_lock);
            }

            w.ready = true;
          } 

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
    });
  }
}

// Procedure: emplace
template <typename Closure>
template <typename... ArgsT>
void PrivatizedThreadpool<Closure>::emplace(ArgsT&&... args){

  //no worker thread available
  if(num_workers() == 0){
    Closure{std::forward<ArgsT>(args)...}();
    return;
  }

  Closure t {std::forward<ArgsT>(args)...};

  PerThread* pt = _per_thread();
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
    target = (++_next_queue) % _workers.size(); 
    Worker& w = _workers[target];
    _next_queue = target;
    

    {
      std::scoped_lock worker_lock(w.mutex);

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
}   */


// ==========
// Chun-Xun's implementation
// ==========

// ---------------------------------------------------------------------------- 
// Privatized queue of worker. The lock-free queue is inspired by 
// http://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue
template<typename T, size_t C>
class PrivatizedWorkQueue {

  public:

  PrivatizedWorkQueue() : _buffer_mask(C - 1) {
    for (size_t i = 0; i < C; i ++){
      _buffer[i].sequence_.store(i, std::memory_order_relaxed);
    }
    _front.store(0, std::memory_order_relaxed);
    _back.store(0, std::memory_order_relaxed);
  }

  ~PrivatizedWorkQueue(){}

  bool enqueue(T& data){
    size_t pos = _front.load(std::memory_order_relaxed);
    cell_t* cell = &_buffer[pos & _buffer_mask];
    size_t seq = cell->sequence_.load(std::memory_order_acquire);
    intptr_t dif = (intptr_t)seq - (intptr_t)pos;
    if (dif == 0 && _front.compare_exchange_strong(pos, pos + 1, std::memory_order_relaxed)){
      _num_items.fetch_add(1, std::memory_order_relaxed);
      cell->data = std::move(data);
      cell->sequence_.store(pos + 1, std::memory_order_release);
      return true;
    }
    return false;
  }

  bool dequeue(std::optional<T>& data){
    if(empty()) return false;
    size_t pos = _back.load(std::memory_order_relaxed);
    cell_t* cell = &_buffer[pos & _buffer_mask];
    size_t seq = cell->sequence_.load(std::memory_order_acquire);
    intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);
    if (dif == 0 && _back.compare_exchange_strong(pos, pos + 1, std::memory_order_relaxed)){
      _num_items.fetch_sub(1, std::memory_order_relaxed);
      data = std::move(cell->data);
      cell->sequence_.store(pos + _buffer_mask + 1, std::memory_order_release);
      return true;
    }
    return false;
  }

  bool empty() const {
    return _front.load(std::memory_order_relaxed) == 
           _back.load(std::memory_order_relaxed);
  };

  size_t size() const {
    return _num_items.load(std::memory_order_relaxed);
  }

  private:
  
  struct cell_t {
    std::atomic<size_t>   sequence_;
    T                     data;
  };

  static size_t const     cacheline_size = 64;
  typedef char            cacheline_pad_t [cacheline_size];

  cacheline_pad_t         _pad0;
  std::array<cell_t, C>   _buffer;
  size_t const            _buffer_mask;
  cacheline_pad_t         _pad1;
  std::atomic<size_t>     _front;
  cacheline_pad_t         _pad2;
  std::atomic<size_t>     _back;
  cacheline_pad_t         _pad3;
  std::atomic<size_t>     _num_items {0};
}; 


// ----------------------------------------------------------------------------

// Class: PrivatizedThreadpool
template <typename Closure>
class PrivatizedThreadpool {

  struct Worker{
    std::condition_variable cv;
    PrivatizedWorkQueue<Closure, 1024> queue;
    std::optional<Closure> cache;
    bool exit {false};

    size_t last_victim {0};
  };
  
  struct PerThread {
    PrivatizedThreadpool* pool {nullptr};
    Worker* worker {nullptr};
  };

  public:

    PrivatizedThreadpool(unsigned);
    ~PrivatizedThreadpool();

    size_t num_tasks() const;
    size_t num_workers() const;

    bool is_owner() const;

    template <typename... ArgsT>
    void emplace(ArgsT&&...);

    void batch(std::vector<Closure>&&);

  private:
    
    const std::thread::id _owner {std::this_thread::get_id()};

    mutable std::mutex _mutex;

    uint64_t _seed {1};

    std::vector<Closure> _tasks;
    std::vector<std::thread> _threads;
    std::vector<Worker> _workers;
    
    std::unordered_map<std::thread::id, size_t> _worker_maps;

    size_t _next_queue {0};

    std::atomic<size_t> _num_tasks {0};
    
    std::optional<size_t> _nonempty_worker_queue() const;

    bool _steal(std::optional<Closure>&, uint64_t&, const size_t);

    void _shutdown();
    void _spawn(unsigned);

    PerThread& _per_thread() const;

    unsigned _randomize(uint64_t&);
    uint32_t _fast_modulo(const uint32_t, const uint32_t);
};

// Function: _nonempty_worker_queue
template <typename Closure>
std::optional<size_t> PrivatizedThreadpool<Closure>::_nonempty_worker_queue() const {
  for(size_t i=0;i <_workers.size(); ++i){
    if(!_workers[i].queue.empty()){
      return i;
    }
  }
  return {};
}

// Function: _randomize
template <typename Closure>
unsigned PrivatizedThreadpool<Closure>::_randomize(uint64_t& state) {
  uint64_t current = state;
  state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
  // Generate the random output (using the PCG-XSH-RS scheme)
  return static_cast<unsigned>((current ^ (current >> 22)) >> (22 + (current >> 61)));
}

// http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
template <typename Closure>
uint32_t PrivatizedThreadpool<Closure>::_fast_modulo(const uint32_t x, const uint32_t N) {
  return ((uint64_t) x * (uint64_t) N) >> 32;
}

// Function: _steal 
template <typename Closure>
bool PrivatizedThreadpool<Closure>::_steal(std::optional<Closure>& w, uint64_t& seed, const size_t id){
  const auto queue_num = _workers.size();
  auto victim = _workers[id].last_victim;
  for(size_t i=0; i<queue_num; i++){
    if(_workers[victim].queue.dequeue(w)){
      _workers[id].last_victim = victim;
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
template <typename Closure>
PrivatizedThreadpool<Closure>::PrivatizedThreadpool(unsigned N): _workers {N} {
  _spawn(N);
}

// Destructor
template <typename Closure>
PrivatizedThreadpool<Closure>::~PrivatizedThreadpool(){
  _shutdown();
}

// Function: _per_thread
template <typename Closure>
typename PrivatizedThreadpool<Closure>::PerThread&
PrivatizedThreadpool<Closure>::_per_thread() const {
  thread_local PerThread pt {this};
  return pt;
}

// Function: is_owner
template <typename Closure>
bool PrivatizedThreadpool<Closure>::is_owner() const {
  return std::this_thread::get_id() == _owner;
}

// Function: num_tasks
template <typename Closure>
size_t PrivatizedThreadpool<Closure>::num_tasks() const { 
  return _tasks.size(); 
}

// Function: num_workers
template <typename Closure>
size_t PrivatizedThreadpool<Closure>::num_workers() const { 
  return _threads.size();  
}

// Function: shutdown
template <typename Closure>
void PrivatizedThreadpool<Closure>::_shutdown(){

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
template <typename Closure>
void PrivatizedThreadpool<Closure>::_spawn(unsigned N) {

  assert(is_owner());

  // Lock to synchronize all workers before creating _worker_mapss
  std::scoped_lock lock(_mutex);

  for(size_t i=0; i<N; ++i){
    _threads.emplace_back([this, i=i] () -> void {

      std::optional<Closure> t;
      Worker& w = (_workers[i]);
      uint64_t seed = i+1;
      std::unique_lock lock(_mutex, std::defer_lock);

      while(!w.exit){
        if(_num_tasks == 0 || (!w.queue.dequeue(t) && !_steal(t, seed, i))) {
          lock.lock();
          if(!_tasks.empty()) {
            t = std::move(_tasks.back());
            _tasks.pop_back();
          }
          else{
            while(!w.exit && _num_tasks.load(std::memory_order_relaxed) == 0){
              w.cv.wait(lock);
            }
          } 
          lock.unlock();
        } // End of pop_front 

        while(t) {
          _num_tasks.fetch_sub(1, std::memory_order_relaxed);
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

template <typename Closure>
template <typename... ArgsT>
void PrivatizedThreadpool<Closure>::emplace(ArgsT&&... args){

  //no worker thread available
  if(num_workers() == 0){
    Closure{std::forward<ArgsT>(args)...}();
    return;
  }

  _num_tasks.fetch_add(1, std::memory_order_relaxed);

  Closure t {std::forward<ArgsT>(args)...};
  
  // caller is not the owner
  if(auto tid = std::this_thread::get_id(); tid != _owner){

    // the caller is the worker of the threadpool
    if(auto itr = _worker_maps.find(tid); itr != _worker_maps.end()){
      // Speculation
      if(!_workers[itr->second].cache.has_value()){
        _workers[itr->second].cache = std::move(t);
        return ;
      }
     
      auto other = _fast_modulo(_randomize(_seed), _workers.size());
      if(_workers[other].queue.size() > _workers[itr->second].queue.size()){
        if(!_workers[itr->second].queue.enqueue(t)){
          std::scoped_lock lock(_mutex);       
          _tasks.push_back(std::move(t));
        }
      }
      else{
        if(!_workers[other].queue.enqueue(t)){
          std::scoped_lock lock(_mutex);       
          _tasks.push_back(std::move(t));
        }
        _workers[other].cv.notify_one();
      }
      return ;
    }
  }
  
  auto id = _fast_modulo(_randomize(_seed), _workers.size());

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



template <typename Closure>
void PrivatizedThreadpool<Closure>::batch(std::vector<Closure>&& tasks){

  //no worker thread available
  if(num_workers() == 0){
    for(auto &t: tasks){
      t();
    }
    return;
  }

  _num_tasks.fetch_add(tasks.size(), std::memory_order_relaxed);

  bool notify_all = tasks.size() > 1;
  {
    std::scoped_lock lock(_mutex);
    std::move(tasks.begin(), tasks.end(), std::back_inserter(_tasks));
  }

  if(!notify_all) {
    auto id = _fast_modulo(_randomize(_seed), _workers.size());
    _workers[id].cv.notify_one();
  }
  else {
    for(size_t i=0; i<_workers.size(); ++i) {
      _workers[i].cv.notify_one();
    }
  }
}


};  // namespace tf -----------------------------------------------------------




