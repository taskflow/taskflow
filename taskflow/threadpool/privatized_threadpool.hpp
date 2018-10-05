// 2018/09/21 - modified by Tsung-Wei and Chun-Xun
//   - refactored the code
//  
// TODO:
//   - Problems can occur when external threads insert tasks during spawn.
//
// 2018/09/12 - created by Tsung-Wei Huang and Chun-Xun Lin
//
// Implemented PrivatizedThreadpool using the data structre inspired
// Eigen CXX/Threadpool.

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

// ---------------------------------------------------------------------------- 
// Privatized queue of worker. The lock-free queue is inspired by 
//   http://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue
template<typename T, size_t buffer_size>
class PrivatizedTaskQueue {
public:
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

  PrivatizedTaskQueue(PrivatizedTaskQueue const&);
  void operator = (PrivatizedTaskQueue const&);
}; 



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
    std::condition_variable _empty_cv;

    std::vector<Task> _tasks;
    std::vector<std::thread> _threads;
    
    std::unordered_map<std::thread::id, size_t> _worker_maps;
    std::vector<Worker> _workers;

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

  if(!is_owner()){
    throw std::runtime_error("Worker thread cannot shut down the pool");
  }

  if(_threads.empty()) {
    return;
  }

  {
    std::scoped_lock<std::mutex> lock(_mutex);
    // Notify workers to exit
    for(auto& w : _workers){
      w.exit = true;
      w.cv.notify_one();
    }
  } // Release lock

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

  if(! is_owner()){
    throw std::runtime_error("Worker thread cannot spawn threads");
  }

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

        while(t){
          (*t)();
          std::swap(t, w.cache);
          w.cache = std::nullopt;
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

  if(auto tid = std::this_thread::get_id(); tid != _owner){
    if(auto itr = _worker_maps.find(tid); itr != _worker_maps.end()){
      if(!_workers[itr->second].cache.has_value()){
        _workers[itr->second].cache = std::move(t);
        return ;
      }
      if(!_workers[itr->second].queue.enqueue(t)){
        std::scoped_lock<std::mutex> lock(_mutex);       
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








/*
// Class: BasicPrivatizedThreadpool
template < template<typename...> class Func >
class BasicPrivatizedThreadpool {

  using TaskType = Func<void()>;
  using WorkQueue = RunQueue<TaskType, 1024>;

  struct Worker{
    std::condition_variable cv;
    WorkQueue queue;
  };

  public:

    BasicPrivatizedThreadpool(unsigned);
    ~BasicPrivatizedThreadpool();

    size_t num_tasks() const;
    size_t num_workers() const;

    bool is_owner() const;

    void shutdown();
    void spawn(unsigned);
    void wait_for_all();

    template <typename C>
    void silent_async(C&&);

    template <typename C>
    auto async(C&&);

  private:

    const std::thread::id _owner {std::this_thread::get_id()};

    mutable std::mutex _mutex;

    std::condition_variable _empty_cv;

    std::deque<TaskType> _task_queue;

    std::vector<std::thread> _threads;
    std::vector<std::unique_ptr<Worker>> _workers;
    std::vector<size_t> _coprimes;

    size_t _num_idlers {0}; 
    size_t _next_queue {0};

    std::unordered_map<std::thread::id, size_t> _worker_maps;    
    
    bool _exiting      {false};
    bool _wait_for_all {false};

    size_t _nonempty_worker_queue() const;

    void _xorshift32(uint32_t&);
    bool _steal(TaskType&, uint32_t&);

};  // class BasicPrivatizedThreadpool. --------------------------------------


// Function: _nonempty_worker_queue
template < template<typename...> class Func >
size_t BasicPrivatizedThreadpool<Func>::_nonempty_worker_queue() const {
  for(size_t i=0;i <_workers.size(); ++i){
    if(!_workers[i]->queue.empty()){
      return i;
    }
  }
  return _workers.size();
}

// Function: _xorshift32
template < template<typename...> class Func >
void BasicPrivatizedThreadpool<Func>::_xorshift32(uint32_t& x){
  // x must be non zero: https://en.wikipedia.org/wiki/Xorshift
  // Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" 
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
}

// Function: _steal
template < template<typename...> class Func >
bool BasicPrivatizedThreadpool<Func>::_steal(TaskType& w, uint32_t& seed){
  _xorshift32(seed);
  const auto inc = _coprimes[seed % _coprimes.size()];
  const auto queue_num = num_workers();
  auto victim = seed % queue_num;
  for(size_t i=0; i<queue_num; i++){
    if(_workers[victim]->queue.pop_back(w)){
      return true;
    }
    victim += inc;
    if(victim >= queue_num){
      victim -= queue_num;
    }
  }
  return false;
}



// Constructor
template < template<typename...> class Func >
BasicPrivatizedThreadpool<Func>::BasicPrivatizedThreadpool(unsigned N){
  spawn(N);
}

// Destructor
template < template<typename...> class Func >
BasicPrivatizedThreadpool<Func>::~BasicPrivatizedThreadpool(){
  shutdown();
}

// Function: is_owner
template < template<typename...> class Func >
bool BasicPrivatizedThreadpool<Func>::is_owner() const {
  return std::this_thread::get_id() == _owner;
}

// Function: num_tasks
template < template<typename...> class Func >
size_t BasicPrivatizedThreadpool<Func>::num_tasks() const { 
  return _task_queue.size(); 
}

// Function: num_workers
template < template<typename...> class Func >
size_t BasicPrivatizedThreadpool<Func>::num_workers() const { 
  return _threads.size();  
}

// Function: shutdown
template < template<typename...> class Func >
void BasicPrivatizedThreadpool<Func>::shutdown(){

  if(!is_owner()){
    throw std::runtime_error("Worker thread cannot shut down the pool");
  }

  if(_threads.empty()) {
    return;
  }

  { 
    std::unique_lock<std::mutex> lock(_mutex);
    _wait_for_all = true;

    // Wake up all workers in case they are already idle
    for(const auto& w : _workers){
      w->cv.notify_one();
    }

    //while(_num_idlers != num_workers()) {
    while(_wait_for_all){
      _empty_cv.wait(lock);
    }
    _exiting = true;

    for(auto& w : _workers){
      // TODO: can we replace this dummy task with state?
      w->queue.push_back([](){});
      w->cv.notify_one();
    }
  }

  for(auto& t : _threads){
    t.join();
  } 
  _threads.clear();  

  _workers.clear();
  _worker_maps.clear();
  
  _wait_for_all = false;
  _exiting = false;
  
  // task queue might have tasks added by threads outside this pool...
  //while(not _task_queue.empty()) {
  //  std::invoke(_task_queue.front());
  //  _task_queue.pop_front();
  //}
}

// Function: spawn 
template < template<typename...> class Func >
void BasicPrivatizedThreadpool<Func>::spawn(unsigned N) {

  if(!is_owner()){
    throw std::runtime_error("Worker thread cannot spawn threads");
  }

  // Wait untill all workers become idle if any
  if(!_threads.empty()){
    wait_for_all();
  }

  const size_t sz = _threads.size();

  // Lock to synchronize all workers before creating _worker_mapss
  std::scoped_lock<std::mutex> lock(_mutex);

  _coprimes.clear();
  for(size_t i=1; i<=sz+N; i++){
    if(std::gcd(i, sz+N) == 1){
      _coprimes.push_back(i);
    }
  }

  for(size_t i=0; i<N; ++i){
    _workers.push_back(std::make_unique<Worker>());
  }

  for(size_t i=0; i<N; ++i){
    _threads.emplace_back([this, i=i+sz]() -> void {

      TaskType t {nullptr};
      Worker& w = *(_workers[i]);
      uint32_t seed = i+1;
      std::unique_lock<std::mutex> lock(_mutex);

      while(!_exiting){
        
        lock.unlock();
        if(t) {
          t();
          t = nullptr;
        }
        while(w.queue.pop_front(t)) {
          t();
          t = nullptr;
        };
        lock.lock();
        

        //if(!w.queue.pop_front(t)){
          if(_steal(t, seed)) {
          }
          else if(!_task_queue.empty()) {
            t = std::move(_task_queue.front());
            _task_queue.pop_front();
          } 
          else {
            // Last idler
            if(++_num_idlers == num_workers() && _wait_for_all){
              if(auto ret = _nonempty_worker_queue(); ret == num_workers()){
                _wait_for_all = false;
                _empty_cv.notify_one();
              }
              else{
                if(ret == i){
                  --_num_idlers;
                  continue;
                }
                _workers[ret]->cv.notify_one();
              }
            } 
            w.cv.wait(lock);
            --_num_idlers;
          }
        //} // End of first if

        //if(t){
        //  _mutex.unlock();
        //  // speculation
        //  do {
        //    t();
        //    t = nullptr;
        //  } while(w.queue.pop_front(t));
        //  _mutex.lock();
        //}
      } // End of while ------------------------------------------------------
    });     

    _worker_maps.insert({_threads.back().get_id(), i+sz});
  } // End of For ---------------------------------------------------------------------------------

}


// Function: async
template < template<typename...> class Func >
template <typename C>
auto BasicPrivatizedThreadpool<Func>::async(C&& c){

  using R = std::invoke_result_t<C>;

  std::promise<R> p;
  auto fu = p.get_future();
  
  // master thread
  if(num_workers() == 0){
    if constexpr(std::is_same_v<void, R>){
      c();
      p.set_value();
    }
    else{
      p.set_value(c());
    } 
  }
  // have worker(s)
  else{
    if constexpr(std::is_same_v<void, R>){
      silent_async( 
        [p = MoC(std::move(p)), c = std::forward<C>(c)]() mutable {
          c();
          p.get().set_value(); 
        }
      );
    }
    else{
      silent_async( 
        [p = MoC(std::move(p)), c = std::forward<C>(c)]() mutable {
          p.get().set_value(c()); 
        }
      );
    }
  }
  return fu;
}

template < template<typename...> class Func >
template <typename C>
void BasicPrivatizedThreadpool<Func>::silent_async(C&& c){

  TaskType t {std::forward<C>(c)};

  //no worker thread available
  if(num_workers() == 0){
    t();
    return;
  }

  if(std::this_thread::get_id() != _owner){
    auto tid = std::this_thread::get_id();
    if(_worker_maps.find(tid) != _worker_maps.end()){
      if(!_workers[_worker_maps.at(tid)]->queue.push_front(t)){
        std::scoped_lock<std::mutex> lock(_mutex);       
        _task_queue.push_back(std::move(t));
      }
      return ;
    }
  }

  // owner thread or other threads
  auto id = (++_next_queue)%_workers.size();
  if(!_workers[id]->queue.push_back(t)){
    std::scoped_lock<std::mutex> lock(_mutex);
    _task_queue.push_back(std::move(t));
  }

  // Make sure at least one worker will handle the task
  _workers[id]->cv.notify_one();
}


// Function: wait_for_all
template < template<typename...> class Func >
void BasicPrivatizedThreadpool<Func>::wait_for_all() {

  if(!is_owner()){
    throw std::runtime_error("Worker thread cannot wait for all");
  }

  if(num_workers() == 0) return ;

  std::unique_lock<std::mutex> lock(_mutex);

  _wait_for_all = true;

  // Wake up all workers in case they are already idle
  for(const auto& w : _workers){
    w->cv.notify_one();
  }
  
  while(_wait_for_all) {
    _empty_cv.wait(lock);
  }
} 
*/

};  // namespace tf -----------------------------------------------------------




