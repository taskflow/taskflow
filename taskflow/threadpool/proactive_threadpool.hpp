// 2018/09/11 - modified by Tsung-Wei Huang & Guannan
//   - bug fix: shutdown method might hang due to dynamic tasking;
//     it can be non-empty task queue while all threads are gone;
//     workers need to be cleared as well under lock, since *_async
//     will access the worker data structure;
//     
// 2018/09/03 - modified by Guannan Guo
// 
// BasicProactiveThreadpool schedules independent jobs in a greedy manner.
// Whenever a job is inserted into the threadpool, the threadpool will check if there
// are any spare threads available. The spare thread will be woken through its local 
// condition variable. The new job will be directly moved into
// this thread instead of pushed at the back of the pending queue.

#pragma once

#include <iostream>
#include <functional>
#include <vector>
#include <mutex>
#include <deque>
#include <thread>
#include <stdexcept>
#include <condition_variable>
#include <memory>
#include <future>
#include <unordered_set>

namespace proactive_threadpool {
  
// Struct: MoC
// Move-on-copy wrapper.
template <typename T>
struct MoC {

  MoC(T&& rhs): object(std::move(rhs)) {}
  MoC(const MoC& other) : object(std::move(other.object)) {}

  T& get() {return object; }
  
  mutable T object;
};

// Class: BasicProactiveThreadpool
template < template<typename...> class Func >
class BasicProactiveThreadpool {

  using TaskType = Func<void()>;

  // Struct: Worker
  struct Worker{
    std::condition_variable cv;
    TaskType task;
    bool ready;
  };

  public:

    BasicProactiveThreadpool(unsigned);
    ~BasicProactiveThreadpool();

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

    std::thread::id _owner {std::this_thread::get_id()};

    mutable std::mutex _mutex;

    std::condition_variable _empty_cv;

    std::deque<TaskType> _task_queue;
    std::vector<std::thread> _threads;
    std::vector<Worker*> _workers; 

    bool _exiting {false};
    bool _wait_for_all {false};
};
    
// Constructor
template < template<typename...> class TaskType >
BasicProactiveThreadpool<TaskType>::BasicProactiveThreadpool(unsigned N){
  spawn(N);
}

// Destructor
template < template<typename...> class TaskType >
BasicProactiveThreadpool<TaskType>::~BasicProactiveThreadpool(){
  shutdown();
}

// Function: is_owner
template < template<typename...> class TaskType >
bool BasicProactiveThreadpool<TaskType>::is_owner() const {
  return std::this_thread::get_id() == _owner;
}

// Function: num_tasks    
template < template<typename...> class TaskType >
size_t BasicProactiveThreadpool<TaskType>::num_tasks() const { 
  return _task_queue.size(); 
}

// Function: num_workers
template < template<typename...> class TaskType >
size_t BasicProactiveThreadpool<TaskType>::num_workers() const { 
  return _threads.size();  
}

// Procedure: shutdown
template < template<typename...> class TaskType >
void BasicProactiveThreadpool<TaskType>::shutdown() {
  
  if(!is_owner()){
    throw std::runtime_error("Worker thread cannot shut down the pool");
  }

  if(_threads.empty()) {
    return;
  }

  { 
    std::unique_lock<std::mutex> lock(_mutex);

    _wait_for_all = true;

    while(_workers.size() != num_workers()) {
      _empty_cv.wait(lock);
    }

    _exiting = true;
    
    // we need to clear the workers under lock
    for(auto w : _workers){
      w->ready = true;
      w->task = nullptr;
      w->cv.notify_one();
    }
    _workers.clear();
  }
  
  for(auto& t : _threads){
    t.join();
  } 
  _threads.clear();  

  _wait_for_all = false;
  _exiting = false;

  // task queue might have tasks added by threads outside this pool...
  //assert(_task_queue.empty());
  //while(!_task_queue.empty()) {
  //  std::invoke(_task_queue.front());
  //  _task_queue.pop_front();
  //}
}

// Procedure: spawn
template < template<typename...> class TaskType >
void BasicProactiveThreadpool<TaskType>::spawn(unsigned N) {

  if(!is_owner()){
    throw std::runtime_error("Worker thread cannot spawn threads");
  }

  for(size_t i=0; i<N; ++i){
  
    _threads.emplace_back([this] () -> void {
      
      Worker w;
      TaskType t; 

      std::unique_lock<std::mutex> lock(_mutex);

      while(!_exiting){

        if(_task_queue.empty()){
          w.ready = false;
          _workers.push_back(&w);

          if(_wait_for_all && _workers.size() == num_workers()) {
            _empty_cv.notify_one();
          }

          while(!w.ready) {
            w.cv.wait(lock);
          }

          t = std::move(w.task);
        }
        else{
          t = std::move(_task_queue.front());
          _task_queue.pop_front();
        } 

        if(t){
          _mutex.unlock();
          t(); // run task in parallel
          t = nullptr;
          _mutex.lock();
        }
      }

    });     

  } 
}

// Procedure: silent_async
template < template<typename...> class TaskType >
template <typename C>
void BasicProactiveThreadpool<TaskType>::silent_async(C&& c){

  TaskType t {std::forward<C>(c)};

  //no worker thread available
  if(num_workers() == 0){
    t();
    return;
  }

  std::scoped_lock<std::mutex> lock(_mutex);
  if(_workers.empty()){
    _task_queue.push_back(std::move(t));
  } 
  else{
    Worker* w = _workers.back();
    _workers.pop_back();
    w->ready = true;
    w->task = std::move(t);
    w->cv.notify_one();   
  }
}

// Function: async
template < template<typename...> class TaskType >
template <typename C>
auto BasicProactiveThreadpool<TaskType>::async(C&& c) {

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
    std::scoped_lock<std::mutex> lock(_mutex);     
    if constexpr(std::is_same_v<void, R>){
      // all workers are busy.
      if(_workers.empty()){
        _task_queue.emplace_back(
          [p = MoC(std::move(p)), c = std::forward<C>(c)]() mutable {
            c();
            p.get().set_value(); 
          }
        );
      }
      // Got an idle work
      else{
        Worker* w = _workers.back();
        _workers.pop_back();
        w->ready = true;
        w->task = [p = MoC(std::move(p)), c = std::forward<C>(c)]() mutable {
          c();
          p.get().set_value(); 
        };
        w->cv.notify_one(); 
      }
    }
    else{

      if(_workers.empty()){
        _task_queue.emplace_back(
          [p = MoC(std::move(p)), c = std::forward<C>(c)]() mutable {
            p.get().set_value(c());
            return; 
          }
        );
      }
      else{
        Worker* w = _workers.back();
        _workers.pop_back();
        w->ready = true;
        w->task = [p = MoC(std::move(p)), c = std::forward<C>(c)]() mutable {
          p.get().set_value(c()); 
          return;
        };
        w->cv.notify_one(); 
      }
    }
  }

  return fu;
}

// Function: wait_for_all
template < template<typename...> class TaskType >
void BasicProactiveThreadpool<TaskType>::wait_for_all() {

  if(!is_owner()){
    throw std::runtime_error("Worker thread cannot wait for all");
  }

  std::unique_lock<std::mutex> lock(_mutex);
  _wait_for_all = true;
  while(_workers.size() != num_workers()) {
    _empty_cv.wait(lock);
  }
  _wait_for_all = false;
}

};  // namespace proactive_threadpool -----------------------------------------

namespace tf {

using ProactiveThreadpool = proactive_threadpool::BasicProactiveThreadpool<std::function>;

};  // namespace tf. ----------------------------------------------------------



