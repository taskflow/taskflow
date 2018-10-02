// 2018/09/14 - modified by Guannan
//   - added wait_for_all method
//
// 2018/04/01 - contributed by Tsung-Wei Huang and Chun-Xun Lin
// 
// The basic threadpool implementation based on C++17.

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

#include "move_on_copy.hpp"

namespace tf {

// Class: SimpleThreadpool
class SimpleThreadpool {

  enum class Signal {
    STANDARD,
    SHUTDOWN
  };

  public:

    inline SimpleThreadpool(unsigned);
    inline ~SimpleThreadpool();
    
    template <typename C>
    auto async(C&&, Signal = Signal::STANDARD);

    template <typename C>
    auto silent_async(C&&, Signal = Signal::STANDARD);
    
    inline void shutdown();
    inline void spawn(unsigned);
    inline void wait_for_all();
    
    inline size_t num_tasks() const;
    inline size_t num_workers() const;

    inline bool is_owner() const;

  private:

    const std::thread::id _owner {std::this_thread::get_id()};

    mutable std::mutex _mutex;

    std::condition_variable _worker_signal;
    std::deque<std::function<Signal()>> _task_queue;
    std::vector<std::thread> _threads;

    //variables for wait_for_all
    std::condition_variable _complete;
    size_t _num_idlers  {0};
    bool _wait_for_all    {false};
};

// Constructor
inline SimpleThreadpool::SimpleThreadpool(unsigned N) {
  spawn(N);
}

// Destructor
inline SimpleThreadpool::~SimpleThreadpool() {
  shutdown();
}

// Function: num_tasks
// Return the number of "unfinished" tasks. 
// Notice that this value is not necessary equal to the size of the task_queue 
// since the task can be popped out from the task queue while 
// not yet finished.
inline size_t SimpleThreadpool::num_tasks() const {
  return _task_queue.size();
}

inline size_t SimpleThreadpool::num_workers() const {
  return _threads.size();
}

// Function: is_owner
inline bool SimpleThreadpool::is_owner() const {
  return std::this_thread::get_id() == _owner;
}

// Procedure: spawn
// The procedure spawns "n" threads monitoring the task queue and executing each task. 
// After the task is finished, the thread reacts to the returned signal.
inline void SimpleThreadpool::spawn(unsigned N) {

  if(!is_owner()) {
    throw std::runtime_error("Worker thread cannot spawn threads");
  }
  
  for(size_t i=0; i<N; ++i) {

    _threads.emplace_back([this] () -> void { 

      bool stop {false}; 

      while(!stop) {
        decltype(_task_queue)::value_type task;

        { // Acquire lock. --------------------------------
          std::unique_lock<std::mutex> lock(_mutex);

          _num_idlers++;
          if(_num_idlers == num_workers() && _task_queue.size() == 0 && _wait_for_all){
            _complete.notify_one();
          }
          _worker_signal.wait(lock, [this] () { return _task_queue.size() != 0; });
          _num_idlers--;

          task = std::move(_task_queue.front());
          _task_queue.pop_front();
        } // Release lock. --------------------------------

        // Execute the task and react to the returned signal.
        switch(task()) {
          case Signal::SHUTDOWN:
            stop = true;
          break;      

          default:
          break;
        };

      } // End of worker loop.

    });
  }
}

// Function: silent_async
// Insert a task without giving future.
template <typename C>
auto SimpleThreadpool::silent_async(C&& c, Signal sig) {
  
  // No worker, do this right away.
  if(num_workers() == 0) {
    c();
  }
  // Dispatch this to a thread.
  else {
    {
      std::unique_lock lock(_mutex);
      _task_queue.emplace_back(
        [c=std::forward<C>(c), ret=sig] () mutable { 
          c();
          return ret;
        }
      );
    }
    _worker_signal.notify_one();
  }
}

// Function: async
// Insert a callable task and return a future representing the task.
template<typename C>
auto SimpleThreadpool::async(C&& c, Signal sig) {

  using R = std::invoke_result_t<C>;
  
  std::promise<R> p;
  auto fu = p.get_future();
  
  // No worker, do this immediately.
  if(_threads.empty()) {
    if constexpr(std::is_same_v<void, R>) {
      c();
      p.set_value();
    }
    else {
      p.set_value(c());
    }
  }
  // Schedule a thread to do this.
  else {
    {
      std::unique_lock lock(_mutex);
      
      if constexpr(std::is_same_v<void, R>) {
        _task_queue.emplace_back(
          [p = MoC(std::move(p)), c = std::forward<C>(c), ret = sig]() mutable {
            c();
            p.get().set_value();
            return ret;
          }
        );
      }
      else {
        _task_queue.emplace_back(
          [p = MoC(std::move(p)), c = std::forward<C>(c), ret = sig]() mutable {
            p.get().set_value(c());
            return ret;
          }
        );
      }
    }
    _worker_signal.notify_one();
  }
  return fu;
}

// Procedure: wait_for_all
inline void SimpleThreadpool::wait_for_all(){

  if(!is_owner()) {
    throw std::runtime_error("Worker thread cannot wait for all");
  }
  
  std::unique_lock<std::mutex> lock(_mutex);
  _wait_for_all = true;
  _complete.wait(lock, [this](){ return _num_idlers == num_workers() && _task_queue.size() == 0; });
  _wait_for_all = false;
}

// Procedure: shutdown
// Remove a given number of workers. Notice that only the master can call this procedure.
inline void SimpleThreadpool::shutdown() {
  
  if(!is_owner()) {
    throw std::runtime_error("Worker thread cannot shut down the thread pool");
  }

  wait_for_all();

  for(size_t i=0; i<_threads.size(); ++i) {
    silent_async([](){}, Signal::SHUTDOWN);
  }
  
  for(auto& t : _threads) {
    t.join();
  }

  _threads.clear();
} 

// ----------------------------------------------------------------------------

// Class: SimpleThreadpool2
template <typename Task>
class SimpleThreadpool2 {

  public:

    explicit SimpleThreadpool2(unsigned);

    ~SimpleThreadpool2();
    
    template <typename... ArgsT>
    void emplace(ArgsT&&...);

    void push(Task&&);

    size_t num_tasks() const;
    size_t num_workers() const;

    bool is_owner() const;

  private:

    const std::thread::id _owner {std::this_thread::get_id()};

    mutable std::mutex _mutex;

    std::condition_variable _worker_signal;
    std::vector<Task> _tasks;
    std::vector<std::thread> _threads;
    
    bool _stop {false};

    void _spawn(unsigned);
    void _shutdown();
};

// Constructor
template <typename Task>
SimpleThreadpool2<Task>::SimpleThreadpool2(unsigned N) {
  _spawn(N);
}

// Destructor
template <typename Task>
SimpleThreadpool2<Task>::~SimpleThreadpool2() {
  _shutdown();
}

// Function: num_tasks
// Return the number of "unfinished" tasks. 
// Notice that this value is not necessary equal to the size of the task_queue 
// since the task can be popped out from the task queue while 
// not yet finished.
template <typename Task>
size_t SimpleThreadpool2<Task>::num_tasks() const {
  return _tasks.size();
}

template <typename Task>
size_t SimpleThreadpool2<Task>::num_workers() const {
  return _threads.size();
}

// Function: is_owner
template <typename Task>
bool SimpleThreadpool2<Task>::is_owner() const {
  return std::this_thread::get_id() == _owner;
}

// Procedure: spawn
// The procedure spawns "n" threads monitoring the task queue and executing each task. 
// After the task is finished, the thread reacts to the returned signal.
template <typename Task>
void SimpleThreadpool2<Task>::_spawn(unsigned N) {

  assert(is_owner());
    
  for(size_t i=0; i<N; ++i) {
      
    _threads.emplace_back([this] () -> void { 
        
      Task task;
          
      std::unique_lock lock(_mutex);

      while(!_stop) {
        
        if(!_tasks.empty()) {
          task = std::move(_tasks.back());
          _tasks.pop_back();

          // execute the task
          lock.unlock();
          task();
          lock.lock();
        }
        else {
          while(_tasks.empty() && !_stop) {
            _worker_signal.wait(lock);
          }
        }

      } // End of worker loop.

    });
  }
}

// Function: emplace
template <typename Task>
template <typename... ArgsT>
void SimpleThreadpool2<Task>::emplace(ArgsT&&... args) {
  
  // No worker, do this right away.
  if(num_workers() == 0) {
    Task{std::forward<ArgsT>(args)...}();
  }
  // Dispatch this to a thread.
  else {
    std::scoped_lock lock(_mutex);
    _tasks.emplace_back(std::forward<ArgsT>(args)...);
    _worker_signal.notify_one();
  }
}

// Function: push_back
// Insert a task without giving future.
template <typename Task>
void SimpleThreadpool2<Task>::push(Task&& task) {
  
  // No worker, do this right away.
  if(num_workers() == 0) {
    task();
  }
  // Dispatch this to a thread.
  else {
    std::scoped_lock lock(_mutex);
    _tasks.push_back(std::move(task));
    _worker_signal.notify_one();
  }
}

// Procedure: shutdown
// Shut down the threadpool - only the owner can do this.
template <typename Task>
void SimpleThreadpool2<Task>::_shutdown() {
  
  assert(is_owner());

  {
    std::scoped_lock lock(_mutex);
    _stop = true;
    _worker_signal.notify_all();
  }

  for(auto& t : _threads) {
    t.join();
  }

  _threads.clear();
  _stop = false;
} 

};  // end of namespace tf. ---------------------------------------------------

