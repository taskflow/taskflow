// 2018/11/28 - modified by Chun-Xun Lin
// 
// Added the method batch to insert a vector of tasks.
//
// 2018/10/04 - modified by Tsung-Wei Huang
// 
// Removed shutdown, spawn, and wait_for_all to simplify the design
// of the threadpool. The threadpool now can operates on fixed memory
// closure to improve the performance.
//
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

namespace tf {

/**
@class: SimpleThreadpool

@brief Executor that implements a centralized task queue with a simple 
       execution strategy.

@tparam Closure closure type
*/
template <typename Closure>
class SimpleThreadpool {

  public:

    /**
    @brief constructs the executor with a given number of worker threads
    
    @param N the number of worker threads
    */
    explicit SimpleThreadpool(unsigned N);

    /**
    @brief destructs the executor

    Destructing the executor immediately forces all worker threads to stop.
    The executor does not guarantee all tasks to finish upon destruction.
    */
    ~SimpleThreadpool();
    
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

    /**
    @brief queries the number of worker threads
    */
    size_t num_workers() const;

    /**
    @brief queries if the caller is the owner of the executor
    */
    bool is_owner() const;
    
    size_t num_tasks() const;

  private:

    const std::thread::id _owner {std::this_thread::get_id()};

    mutable std::mutex _mutex;

    std::condition_variable _worker_signal;
    std::vector<Closure> _tasks;
    std::vector<std::thread> _threads;
    
    bool _stop {false};

    void _spawn(unsigned);
    void _shutdown();
};

// Constructor
template <typename Closure>
SimpleThreadpool<Closure>::SimpleThreadpool(unsigned N) {
  _spawn(N);
}

// Destructor
template <typename Closure>
SimpleThreadpool<Closure>::~SimpleThreadpool() {
  _shutdown();
}

// Function: num_tasks
// Return the number of "unfinished" tasks. 
// Notice that this value is not necessary equal to the size of the task_queue 
// since the task can be popped out from the task queue while 
// not yet finished.
template <typename Closure>
size_t SimpleThreadpool<Closure>::num_tasks() const {
  return _tasks.size();
}

template <typename Closure>
size_t SimpleThreadpool<Closure>::num_workers() const {
  return _threads.size();
}

// Function: is_owner
template <typename Closure>
bool SimpleThreadpool<Closure>::is_owner() const {
  return std::this_thread::get_id() == _owner;
}

// Procedure: spawn
// The procedure spawns "n" threads monitoring the task queue and executing each task. 
// After the task is finished, the thread reacts to the returned signal.
template <typename Closure>
void SimpleThreadpool<Closure>::_spawn(unsigned N) {

  assert(is_owner());
    
  for(size_t i=0; i<N; ++i) {
      
    _threads.emplace_back([this] () -> void { 
        
      Closure task;
          
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
template <typename Closure>
template <typename... ArgsT>
void SimpleThreadpool<Closure>::emplace(ArgsT&&... args) {
  
  // No worker, do this right away.
  if(num_workers() == 0) {
    Closure{std::forward<ArgsT>(args)...}();
  }
  // Dispatch this to a thread.
  else {
    std::scoped_lock lock(_mutex);
    _tasks.emplace_back(std::forward<ArgsT>(args)...);
    _worker_signal.notify_one();
  }
}


// Function: emplace
template <typename Closure>
void SimpleThreadpool<Closure>::batch(std::vector<Closure>&& tasks) {

  // No worker, do this right away.
  if(num_workers() == 0) {
    for(auto& t: tasks){
      t();
    }
    return ;
  }
  // Dispatch this to a thread.
  else {
    bool notify_all = tasks.size() > 1;
    {
      std::scoped_lock lock(_mutex);
      _tasks.reserve(_tasks.size() + tasks.size());
      std::move(tasks.begin(), tasks.end(), std::back_inserter(_tasks));
    }
    if(notify_all) {
      _worker_signal.notify_all();
    }
    else {
      _worker_signal.notify_one();
    }
  }
}


// Procedure: shutdown
// Shut down the threadpool - only the owner can do this.
template <typename Closure>
void SimpleThreadpool<Closure>::_shutdown() {
  
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

