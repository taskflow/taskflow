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

// Class: SimpleThreadpool
class SimpleThreadpool {

  // Struct: MoC
  template <typename T>
  struct MoC {
  
    MoC(T&& rhs) : object(std::move(rhs)) {}
    MoC(const MoC& other) : object(std::move(other.object)) {}
  
    T& get() { return object; }
    
    mutable T object; 
  };

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

    inline bool is_worker() const;

  private:

    mutable std::mutex _mutex;

    std::condition_variable _worker_signal;
    std::deque<std::function<Signal()>> _task_queue;
    std::vector<std::thread> _threads;
    std::unordered_set<std::thread::id> _worker_ids;

    //variables for wait_for_all
    std::condition_variable _complete;
    size_t _idle_workers  {0};
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

inline bool SimpleThreadpool::is_worker() const {
  std::scoped_lock<std::mutex> lock(_mutex);
  return _worker_ids.find(std::this_thread::get_id()) != _worker_ids.end();
}

// Procedure: spawn
// The procedure spawns "n" threads monitoring the task queue and executing each task. 
// After the task is finished, the thread reacts to the returned signal.
inline void SimpleThreadpool::spawn(unsigned N) {

  if(is_worker()) {
    throw std::runtime_error("Worker thread cannot spawn threads");
  }
  
  for(size_t i=0; i<N; ++i) {

    _threads.emplace_back([this] () -> void { 

      {  // Acquire lock
        std::scoped_lock<std::mutex> lock(_mutex);
        _worker_ids.insert(std::this_thread::get_id());         
      }

      bool stop {false}; 

      while(!stop) {
        decltype(_task_queue)::value_type task;

        { // Acquire lock. --------------------------------
          std::unique_lock<std::mutex> lock(_mutex);

          _idle_workers++;
          if(_idle_workers == num_workers() && _task_queue.size() == 0 && _wait_for_all){
            _complete.notify_one();
          }
          _worker_signal.wait(lock, [this] () { return _task_queue.size() != 0; });
          _idle_workers--;

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

      {  // Acquire lock
        std::scoped_lock<std::mutex> lock(_mutex);
        _worker_ids.erase(std::this_thread::get_id());         
      }

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
// After this method returns, all previously-scheduled tasks in the pool
// will have been executed.
//inline void SimpleThreadpool::wait_for_all() {
//  
//  if(is_worker()) {
//    throw std::runtime_error("Worker thread cannot wait for all");
//  }
//
//  std::mutex barrier_mutex;
//  std::condition_variable barrier_cv;
//  auto threads_to_sync{_threads.size()};
//  std::vector<std::future<void>> futures;
//
//  for(size_t i=0; i<_threads.size(); ++i) {
//    futures.emplace_back(async([&] () {
//      std::unique_lock<std::mutex> lock(barrier_mutex);
//      if (--threads_to_sync; threads_to_sync == 0) {
//        barrier_cv.notify_all();
//      }
//      else {
//        barrier_cv.wait(lock, [&threads_to_sync] {
//          return threads_to_sync == 0;
//        });
//      }
//    }));
//  }
//
//  // Wait for all threads to have finished synchronization
//  for (auto& fu : futures) {
//    fu.get();
//  }
//}

inline void SimpleThreadpool::wait_for_all(){

  if(is_worker()) {
    throw std::runtime_error("Worker thread cannot wait for all");
  }
  
  std::unique_lock<std::mutex> lock(_mutex);
  _wait_for_all = true;
  _complete.wait(lock, [this](){ return _idle_workers == num_workers() && _task_queue.size() == 0; });
  _wait_for_all = false;

}

// Procedure: shutdown
// Remove a given number of workers. Notice that only the master can call this procedure.
inline void SimpleThreadpool::shutdown() {
  
  if(is_worker()) {
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

};  // end of namespace tf. ---------------------------------------------------
