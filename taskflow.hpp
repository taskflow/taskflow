// MIT License
// 
// Copyright (c) 2018 Dr. Tsung-Wei Huang, and Chun-Xun Lin
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TASKFLOW_HPP_
#define TASKFLOW_HPP_

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <deque>
#include <algorithm>
#include <thread>
#include <future>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <sstream>

namespace tf {

//template <typename T>
//struct UniqueKeyGenerator {
//
//	T operator ()() const {
//    
//    if(std::is_integral_v<T>) {
//      static std::atomic<T> key {0};
//      return key++;
//    }
//
//  }
//
//};


// Struct: MoveOnCopy
template <typename T>
struct MoveOnCopy {

  MoveOnCopy(T&& rhs) : object(std::move(rhs)) {}
  MoveOnCopy(const MoveOnCopy& other) : object(std::move(other.object)) {}

  T& get() { return object; }
  
  mutable T object; 
};

// ------------------------------------------------------------------------------------------------

// Class: Threadpool
class Threadpool {

  enum class Signal {
    STANDARD,
    SHUTDOWN
  };

  public:

    inline Threadpool(auto);
    inline ~Threadpool();
    
    template <typename C>
    auto async(C&&, Signal = Signal::STANDARD);

    template <typename C>
    auto silent_async(C&&, Signal = Signal::STANDARD);
    
    inline void shutdown();
    inline void spawn(unsigned);
    
    inline size_t num_tasks() const;
    inline size_t num_workers() const;

    inline bool is_worker() const;

  private:

    mutable std::mutex _mutex;

    std::condition_variable _worker_signal;
    std::deque<std::function<Signal()>> _task_queue;
    std::vector<std::thread> _threads;
    std::unordered_set<std::thread::id> _worker_ids;
};

// Constructor
inline Threadpool::Threadpool(auto N) {
  spawn(N);
}

// Destructor
inline Threadpool::~Threadpool() {
  shutdown();
}

// Function: num_tasks
// Return the number of "unfinished" tasks. Notice that this value is not necessary equal to
// the size of the task_queue since the task can be popped out from the task queue while 
// not yet finished.
inline size_t Threadpool::num_tasks() const {
  return _task_queue.size();
}

inline size_t Threadpool::num_workers() const {
  return _threads.size();
}

inline bool Threadpool::is_worker() const {
  std::scoped_lock<std::mutex> lock(_mutex);
  return _worker_ids.find(std::this_thread::get_id()) != _worker_ids.end();
}

// Procedure: spawn
// The procedure spawns "n" threads monitoring the task queue and executing each task. After the
// task is finished, the thread reacts to the returned signal.
inline void Threadpool::spawn(unsigned N) {

  if(is_worker()) {
    throw std::runtime_error("Worker cannot spawn threads");
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

        { // Acquire lock. --------------------------------------------------------------------------
          std::unique_lock<std::mutex> lock(_mutex);
          _worker_signal.wait(lock, [this] () { return _task_queue.size() != 0; });
          task = std::move(_task_queue.front());
          _task_queue.pop_front();
        } // Release lock. --------------------------------------------------------------------------

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
auto Threadpool::silent_async(C&& c, Signal sig) {
  
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
auto Threadpool::async(C&& c, Signal sig) {

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
      _task_queue.emplace_back(
        [p=MoveOnCopy(std::move(p)), c=std::forward<C>(c), ret=sig] () mutable { 
          if constexpr(std::is_same_v<void, R>) {
            c();
            p.get().set_value();
          }
          else {
            p.get().set_value(c());
          }
          return ret;
        }
      );
    }
    _worker_signal.notify_one();
  }
  return fu;
}

// Procedure: shutdown
// Remove a given number of workers. Notice that only the master can call this procedure.
inline void Threadpool::shutdown() {
  
  if(is_worker()) {
    throw std::runtime_error("Worker cannot shut down thread pool");
  }

  for(size_t i=0; i<_threads.size(); ++i) {
    async([](){}, Signal::SHUTDOWN);
  }
  
  for(auto& t : _threads) {
    t.join();
  }

  _threads.clear();
}

//  -----------------------------------------------------------------------------------------------

// Class: Taskflow
class Taskflow {

  struct Task {

    const int64_t key {-1};
    const std::function<void()> work;
    
    std::vector<Task*> successors;
    std::atomic<int> dependents {0};

    Task() = default;

    template <typename C>
    Task(int64_t, C&&);

    template <typename C>
    Task(C&&);

    inline void precede(Task&);
  };

  public:
    
    Taskflow(auto);
    
    template <typename C>
    auto emplace(C&&);

    template <typename C>
    auto silent_emplace(C&&);

    auto dispatch();

    inline void precede(int64_t, int64_t);
    inline void wait_for_all();

    inline size_t num_tasks() const;
    inline size_t num_workers() const;

    inline std::string dump() const;

  private:

    Threadpool _threadpool;

    inline int64_t _unique_key() const;
    
    inline void _schedule(Task&);

    std::unordered_map<int64_t, Task> _tasks;
};
    
template <typename C>
Taskflow::Task::Task(int64_t k, C&& c) : key {k}, work {std::forward<C>(c)} {
}

template <typename C>
Taskflow::Task::Task(C&& c) : work {std::forward<C>(c)} {
}

// Procedure: precede
inline void Taskflow::Task::precede(Task& v) {
  successors.push_back(&v);
  v.dependents++;
}

// Constructor
inline Taskflow::Taskflow(auto N) : _threadpool{N} {
}

// Function: _unique_key
inline int64_t Taskflow::_unique_key() const {
  static int64_t _key {0};
  return _key++;
}

// Function: num_tasks
inline size_t Taskflow::num_tasks() const {
  return _tasks.size();
}

// Function: num_workers
inline size_t Taskflow::num_workers() const {
  return _threadpool.num_workers();
}

// Procedure: precede
inline void Taskflow::precede(int64_t from, int64_t to) {

  auto fitr = _tasks.find(from);
  auto titr = _tasks.find(to);

  if(fitr == _tasks.end() or titr == _tasks.end()) {
    throw std::runtime_error("task not found in the taskflow");
  }

  fitr->second.precede(titr->second);
}

// Procedure: wait_for_all
inline void Taskflow::wait_for_all() {

  if(_tasks.empty()) return;

  // TODO: add code to detect cycle
  
  // Create a barrier.
  std::promise<void> barrier;
  auto future = barrier.get_future();
  
  // Create a source/target
  Task source;
  Task target{-1, [&barrier] () mutable { barrier.set_value(); }};

  source.precede(target);
  
  // Build the super source and super target.
  for(auto& kvp : _tasks) {
    if(kvp.second.dependents == 0) {
      source.precede(kvp.second);
    }
    if(kvp.second.successors.size() == 0) {
      kvp.second.precede(target);
    }
  }

  // Start the taskflow
  _schedule(source);
  
  // Wait until all finishes.
  future.get();

  // clean up the tasks
  _tasks.clear();
}

// Function: silent_emplace
template <typename C>
auto Taskflow::silent_emplace(C&& c) {
  
  const auto key = _unique_key();  
  
  if(auto ret = _tasks.try_emplace(key, key, std::forward<C>(c)); !ret.second) {
    throw std::runtime_error("failed to insert task to taskflow");
  }
  
  return key;
}

// Function: emplace
template <typename C>
auto Taskflow::emplace(C&& c) {
  
  const auto key = _unique_key();  
  
  using R = std::invoke_result_t<C>;
  
  std::promise<R> p;
  auto fu = p.get_future();

  auto ret = _tasks.try_emplace(key, key, 
    [p=MoveOnCopy(std::move(p)), c=std::forward<C>(c)] () mutable { 
      if constexpr(std::is_same_v<void, R>) {
        c();
        p.get().set_value();
      }
      else {
        p.get().set_value(c());
      }
    }
  );
  
  if(ret.second == false) {
    throw std::runtime_error("failed to insert task to taskflow");
  }
  
  return std::make_tuple(key, std::move(fu));
}

// Procedure: _schedule
inline void Taskflow::_schedule(Task& task) {
  _threadpool.silent_async([this, &task](){
    if(task.work) {
      task.work();
    }
    for(const auto& succ : task.successors) {
      if(--(succ->dependents) == 0) {
        _schedule(*succ);
      }
    }
  });
}

// Function: dump
inline std::string Taskflow::dump() const {
  std::ostringstream oss;  
  for(const auto& t : _tasks) {
    oss << "Task \"" << t.second.key << "\" [dependents:" << t.second.dependents.load()
        << "|successors:" << t.second.successors.size() << "]\n";
    for(const auto s : t.second.successors) {
      oss << "  |--> " << "task \"" << s->key << "\"\n";
    }
  }
  return oss.str();
}





};  // end of namespace tf. -----------------------------------------------------------------------


#endif












