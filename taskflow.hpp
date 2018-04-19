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
#include <list>

namespace tf {

inline void __throw__(const char* fname, const size_t line, auto&&... args) {
  std::ostringstream oss;
  oss << "[" << fname << ":" << line << "] ";
  (oss << ... << args);
  throw std::runtime_error(oss.str());
}

#define TF_THROW(...) __throw__(__FILE__, __LINE__, __VA_ARGS__);

// ------------------------------------------------------------------------------------------------

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
template <typename KeyT = int64_t>
class Taskflow {

  struct Task {

    KeyT key;
    std::function<void()> work;
    
    std::vector<Task*> successors;
    std::atomic<int> dependents {0};

    Task() = default;

    template <typename C>
    Task(KeyT, C&&);

    template <typename C>
    Task(C&&);

    inline void precede(Task&);
  };

  struct Topology{
    Topology(std::unordered_map<KeyT, Task>&&);

    std::unordered_map<KeyT, Task> tasks;
    std::shared_future<void> future;
    Task source;
    Task target;
  };


  public:
    
    Taskflow(auto);
    
    template <typename C>
    auto emplace(C&&);

    template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
    auto emplace(C&&...);

    template <typename C>
    auto silent_emplace(C&&);

    template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
    auto silent_emplace(C&&...);

    inline auto dispatch();
    inline auto silent_dispatch();


    Taskflow& precede(const KeyT&, const KeyT&);
    Taskflow& linearize(const std::vector<KeyT>&);
    Taskflow& linearize(std::initializer_list<KeyT>);
    Taskflow& broadcast(const KeyT&, const std::vector<KeyT>&);
    Taskflow& broadcast(const KeyT&, std::initializer_list<KeyT>);
    Taskflow& gather(const std::vector<KeyT>&, const KeyT&);
    Taskflow& gather(std::initializer_list<KeyT>, const KeyT&);
    Taskflow& wait_for_all();

    size_t num_tasks() const;
    size_t num_workers() const;

    std::string dump() const;

  private:

    Threadpool _threadpool;

    KeyT _gen {0};

    std::unordered_map<KeyT, Task> _tasks;
    std::list<Topology> _topologies;
    
    void _schedule(Task&);

    template <typename L>
    void _linearize(const L&);

    template <typename L>
    void _broadcast(const KeyT&, const L&);

    template <typename L>
    void _gather(const L&, const KeyT&);
};


template<typename KeyT>
Taskflow<KeyT>::Topology::Topology(std::unordered_map<KeyT, Task>&& t) : 
  tasks(std::move(t)) {
  
  std::promise<void> promise;

  future = promise.get_future().share();
  target.work = [p=MoveOnCopy{std::move(promise)}] () mutable { p.get().set_value(); };

  source.precede(target);

  // Build the super source and super target.
  for(auto& kvp : tasks) {
    if(kvp.second.dependents == 0) {
      source.precede(kvp.second);
    }
    if(kvp.second.successors.size() == 0) {
      kvp.second.precede(target);
    }
  }
}

    
template <typename KeyT>
template <typename C>
Taskflow<KeyT>::Task::Task(KeyT k, C&& c) : key {k}, work {std::forward<C>(c)} {
}

template <typename KeyT>
template <typename C>
Taskflow<KeyT>::Task::Task(C&& c) : work {std::forward<C>(c)} {
}

// Procedure: precede
template <typename KeyT>
void Taskflow<KeyT>::Task::precede(Task& v) {
  successors.push_back(&v);
  v.dependents++;
}

// Constructor
template <typename KeyT>
Taskflow<KeyT>::Taskflow(auto N) : _threadpool{N} {
}

// Function: num_tasks
template <typename KeyT>
size_t Taskflow<KeyT>::num_tasks() const {
  return _tasks.size();
}

// Function: num_workers
template <typename KeyT>
size_t Taskflow<KeyT>::num_workers() const {
  return _threadpool.num_workers();
}

// Procedure: precede
template <typename KeyT>
Taskflow<KeyT>& Taskflow<KeyT>::precede(const KeyT& from, const KeyT& to) {

  auto fitr = _tasks.find(from);
  auto titr = _tasks.find(to);

  if(fitr == _tasks.end() or titr == _tasks.end()) {
    TF_THROW("precede error (invalid pair ", from, "->", to, ")");
  }

  fitr->second.precede(titr->second);

  return *this;
}

// Procedure: _linearize
template <typename KeyT>
template <typename L>
void Taskflow<KeyT>::_linearize(const L& keys) {

  std::adjacent_find(keys.begin(), keys.end(), [this] (const KeyT& from, const KeyT& to) {

    auto fitr = _tasks.find(from);
    auto titr = _tasks.find(to);

    if(fitr == _tasks.end() or titr == _tasks.end()) {
      TF_THROW("linearize error (invalid key ", from, "->", to, ")");
    }
    fitr->second.precede(titr->second); 
    return false;
  });

}

// Procedure: linearize
template <typename KeyT>
Taskflow<KeyT>& Taskflow<KeyT>::linearize(const std::vector<KeyT>& keys) {
  _linearize(keys); 
  return *this;
}

// Procedure: linearize
template <typename KeyT>
Taskflow<KeyT>& Taskflow<KeyT>::linearize(std::initializer_list<KeyT> keys) {
  _linearize(keys);
  return *this;
}

// Procedure: broadcast
template <typename KeyT>
template <typename S>
void Taskflow<KeyT>::_broadcast(const KeyT& from, const S& keys) {

  auto fitr = _tasks.find(from);

  if(fitr == _tasks.end()) {
    TF_THROW("broadcast error (invalid from key ", from, ")");
  }

  for(const auto& to : keys) {
    if(auto titr = _tasks.find(to); titr == _tasks.end()) {
      TF_THROW("broadcast error (invalid to key ", to, ")");
    }
    else {
      fitr->second.precede(titr->second);
    }
  }
}

// Procedure: broadcast
template <typename KeyT>
Taskflow<KeyT>& Taskflow<KeyT>::broadcast(const KeyT& from, const std::vector<KeyT>& keys) {
  _broadcast(from, keys);
  return *this;
}

// Procedure: broadcast
template <typename KeyT>
Taskflow<KeyT>& Taskflow<KeyT>::broadcast(const KeyT& from, std::initializer_list<KeyT> keys) {
  _broadcast(from, keys);
  return *this;
}

// Procedure: gather
template <typename KeyT>
template <typename S>
void Taskflow<KeyT>::_gather(const S& keys, const KeyT& to) {

  auto titr = _tasks.find(to);

  if(titr == _tasks.end()) {
    TF_THROW("gather error (invalid to key ", to, ")");
  }

  for(const auto& from : keys) {
    if(auto fitr = _tasks.find(from); fitr == _tasks.end()) {
      TF_THROW("gather error (invalid from key ", from, ")");
    }
    else {
      fitr->second.precede(titr->second);
    }
  }
}

// Function: gather
template <typename KeyT>
Taskflow<KeyT>& Taskflow<KeyT>::gather(const std::vector<KeyT>& keys, const KeyT& to) {
  _gather(keys, to);
  return *this;
}

// Function: gather
template <typename KeyT>
Taskflow<KeyT>& Taskflow<KeyT>::gather(std::initializer_list<KeyT> keys, const KeyT& to) {
  _gather(keys, to);
  return *this;
}

// Procedure: wait_for_all
template <typename KeyT>
Taskflow<KeyT>& Taskflow<KeyT>::wait_for_all() {
  silent_dispatch();
  for(auto& t: _topologies){
    t.future.get();
  }
  return *this;

  /*
  if(_tasks.empty()) return *this;

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

  return *this;
  */
}

// Function: silent_emplace
template <typename KeyT>
template <typename C>
auto Taskflow<KeyT>::silent_emplace(C&& c) {
  
  const auto key = ++_gen;  
  
  if(auto ret = _tasks.try_emplace(key, key, std::forward<C>(c)); !ret.second) {
    TF_THROW("silent_emplace failed (dumplicate key ", key, ")");
  }
  
  return key;
}

template <typename KeyT>
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto Taskflow<KeyT>::silent_emplace(C&&... cs) {
  return std::make_tuple(silent_emplace(std::forward<C>(cs))...);
}

// Function: emplace
template <typename KeyT>
template <typename C>
auto Taskflow<KeyT>::emplace(C&& c) {
  
  const auto key = ++_gen;
  
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
    TF_THROW("emplace failed (dumplicate key ", key, ")");
  }
  
  return std::make_pair(key, std::move(fu));
}

template <typename KeyT>
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto Taskflow<KeyT>::emplace(C&&... cs) {
  return std::make_tuple(emplace(std::forward<C>(cs))...);
}

// Procedure: _schedule
template <typename KeyT>
void Taskflow<KeyT>::_schedule(Task& task) {
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
template <typename KeyT>
std::string Taskflow<KeyT>::dump() const {
  std::ostringstream oss;  
  for(const auto& [key, t] : _tasks) {
    oss << "Task \"" << key << "\" [dependents:" << t.dependents.load()
        << "|successors:" << t.successors.size() << "]\n";
    for(const auto s : t.successors) {
      oss << "  |--> " << "task \"" << s->key << "\"\n";
    }
  }
  return oss.str();
}



// Procedure: silent_dispatch 
template <typename KeyT>
inline auto Taskflow<KeyT>::silent_dispatch() {

  if(_tasks.empty()) return;

  auto& topology = _topologies.emplace_back(std::move(_tasks));

  // Start the taskflow
  _schedule(topology.source);
}


// Procedure: dispatch 
template <typename KeyT>
inline auto Taskflow<KeyT>::dispatch() {

  if(_tasks.empty()) {
    return std::async(std::launch::deferred, [](){}).share();
  }

  auto& topology = _topologies.emplace_back(std::move(_tasks));

  // Start the taskflow
  _schedule(topology.source);
  
  return topology.future;
}




};  // end of namespace tf. -----------------------------------------------------------------------


#endif












