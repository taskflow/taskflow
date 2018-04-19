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

// Struct: Task
template <typename F = std::function<void()>>
struct Task {

  template <typename T> friend class Taskflow;

  public:
    
    Task() = default;

    template <typename C>
    Task(C&&);
    
    void precede(Task&);

    void name(const std::string&);
    const std::string& name() const;

    size_t num_successors() const;
    size_t dependents() const;

    std::string dump() const;

  private:

    std::string _name;

    F _work;
    
    std::vector<Task*> _successors;
    std::atomic<int> _dependents {0};
};
  
// Constructor
template <typename F>
template <typename C>
Task<F>::Task(C&& c) : _work {std::forward<C>(c)} {
}

// Procedure:
template <typename F>
void Task<F>::precede(Task& v) {
  _successors.push_back(&v);
  v._dependents++;
}

// Function: name
template <typename F>
void Task<F>::name(const std::string& n) {
  _name = n;
}

// Function: name
template <typename F>
const std::string& Task<F>::name() const {
  return _name;
}

// Function: num_successors
template <typename F>
size_t Task<F>::num_successors() const {
  return _successors.size();
}

// Function: dependents
template <typename F>
size_t Task<F>::dependents() const {
  return _dependents.load();
}

// Procedure: dump
template <typename F>
std::string Task<F>::dump() const {

  std::ostringstream oss;

  oss << "Task \"";

  if(_name.empty()) {
    oss << this;
  }
  else {
    oss << _name;
  }
  
  oss << "\" [dependents:" << dependents()
      << "|successors:" << num_successors() << "]\n";

  for(const auto s : _successors) {
    oss << "  |--> " << "task \"";
    if(s->_name.empty()) {
      oss << s;
    }
    else {
      oss << s->_name;
    }
    oss << "\"\n";
  }

  return oss.str();
}

// ------------------------------------------------------------------------------------------------

// Class: Taskflow
template <typename F = std::function<void()>>
class Taskflow {
  
  public:
  
  using task_type = Task<F>;
  
  // Struct: Topology
  struct Topology{

    Topology(std::list<task_type>&&);

    std::list<task_type> tasks;
    std::shared_future<void> future;

    task_type source;
    task_type target;
  };
  
  // Class: TaskBuilder
  class TaskBuilder {

    friend class Taskflow;
  
    public:
      
      TaskBuilder() = delete;
      TaskBuilder(const TaskBuilder&);
      TaskBuilder(TaskBuilder&&);
  
      operator const auto ();
      const auto operator -> ();

      TaskBuilder& name(const std::string&);
      TaskBuilder& precede(TaskBuilder);
      TaskBuilder& broadcast(std::vector<TaskBuilder>&);
      TaskBuilder& broadcast(std::initializer_list<TaskBuilder>);
      TaskBuilder& gather(std::vector<TaskBuilder>&);
      TaskBuilder& gather(std::initializer_list<TaskBuilder>);

    private:
  
      TaskBuilder(task_type*);
  
      task_type* _task {nullptr};

      template<typename S>
      void _broadcast(S&);

      template<typename S>
      void _gather(S&);
  };

    
    Taskflow(unsigned);
    
    template <typename C>
    auto emplace(C&&);

    template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
    auto emplace(C&&...);

    template <typename C>
    auto silent_emplace(C&&);

    template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
    auto silent_emplace(C&&...);

    auto dispatch();
    auto silent_dispatch();

    Taskflow& precede(TaskBuilder, TaskBuilder);
    Taskflow& linearize(std::vector<TaskBuilder>&);
    Taskflow& linearize(std::initializer_list<TaskBuilder>);
    Taskflow& broadcast(TaskBuilder, std::vector<TaskBuilder>&);
    Taskflow& broadcast(TaskBuilder, std::initializer_list<TaskBuilder>);
    Taskflow& gather(std::vector<TaskBuilder>&, TaskBuilder);
    Taskflow& gather(std::initializer_list<TaskBuilder>, TaskBuilder);
    Taskflow& wait_for_all();

    size_t num_tasks() const;
    size_t num_workers() const;

    std::string dump() const;

  private:

    Threadpool _threadpool;

    std::list<task_type> _tasks;
    std::list<Topology> _topologies;

    void _schedule(task_type&);

    template <typename L>
    void _linearize(L&);
};

// Constructor
template <typename F>
Taskflow<F>::Topology::Topology(std::list<task_type>&& t) : 
  tasks(std::move(t)) {
  
  std::promise<void> promise;

  future = promise.get_future().share();
  target._work = [p=MoveOnCopy{std::move(promise)}] () mutable { p.get().set_value(); };

  source.precede(target);

  // Build the super source and super target.
  for(auto& task : tasks) {
    if(task.dependents() == 0) {
      source.precede(task);
    }
    if(task.num_successors() == 0) {
      task.precede(target);
    }
  }
}
    
// Constructor
template <typename F>
Taskflow<F>::TaskBuilder::TaskBuilder(const TaskBuilder& rhs) : _task{rhs._task} {
}

template <typename F>
typename Taskflow<F>::TaskBuilder& Taskflow<F>::TaskBuilder::precede(TaskBuilder tgt) {
  _task->precede(tgt._task);
  return *this;
}

template <typename F>
template <typename S>
void Taskflow<F>::TaskBuilder::_broadcast(S& tgts) {
  for(auto& to : tgts) {
    _task->precede(*(to._task));
  }
}
      
// Function: broadcast
template <typename F>
typename Taskflow<F>::TaskBuilder& Taskflow<F>::TaskBuilder::broadcast(std::vector<TaskBuilder>& tgts) {
  _broadcast(tgts);
  return *this;
}

// Function: broadcast
template <typename F>
typename Taskflow<F>::TaskBuilder& Taskflow<F>::TaskBuilder::broadcast(std::initializer_list<TaskBuilder> tgts) {
  _broadcast(tgts);
  return *this;
}

template <typename F>
template <typename S>
void Taskflow<F>::TaskBuilder::_gather(S& tgts) {
  for(auto& from : tgts) {
    from._task->precede(*_task);
  }
}

// Function: gather
template <typename F>
typename Taskflow<F>::TaskBuilder& Taskflow<F>::TaskBuilder::gather(std::vector<TaskBuilder>& tgts) {
  _gather(tgts);
  return *this;
}

// Function: gather
template <typename F>
typename Taskflow<F>::TaskBuilder& Taskflow<F>::TaskBuilder::gather(std::initializer_list<TaskBuilder> tgts) {
  _gather(tgts);
  return *this;
}

// Constructor
template <typename F>
Taskflow<F>::TaskBuilder::TaskBuilder(TaskBuilder&& rhs) : _task{rhs._task} { 
  rhs._task = nullptr; 
}

// Operator
template <typename F>  
Taskflow<F>::TaskBuilder::operator const auto () { 
  return _task; 
}

// Function: get
template <typename F>  
const auto Taskflow<F>::TaskBuilder::operator -> () { 
  return _task; 
}

// Function: name
template <typename F>
typename Taskflow<F>::TaskBuilder& Taskflow<F>::TaskBuilder::name(const std::string& name) {
  _task->name(name);
  return *this;
}

// Constructor
template <typename F>
Taskflow<F>::TaskBuilder::TaskBuilder(task_type* t) : _task {t} {
}

// Constructor
template <typename F>
Taskflow<F>::Taskflow(unsigned N) : _threadpool{N} {
}

// Function: num_tasks
template <typename F>
size_t Taskflow<F>::num_tasks() const {
  return _tasks.size();
}

// Function: num_workers
template <typename F>
size_t Taskflow<F>::num_workers() const {
  return _threadpool.num_workers();
}

// Procedure: precede
template <typename F>
Taskflow<F>& Taskflow<F>::precede(TaskBuilder from, TaskBuilder to) {
  from._task->precede(*(to._task));
  return *this;
}

// Procedure: _linearize
template <typename F>
template <typename L>
void Taskflow<F>::_linearize(L& keys) {
  std::adjacent_find(
    keys.begin(), keys.end(), 
    [this] (auto& from, auto& to) {
      from._task->precede(*(to._task));
      return false;
    }
  );
}

// Procedure: linearize
template <typename F>
Taskflow<F>& Taskflow<F>::linearize(std::vector<TaskBuilder>& keys) {
  _linearize(keys); 
  return *this;
}

// Procedure: linearize
template <typename F>
Taskflow<F>& Taskflow<F>::linearize(std::initializer_list<TaskBuilder> keys) {
  _linearize(keys);
  return *this;
}

// Procedure: broadcast
template <typename F>
Taskflow<F>& Taskflow<F>::broadcast(TaskBuilder from, std::vector<TaskBuilder>& keys) {
  from.broadcast(keys);
  return *this;
}

// Procedure: broadcast
template <typename F>
Taskflow<F>& Taskflow<F>::broadcast(TaskBuilder from, std::initializer_list<TaskBuilder> keys) {
  from.broadcast(keys);
  return *this;
}

// Function: gather
template <typename F>
Taskflow<F>& Taskflow<F>::gather(std::vector<TaskBuilder>& keys, TaskBuilder to) {
  to.gather(keys);
  return *this;
}

// Function: gather
template <typename F>
Taskflow<F>& Taskflow<F>::gather(std::initializer_list<TaskBuilder> keys, TaskBuilder to) {
  to.gather(keys);
  return *this;
}

// Procedure: silent_dispatch 
template <typename F>
auto Taskflow<F>::silent_dispatch() {

  if(_tasks.empty()) return;

  //_topologies.remove_if([](auto &t){ 
  //   auto status = t.future.wait_for(std::chrono::seconds(0));
  //   if(status == std::future_status::ready){
  //     return true;
  //   }
  //  return false; 
  //});

  auto& topology = _topologies.emplace_back(std::move(_tasks));

  // Start the taskflow
  _schedule(topology.source);
}

// Procedure: dispatch 
template <typename F>
auto Taskflow<F>::dispatch() {

  if(_tasks.empty()) {
    return std::async(std::launch::deferred, [](){}).share();
  }

  //_topologies.remove_if([](auto &t){ 
  //   auto status = t.future.wait_for(std::chrono::seconds(0));
  //   if(status == std::future_status::ready){
  //     return true;
  //   }
  //  return false; 
  //});

  auto& topology = _topologies.emplace_back(std::move(_tasks));

  // Start the taskflow
  _schedule(topology.source);
  
  return topology.future;
}

// Procedure: wait_for_all
template <typename F>
Taskflow<F>& Taskflow<F>::wait_for_all() {

  if(!_tasks.empty()) {
    silent_dispatch();
  }

  for(auto& t: _topologies){
    t.future.get();
  }

  _topologies.clear();

  return *this;
}

// Function: silent_emplace
template <typename F>
template <typename C>
auto Taskflow<F>::silent_emplace(C&& c) {
  auto& task = _tasks.emplace_back(std::forward<C>(c));
  return TaskBuilder(&task);
}

template <typename F>
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto Taskflow<F>::silent_emplace(C&&... cs) {
  return std::make_tuple(silent_emplace(std::forward<C>(cs))...);
}

// Function: emplace
template <typename F>
template <typename C>
auto Taskflow<F>::emplace(C&& c) {
  
  using R = std::invoke_result_t<C>;
  
  std::promise<R> p;
  auto fu = p.get_future();

  auto& task = _tasks.emplace_back([p=MoveOnCopy(std::move(p)), c=std::forward<C>(c)] () mutable { 
    if constexpr(std::is_same_v<void, R>) {
      c();
      p.get().set_value();
    }
    else {
      p.get().set_value(c());
    }
  });
  
  return std::make_pair(TaskBuilder(&task), std::move(fu));
}

template <typename F>
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto Taskflow<F>::emplace(C&&... cs) {
  return std::make_tuple(emplace(std::forward<C>(cs))...);
}

// Procedure: _schedule
template <typename F>
void Taskflow<F>::_schedule(task_type& task) {
  _threadpool.silent_async([this, &task](){
    if(task._work) {
      task._work();
    }
    for(const auto& succ : task._successors) {
      if(--(succ->_dependents) == 0) {
        _schedule(*succ);
      }
    }
  });
}

//// Function: dump
template <typename F>
std::string Taskflow<F>::dump() const {
  std::ostringstream oss;  
  for(const auto& t : _tasks) {
    oss << t.dump();
  }
  return oss.str();
}


};  // end of namespace tf. -----------------------------------------------------------------------


#endif

