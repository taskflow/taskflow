// MIT License
// 
// Copyright (c) 2018 Tsung-Wei Huang, Chun-Xun Lin, and Martin Wong
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
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <list>
#include <forward_list>
#include <numeric>
#include <iomanip>
#include <cassert>
#include <variant>

namespace tf {

template <typename... ArgsT>
inline void throw_re(const char* fname, const size_t line, ArgsT&&... args) {
  std::ostringstream oss;
  oss << '[' << fname << ':' << line << "] ";
  (oss << ... << std::forward<ArgsT>(args));
  throw std::runtime_error(oss.str());
}

#define TF_THROW(...) throw_re(__FILE__, __LINE__, __VA_ARGS__);

//-------------------------------------------------------------------------------------------------
// Traits
//-------------------------------------------------------------------------------------------------

// Struct: dependent_false
template <typename... T>
struct dependent_false { 
  static constexpr bool value = false; 
};

// Struct: is_iterator
template <typename T, typename = void>
struct is_iterator {
  static constexpr bool value = false;
};

template <typename T>
struct is_iterator<T, std::enable_if_t<!std::is_same_v<typename std::iterator_traits<T>::value_type, void>>> {
  static constexpr bool value = true;
};

template <typename T>
inline constexpr bool is_iterator_v = is_iterator<T>::value;

// Struct: is_iterable
template <typename T, typename = void>
struct is_iterable : std::false_type {
};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::declval<T>().begin()),
                                  decltype(std::declval<T>().end())>>
  : std::true_type {
};

template <typename T>
inline constexpr bool is_iterable_v = is_iterable<T>::value;

//-------------------------------------------------------------------------------------------------
// Utility
//-------------------------------------------------------------------------------------------------

// Struct: MoveOnCopy
template <typename T>
struct MoveOnCopy {

  MoveOnCopy(T&& rhs) : object(std::move(rhs)) {}
  MoveOnCopy(const MoveOnCopy& other) : object(std::move(other.object)) {}

  T& get() { return object; }
  
  mutable T object; 
};

template <typename T>
MoveOnCopy(T&&) -> MoveOnCopy<T>;

//-------------------------------------------------------------------------------------------------
// Threadpool definition
//-------------------------------------------------------------------------------------------------

// Class: Threadpool
class Threadpool {

  enum class Signal {
    STANDARD,
    SHUTDOWN
  };

  public:

    inline Threadpool(unsigned);
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
inline Threadpool::Threadpool(unsigned N) {
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
          _worker_signal.wait(lock, [this] () { return _task_queue.size() != 0; });
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
      
      if constexpr(std::is_same_v<void, R>) {
        _task_queue.emplace_back(
          [p = MoveOnCopy(std::move(p)), c = std::forward<C>(c), ret = sig]() mutable {
            c();
            p.get().set_value();
            return ret;
          }
        );
      }
      else {
        _task_queue.emplace_back(
          [p = MoveOnCopy(std::move(p)), c = std::forward<C>(c), ret = sig]() mutable {
            p.get().set_value(c());
            return ret;
          }
        );
      }
      
      // This can cause MSVS not to compile ...
      /*_task_queue.emplace_back(
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
      );*/
    }
    _worker_signal.notify_one();
  }
  return fu;
}

// Procedure: shutdown
// Remove a given number of workers. Notice that only the master can call this procedure.
inline void Threadpool::shutdown() {
  
  if(is_worker()) {
    throw std::runtime_error("Worker thread cannot shut down the thread pool");
  }

  for(size_t i=0; i<_threads.size(); ++i) {
    silent_async([](){}, Signal::SHUTDOWN);
  }
  
  for(auto& t : _threads) {
    t.join();
  }

  _threads.clear();
}

//-------------------------------------------------------------------------------------------------
// Taskflow definition
//------------------------------------------------------------------------------------------------- 


template <typename Traits>
class BasicTaskflow;

class FlowBuilder;

using WorkType = std::function<void()>;
using SubworkType = std::function<void(FlowBuilder&)>;

// ------------------------------------------------------------------------------------------------

// Struct: Node
class Node {
  
  template <typename Traits> friend class BasicTaskflow;

  public:

  Node() = default;

  template <typename C>
  Node(C&&);

  const std::string& name() const;
  
  void precede(Node&);

  size_t num_successors() const;
  size_t dependents() const;

  std::string _name;

  std::variant<WorkType, SubworkType> _work;
  
  std::vector<Node*> _successors;
  std::atomic<int> _dependents {0};

  private:

    std::forward_list<Node> _subnodes;
};

// ------------------------------------------------------------------------------------------------

// Class: Task
class Task {

  friend class FlowBuilder;
  template<typename Traits> friend class BasicTaskflow;

  public:
    
    Task(const Task&);
    Task(Task&&);

    Task& operator = (const Task&);

    const std::string& name() const;

    size_t num_successors() const;
    size_t num_dependents() const;

    Task& name(const std::string&);
    Task& precede(Task);
    Task& broadcast(std::vector<Task>&);
    Task& broadcast(std::initializer_list<Task>);
    Task& gather(std::vector<Task>&);
    Task& gather(std::initializer_list<Task>);

    template <typename C>
    Task& work(C&&);
  
    template <typename... Bs>
    Task& broadcast(Bs&&...);

    template <typename... Bs>
    Task& gather(Bs&&...);

  private:

    Task(Node*);

    Node* _node {nullptr};

    template<typename S>
    void _broadcast(S&);

    template<typename S>
    void _gather(S&);
};

// ------------------------------------------------------------------------------------------------

class FlowBuilder{
  public:

   FlowBuilder(std::forward_list<Node>& nodes, size_t num_workers) : 
     _nodes {nodes}, _num_workers{num_workers} {
   }    

   template <typename C>
   auto emplace(C&&);

   template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
   auto emplace(C&&...);

   template <typename C>
   auto silent_emplace(C&&);

   template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
   auto silent_emplace(C&&...);

   template <typename I, typename C>
   auto parallel_for(I, I, C&&, size_t = 1);

   template <typename T, typename C, std::enable_if_t<is_iterable_v<T>, void>* = nullptr>
   auto parallel_for(T&, C&&, size_t = 1);

   template <typename I, typename T, typename B>
   std::pair<Task, Task> reduce(I, I, T&, B&&);

   template <typename I, typename T>
   auto reduce_min(I, I, T&);
   
   template <typename I, typename T>
   auto reduce_max(I, I, T&);

   template <typename I, typename T, typename B, typename U>
   std::pair<Task, Task> transform_reduce(I, I, T&, B&&, U&&);

   template <typename I, typename T, typename B, typename P, typename U>
   std::pair<Task, Task> transform_reduce(I, I, T&, B&&, P&&, U&&);
   
   Task placeholder();
   
   std::shared_future<void> dispatch();

   void precede(Task, Task);
   void linearize(std::vector<Task>&);
   void linearize(std::initializer_list<Task>);
   void broadcast(Task, std::vector<Task>&);
   void broadcast(Task, std::initializer_list<Task>);
   void gather(std::vector<Task>&, Task);
   void gather(std::initializer_list<Task>, Task);  

  private:

    std::forward_list<Node>& _nodes;
    size_t _num_workers;

    template <typename L>
    void _linearize(L&);
};

// ------------------------------------------------------------------------------------------------

// Class: BasicTaskflow
template <typename Traits>
class BasicTaskflow {
 
  // Struct: Topology
  struct Topology{

    Topology(std::forward_list<Node>&&);

    std::forward_list<Node> nodes;
    std::shared_future<void> future;

    Node source ;
    Node target ;
  };

  public:
  
    BasicTaskflow();
    BasicTaskflow(unsigned);
    ~BasicTaskflow();
    
    template <typename C>
    auto emplace(C&&);

    template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
    auto emplace(C&&...);

    template <typename C>
    auto silent_emplace(C&&);

    template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
    auto silent_emplace(C&&...);


    template <typename I, typename C>
    auto parallel_for(I, I, C&&, size_t = 1);

    template <typename T, typename C, std::enable_if_t<is_iterable_v<T>, void>* = nullptr>
    auto parallel_for(T&, C&&, size_t = 1);

    template <typename I, typename T, typename B>
    auto reduce(I, I, T&, B&&);

    template <typename I, typename T>
    auto reduce_min(I, I, T&);
    
    template <typename I, typename T>
    auto reduce_max(I, I, T&);

    template <typename I, typename T, typename B, typename U>
    std::pair<Task, Task> transform_reduce(I, I, T&, B&&, U&&);

    template <typename I, typename T, typename B, typename P, typename U>
    std::pair<Task, Task> transform_reduce(I, I, T&, B&&, P&&, U&&);

    Task placeholder();

    std::shared_future<void> dispatch();

    void precede(Task, Task);
    void linearize(std::vector<Task>&);
    void linearize(std::initializer_list<Task>);
    void broadcast(Task, std::vector<Task>&);
    void broadcast(Task, std::initializer_list<Task>);
    void gather(std::vector<Task>&, Task);
    void gather(std::initializer_list<Task>, Task);  
    void silent_dispatch();
    void wait_for_all();
    void wait_for_topologies();
    void num_workers(size_t);

    size_t num_nodes() const;
    size_t num_workers() const;
    size_t num_topologies() const;

    std::string dump() const;

  private:

    Threadpool _threadpool;

    std::forward_list<Node> _nodes;
    std::forward_list<Topology> _topologies;

    void _schedule(Node&);

    template <typename L>
    void _linearize(L&);
};

//---------------------------------------------------------
// Node
//---------------------------------------------------------

// Constructor
template <typename C>
Node::Node(C&& c) : _work {std::forward<C>(c)} {
}

// Procedure:
void Node::precede(Node& v) {
  _successors.push_back(&v);
  v._dependents++;
}

// Function: num_successors
size_t Node::num_successors() const {
  return _successors.size();
}

// Function: dependents
size_t Node::dependents() const {
  return _dependents.load();
}

// Function: name
const std::string& Node::name() const {
  return _name;
}



//---------------------------------------------------------
// BasicTaskflow::Topology
//---------------------------------------------------------

// Constructor
template <typename Traits>
BasicTaskflow<Traits>::Topology::Topology(std::forward_list<Node>&& t) : 
  nodes(std::move(t)) {
  
  std::promise<void> promise;

  future = promise.get_future().share();
  target._work = [p=MoveOnCopy{std::move(promise)}] () mutable { p.get().set_value(); };

  source.precede(target);

  // Build the super source and super target.
  for(auto& task : nodes) {
    if(task.dependents() == 0) {
      source.precede(task);
    }
    if(task.num_successors() == 0) {
      task.precede(target);
    }
  }
}

//---------------------------------------------------------
// Task
//---------------------------------------------------------
    
// Constructor
Task::Task(const Task& rhs) : _node {rhs._node} {
}

// Function: precede
Task& Task::precede(Task tgt) {
  _node->precede(*(tgt._node));
  return *this;
}

// Function: broadcast
template <typename... Bs>
Task& Task::broadcast(Bs&&... tgts) {
  (_node->precede(*(tgts._node)), ...);
  return *this;
}

// Procedure: _broadcast
template <typename S>
void Task::_broadcast(S& tgts) {
  for(auto& to : tgts) {
    _node->precede(*(to._node));
  }
}
      
// Function: broadcast
Task& Task::broadcast(std::vector<Task>& tgts) {
  _broadcast(tgts);
  return *this;
}

// Function: broadcast
Task& Task::broadcast(std::initializer_list<Task> tgts) {
  _broadcast(tgts);
  return *this;
}

// Function: gather
template <typename... Bs>
Task& Task::gather(Bs&&... tgts) {
  (tgts.precede(*this), ...);
  return *this;
}

// Procedure: _gather
template <typename S>
void Task::_gather(S& tgts) {
  for(auto& from : tgts) {
    from._node->precede(*_node);
  }
}

// Function: gather
Task& Task::gather(std::vector<Task>& tgts) {
  _gather(tgts);
  return *this;
}

// Function: gather
Task& Task::gather(std::initializer_list<Task> tgts) {
  _gather(tgts);
  return *this;
}

// Operator =
Task& Task::operator = (const Task& rhs) {
  _node = rhs._node;
  return *this;
}

// Constructor
Task::Task(Task&& rhs) : _node{rhs._node} { 
  rhs._node = nullptr; 
}

// Function: work
template <typename C>
Task& Task::work(C&& c) {
  _node->_work = std::forward<C>(c);
  return *this;
}

// Function: name
Task& Task::name(const std::string& name) {
  _node->_name = name;
  return *this;
}

// Function: name
const std::string& Task::name() const {
  return _node->_name;
}

// Function: num_dependents
size_t Task::num_dependents() const {
  return _node->_dependents;
}

// Function: num_successors
size_t Task::num_successors() const {
  return _node->_successors.size();
}

// Constructor
Task::Task(Node* t) : _node {t} {
}



//---------------------------------------------------------
// FlowBuilder
//---------------------------------------------------------

// Procedure: precede
void FlowBuilder::precede(Task from, Task to) {
    from._node->precede(*(to._node));
}

//// Procedure: _broadcast
//template <typename S>
//void FlowBuilder::_broadcast(S& tgts) {
//  for(auto& to : tgts) {
//    _node->precede(*(to._node));
//  }
//}

// Procedure: broadcast
void FlowBuilder::broadcast(Task from, std::vector<Task>& keys) {
  from.broadcast(keys);
}

// Procedure: broadcast
void FlowBuilder::broadcast(Task from, std::initializer_list<Task> keys) {
  from.broadcast(keys);
}

//// Procedure: _gather
//template <typename S>
//void FlowBuilder::_gather(S& tgts) {
//  for(auto& from : tgts) {
//    from._node->precede(*_node);
//  }
//}


// Function: gather
void FlowBuilder::gather(std::vector<Task>& keys, Task to) {
  to.gather(keys);
}

// Function: gather
void FlowBuilder::gather(std::initializer_list<Task> keys, Task to) {
  to.gather(keys);
}

// Function: placeholder
Task FlowBuilder::placeholder() {
  auto& node = _nodes.emplace_front();
  return Task(&node);
}

// Function: emplace
template <typename C>
auto FlowBuilder::emplace(C&& c) {
  
  using R = std::invoke_result_t<C>;
  
  std::promise<R> p;
  auto fu = p.get_future();

  auto& node = _nodes.emplace_front([p=MoveOnCopy(std::move(p)), c=std::forward<C>(c)] () mutable { 
    if constexpr(std::is_same_v<void, R>) {
      c();
      p.get().set_value();
    }
    else {
      p.get().set_value(c());
    }
  });
  
  return std::make_pair(Task(&node), std::move(fu));
}

// Function: emplace
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto FlowBuilder::emplace(C&&... cs) {
  return std::make_tuple(emplace(std::forward<C>(cs))...);
}

// Function: silent_emplace
template <typename C>
auto FlowBuilder::silent_emplace(C&& c) {
  auto& node = _nodes.emplace_front(std::forward<C>(c));
  return Task(&node);
}

// Function: silent_emplace
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto FlowBuilder::silent_emplace(C&&... cs) {
  return std::make_tuple(silent_emplace(std::forward<C>(cs))...);
}


// Function: parallel_for    
template <typename I, typename C>
auto FlowBuilder::parallel_for(I beg, I end, C&& c, size_t g) {

  using category = typename std::iterator_traits<I>::iterator_category;

  if(g == 0) {
    auto d = std::distance(beg, end);
    auto w = std::max(size_t{1}, _num_workers);
    g = (d + w - 1) / w;
  }

  auto source = placeholder();
  auto target = placeholder();
  
  while(beg != end) {

    auto e = beg;
    
    // Case 1: random access iterator
    if constexpr(std::is_same_v<category, std::random_access_iterator_tag>) {
      size_t r = std::distance(beg, end);
      std::advance(e, std::min(r, g));
    }
    // Case 2: non-random access iterator
    else {
      for(size_t i=0; i<g && e != end; ++e, ++i);
    }
      
    // Create a task
    auto task = silent_emplace([beg, e, c] () mutable {
      std::for_each(beg, e, c);
    });
    source.precede(task);
    task.precede(target);

    // adjust the pointer
    beg = e;
  }

  return std::make_pair(source, target); 
}

// Function: parallel_for
template <typename T, typename C, std::enable_if_t<is_iterable_v<T>, void>*>
auto FlowBuilder::parallel_for(T& t, C&& c, size_t group) {
  return parallel_for(t.begin(), t.end(), std::forward<C>(c), group);
}



// Function: reduce_min
// Find the minimum element over a range of items.
template <typename I, typename T>
auto FlowBuilder::reduce_min(I beg, I end, T& result) {
  return reduce(beg, end, result, [] (const auto& l, const auto& r) {
    return std::min(l, r);
  });
}

// Function: reduce_max
// Find the maximum element over a range of items.
template <typename I, typename T>
auto FlowBuilder::reduce_max(I beg, I end, T& result) {
  return reduce(beg, end, result, [] (const auto& l, const auto& r) {
    return std::max(l, r);
  });
}

// Function: transform_reduce    
template <typename I, typename T, typename B, typename U>
std::pair<Task, Task> FlowBuilder::transform_reduce(I beg, I end, T& result, B&& bop, U&& uop) {

  using category = typename std::iterator_traits<I>::iterator_category;
  
  // Even partition
  size_t d = std::distance(beg, end);
  size_t w = std::max(size_t{1}, _num_workers);
  size_t g = std::max((d + w - 1) / w, size_t{2});

  auto source = placeholder();
  auto target = placeholder();

  std::vector<std::future<T>> futures;

  while(beg != end) {

    auto e = beg;
    
    // Case 1: random access iterator
    if constexpr(std::is_same_v<category, std::random_access_iterator_tag>) {
      size_t r = std::distance(beg, end);
      std::advance(e, std::min(r, g));
    }
    // Case 2: non-random access iterator
    else {
      for(size_t i=0; i<g && e != end; ++e, ++i);
    }
      
    // Create a task
    auto [task, future] = emplace([beg, e, bop, uop] () mutable {
      auto init = uop(*beg);
      for(++beg; beg != e; ++beg) {
        init = bop(std::move(init), uop(*beg));          
      }
      return init;
    });
    source.precede(task);
    task.precede(target);
    futures.push_back(std::move(future));

    // adjust the pointer
    beg = e;
  }

  // target synchronizer
  target.work([&result, futures=MoveOnCopy{std::move(futures)}, bop] () {
    for(auto& fu : futures.object) {
      result = bop(std::move(result), fu.get());
    }
  });

  return std::make_pair(source, target); 
}

// Function: transform_reduce    
template <typename I, typename T, typename B, typename P, typename U>
std::pair<Task, Task> FlowBuilder::transform_reduce(
  I beg, I end, T& result, B&& bop, P&& pop, U&& uop
) {

  using category = typename std::iterator_traits<I>::iterator_category;
  
  // Even partition
  size_t d = std::distance(beg, end);
  size_t w = std::max(size_t{1}, _num_workers);
  size_t g = std::max((d + w - 1) / w, size_t{2});

  auto source = placeholder();
  auto target = placeholder();

  std::vector<std::future<T>> futures;

  while(beg != end) {

    auto e = beg;
    
    // Case 1: random access iterator
    if constexpr(std::is_same_v<category, std::random_access_iterator_tag>) {
      size_t r = std::distance(beg, end);
      std::advance(e, std::min(r, g));
    }
    // Case 2: non-random access iterator
    else {
      for(size_t i=0; i<g && e != end; ++e, ++i);
    }
      
    // Create a task
    auto [task, future] = emplace([beg, e, uop, pop] () mutable {
      auto init = uop(*beg);
      for(++beg; beg != e; ++beg) {
        init = pop(std::move(init), *beg);
      }
      return init;
    });
    source.precede(task);
    task.precede(target);
    futures.push_back(std::move(future));

    // adjust the pointer
    beg = e;
  }

  // target synchronizer
  target.work([&result, futures=MoveOnCopy{std::move(futures)}, bop] () {
    for(auto& fu : futures.object) {
      result = bop(std::move(result), fu.get());
    }
  });

  return std::make_pair(source, target); 
}


// Procedure: _linearize
template <typename L>
void FlowBuilder::_linearize(L& keys) {
  std::adjacent_find(
    keys.begin(), keys.end(), 
    [] (auto& from, auto& to) {
      from._node->precede(*(to._node));
      return false;
    }
  );
}

// Procedure: linearize
void FlowBuilder::linearize(std::vector<Task>& keys) {
  _linearize(keys); 
}

// Procedure: linearize
void FlowBuilder::linearize(std::initializer_list<Task> keys) {
  _linearize(keys);
}



// Proceduer: reduce
template <typename I, typename T, typename B>
std::pair<Task, Task> FlowBuilder::reduce(I beg, I end, T& result, B&& op) {
  
  using category = typename std::iterator_traits<I>::iterator_category;
  
  size_t d = std::distance(beg, end);
  size_t w = std::max(size_t{1}, _num_workers);
  size_t g = std::max((d + w - 1) / w, size_t{2});

  auto source = placeholder();
  auto target = placeholder();

  std::vector<std::future<T>> futures;
  
  while(beg != end) {

    auto e = beg;
    
    // Case 1: random access iterator
    if constexpr(std::is_same_v<category, std::random_access_iterator_tag>) {
      size_t r = std::distance(beg, end);
      std::advance(e, std::min(r, g));
    }
    // Case 2: non-random access iterator
    else {
      for(size_t i=0; i<g && e != end; ++e, ++i);
    }
      
    // Create a task
    auto [task, future] = emplace([beg, e, op] () mutable {
      auto init = *beg;
      for(++beg; beg != e; ++beg) {
        init = op(std::move(init), *beg);          
      }
      return init;
    });
    source.precede(task);
    task.precede(target);
    futures.push_back(std::move(future));

    // adjust the pointer
    beg = e;
  }
  
  // target synchronizer
  target.work([&result, futures=MoveOnCopy{std::move(futures)}, op] () {
    for(auto& fu : futures.object) {
      result = op(std::move(result), fu.get());
    }
  });

  return std::make_pair(source, target); 
}




//---------------------------------------------------------
// BasicTaskflow
//---------------------------------------------------------

// Constructor
template <typename Traits>
BasicTaskflow<Traits>::BasicTaskflow() : _threadpool{std::thread::hardware_concurrency()} {
}

// Constructor
template <typename Traits>
BasicTaskflow<Traits>::BasicTaskflow(unsigned N) : _threadpool{N} {
}

// Destructor
template <typename Traits>
BasicTaskflow<Traits>::~BasicTaskflow() {
  wait_for_topologies();
}

// Procedure: num_workers
template <typename Traits>
void BasicTaskflow<Traits>::num_workers(size_t W) {
  _threadpool.shutdown();
  _threadpool.spawn(W);
}

// Function: num_nodes
template <typename Traits>
size_t BasicTaskflow<Traits>::num_nodes() const {
  //return _nodes.size();
  return std::distance(_nodes.begin(), _nodes.end());
}

// Function: num_workers
template <typename Traits>
size_t BasicTaskflow<Traits>::num_workers() const {
  return _threadpool.num_workers();
}

// Function: num_topologies
template <typename Traits>
size_t BasicTaskflow<Traits>::num_topologies() const {
  return _topologies.size();
}

// Procedure: precede
template <typename Traits>
void BasicTaskflow<Traits>::precede(Task from, Task to) {
  from._node->precede(*(to._node));
}

// Procedure: _linearize
//template <typename Traits>
//template <typename L>
//void BasicTaskflow<Traits>::_linearize(L& keys) {
//  std::adjacent_find(
//    keys.begin(), keys.end(), 
//    [] (auto& from, auto& to) {
//      from._node->precede(*(to._node));
//      return false;
//    }
//  );
//}

// Procedure: linearize
template <typename Traits>
void BasicTaskflow<Traits>::linearize(std::vector<Task>& keys) {
  FlowBuilder(_nodes, num_workers()).linearize(keys);
  //_linearize(keys); 
}

// Procedure: linearize
template <typename Traits>
void BasicTaskflow<Traits>::linearize(std::initializer_list<Task> keys) {
  FlowBuilder(_nodes, num_workers()).linearize(keys);
  //_linearize(keys);
}

// Procedure: broadcast
template <typename Traits>
void BasicTaskflow<Traits>::broadcast(Task from, std::vector<Task>& keys) {
  from.broadcast(keys);
}

// Procedure: broadcast
template <typename Traits>
void BasicTaskflow<Traits>::broadcast(Task from, std::initializer_list<Task> keys) {
  from.broadcast(keys);
}


// Function: gather
template <typename Traits>
void BasicTaskflow<Traits>::gather(std::vector<Task>& keys, Task to) {
  to.gather(keys);
}

// Function: gather
template <typename Traits>
void BasicTaskflow<Traits>::gather(std::initializer_list<Task> keys, Task to) {
  to.gather(keys);
}

// Procedure: silent_dispatch 
template <typename Traits>
void BasicTaskflow<Traits>::silent_dispatch() {

  if(_nodes.empty()) return;

  auto& topology = _topologies.emplace_front(std::move(_nodes));

  // Start the taskflow
  _schedule(topology.source);
}

// Procedure: dispatch 
template <typename Traits>
std::shared_future<void> BasicTaskflow<Traits>::dispatch() {

  if(_nodes.empty()) {
    return std::async(std::launch::deferred, [](){}).share();
  }

  auto& topology = _topologies.emplace_front(std::move(_nodes));

  // Start the taskflow
  _schedule(topology.source);
  
  return topology.future;
}

// Procedure: wait_for_all
template <typename Traits>
void BasicTaskflow<Traits>::wait_for_all() {
  if(!_nodes.empty()) {
    silent_dispatch();
  }
  wait_for_topologies();
}

// Procedure: wait_for_topologies
template <typename Traits>
void BasicTaskflow<Traits>::wait_for_topologies() {
  for(auto& t: _topologies){
    t.future.get();
  }
  _topologies.clear();
}

// Function: placeholder
template <typename Traits>
Task BasicTaskflow<Traits>::placeholder() {
  return FlowBuilder(_nodes, num_workers()).placeholder();
  //auto& node = _nodes.emplace_front();
  //return Task(&node);
}

// Function: silent_emplace
template <typename Traits>
template <typename C>
auto BasicTaskflow<Traits>::silent_emplace(C&& c) {
  return FlowBuilder(_nodes, num_workers()).silent_emplace(std::forward<C>(c));
  //auto& node = _nodes.emplace_front(std::forward<C>(c));
  //return Task(&node);
}

// Function: silent_emplace
template <typename Traits>
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto BasicTaskflow<Traits>::silent_emplace(C&&... cs) {
  return FlowBuilder(_nodes, num_workers()).silent_emplace(std::forward<C>(cs)...);
  //return std::make_tuple(silent_emplace(std::forward<C>(cs))...);
}

// Function: emplace
template <typename Traits>
template <typename C>
auto BasicTaskflow<Traits>::emplace(C&& c) {
  return FlowBuilder(_nodes, num_workers()).emplace(std::forward<C>(c));
  
  //using R = std::invoke_result_t<C>;
  //
  //std::promise<R> p;
  //auto fu = p.get_future();

  //auto& node = _nodes.emplace_front([p=MoveOnCopy(std::move(p)), c=std::forward<C>(c)] () mutable { 
  //  if constexpr(std::is_same_v<void, R>) {
  //    c();
  //    p.get().set_value();
  //  }
  //  else {
  //    p.get().set_value(c());
  //  }
  //});
  //
  //return std::make_pair(Task(&node), std::move(fu));
}

// Function: emplace
template <typename Traits>
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto BasicTaskflow<Traits>::emplace(C&&... cs) {
  return FlowBuilder(_nodes, num_workers()).emplace(std::forward<C>(cs)...);
  //return std::make_tuple(emplace(std::forward<C>(cs))...);
}

// Function: parallel_for    
template <typename Traits>
template <typename I, typename C>
auto BasicTaskflow<Traits>::parallel_for(I beg, I end, C&& c, size_t g) {
  return FlowBuilder(_nodes, num_workers()).parallel_for(beg, end, std::forward<C>(c), g);
}

// Function: parallel_for
template <typename Traits>
template <typename T, typename C, std::enable_if_t<is_iterable_v<T>, void>*>
auto BasicTaskflow<Traits>::parallel_for(T& t, C&& c, size_t group) {
  return FlowBuilder(_nodes, num_workers()).parallel_for(t, std::forward<C>(c), group);
  //return parallel_for(t.begin(), t.end(), std::forward<C>(c), group);
}

// Function: reduce 
template <typename Traits>
template <typename I, typename T, typename B>
auto BasicTaskflow<Traits>::reduce(I beg, I end, T& result, B&& op) {
  return FlowBuilder(_nodes, num_workers()).reduce(beg, end, result, std::forward<B>(op));
}

// Function: reduce_min
// Find the minimum element over a range of items.
template <typename Traits>
template <typename I, typename T>
auto BasicTaskflow<Traits>::reduce_min(I beg, I end, T& result) {
  return reduce(beg, end, result, [] (const auto& l, const auto& r) {
    return std::min(l, r);
  });
}

// Function: reduce_max
// Find the maximum element over a range of items.
template <typename Traits>
template <typename I, typename T>
auto BasicTaskflow<Traits>::reduce_max(I beg, I end, T& result) {
  return reduce(beg, end, result, [] (const auto& l, const auto& r) {
    return std::max(l, r);
  });
}

// Function: transform_reduce    
template <typename Traits>
template <typename I, typename T, typename B, typename U>
std::pair<Task, Task> BasicTaskflow<Traits>::transform_reduce(I beg, I end, T& result, B&& bop, U&& uop) {
  return FlowBuilder(_nodes, num_workers()).
           transform_reduce(beg, end, result, std::forward<B>(bop), std::forward<U>(uop));
}

// Function: transform_reduce    
template <typename Traits>
template <typename I, typename T, typename B, typename P, typename U>
std::pair<Task, Task> BasicTaskflow<Traits>::transform_reduce(
  I beg, I end, T& result, B&& bop, P&& pop, U&& uop
) {
  return FlowBuilder(_nodes, num_workers()).
           transform_reduce(beg, end, result, std::forward<B>(bop), std::forward<P>(pop), std::forward<U>(uop));
}

// Procedure: _schedule
template <typename Traits>
void BasicTaskflow<Traits>::_schedule(Node& task) {
  _threadpool.silent_async([this, &task](){

    // Here we need to fetch the num_successors first to avoid the invalid memory
    // access caused by topology clear.
    const auto num_successors = task.num_successors();
    
    // regular task type
    if(auto index=task._work.index(); index == 0){
      //const auto num_successors = task.num_successors();
      if(auto &f = std::get<WorkType>(task._work); f != nullptr){
        f();
      }
    }
    // subflow task type 
    // The first time we enter into the subflow context, "subnodes" must be empty.
    // After executing the user's callback on subflow, there will be at least one
    // task node used as "super source". The second time we enter this context we 
    // don't have to reexecute the work again.
    else if (task._subnodes.empty()){

      assert(std::holds_alternative<SubworkType>(task._work));

      FlowBuilder fb(task._subnodes, num_workers());

      std::invoke(std::get<SubworkType>(task._work), fb);

      auto& super = task._subnodes.emplace_front([](){});

      for(auto itr = std::next(task._subnodes.begin()); itr != task._subnodes.end(); ++itr) {
        if(itr->num_successors() == 0) {
          itr->precede(task);
        }
        if(itr->dependents() == 0) {
          super.precede(*itr);
        }
      }

      super.precede(task);
      _schedule(super);

      return;
    }
    
    // At this point, the task/node storage might be destructed.
    for(size_t i=0; i<num_successors; ++i) {
      if(--(task._successors[i]->_dependents) == 0) {
        _schedule(*(task._successors[i]));
      }
    }
  });
}

// Function: dump
// Dumps the taskflow in graphviz. The result can be viewed at http://www.webgraphviz.com/.
template <typename Traits>
std::string BasicTaskflow<Traits>::dump() const {

  std::ostringstream os;

  os << "digraph Taskflow {\n";
  
  for(const auto& node : _nodes) {
    
    if(node.name().empty()) os << '\"' << &node << '\"';
    else os << std::quoted(node.name());
    os << ";\n";

    for(const auto s : node._successors) {

      if(node.name().empty()) os << '\"' << &node << '\"';
      else os << std::quoted(node.name());

      os << " -> ";
      
      if(s->name().empty()) os << '\"' << s << '\"';
      else os << std::quoted(s->name());

      os << ";\n";
    }
  }

  os << "}";
  
  return os.str();
}

//-------------------------------------------------------------------------------------------------

// Taskflow traits
struct Traits {
  using ThreadpoolType = Threadpool;
};

using Taskflow = BasicTaskflow<Traits>;

};  // end of namespace tf. -----------------------------------------------------------------------


#endif










