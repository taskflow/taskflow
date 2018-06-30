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

// Class: BasicTaskflow
template <typename F>
class BasicTaskflow {
  
  // Struct: Node
  struct Node {
  
    Node() = default;
  
    template <typename C>
    Node(C&&);

    const std::string& name() const;
    
    void precede(Node&);
  
    size_t num_successors() const;
    size_t dependents() const;
  
    std::string _name;
  
    F _work;
    
    std::vector<Node*> _successors;
    std::atomic<int> _dependents {0};
  };
  
  // Struct: Topology
  struct Topology{

    Topology(std::forward_list<Node>&&);

    std::forward_list<Node> nodes;
    std::shared_future<void> future;

    Node source;
    Node target;
  };

  public:
  
  // Class: Task
  class Task {

    friend class BasicTaskflow;
  
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
    auto reduce(I, I, T&, B&&, size_t);
    
    template <typename I, typename T, typename B>
    auto reduce(I, I, T&, B&&);

    template <typename I, typename T>
    auto reduce_min(I, I, T&);
    
    template <typename I, typename T>
    auto reduce_max(I, I, T&);

    template <typename I, typename T, typename B, typename U>
    auto transform_reduce(I, I, T&, B&&, U&&, size_t);

    template <typename I, typename T, typename B, typename U>
    auto transform_reduce(I, I, T&, B&&, U&&);
    
    auto placeholder();
    auto precede(Task, Task);
    auto linearize(std::vector<Task>&);
    auto linearize(std::initializer_list<Task>);
    auto broadcast(Task, std::vector<Task>&);
    auto broadcast(Task, std::initializer_list<Task>);
    auto gather(std::vector<Task>&, Task);
    auto gather(std::initializer_list<Task>, Task);
    auto dispatch();
    auto silent_dispatch();
    auto wait_for_all();

    //template<typename I, typename C>
    //auto parallel_range(const I, const I, C&&, ssize_t = 1);

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
    void _wait_for_topologies();

    template <typename L>
    void _linearize(L&);

    //template <typename I, typename O>
    //auto _reduce(I, size_t, O, size_t, Task&);
};

// Constructor
template <typename F>
template <typename C>
BasicTaskflow<F>::Node::Node(C&& c) : _work {std::forward<C>(c)} {
}

// Procedure:
template <typename F>
void BasicTaskflow<F>::Node::precede(Node& v) {
  _successors.push_back(&v);
  v._dependents++;
}

// Function: num_successors
template <typename F>
size_t BasicTaskflow<F>::Node::num_successors() const {
  return _successors.size();
}

// Function: dependents
template <typename F>
size_t BasicTaskflow<F>::Node::dependents() const {
  return _dependents.load();
}

// Function: name
template <typename F>
const std::string& BasicTaskflow<F>::Node::name() const {
  return _name;
}

//---------------------------------------------------------
// BasicTaskflow::Topology
//---------------------------------------------------------

// Constructor
template <typename F>
BasicTaskflow<F>::Topology::Topology(std::forward_list<Node>&& t) : 
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
// BasicTaskflow::Task
//---------------------------------------------------------
    
// Constructor
template <typename F>
BasicTaskflow<F>::Task::Task(const Task& rhs) : _node {rhs._node} {
}

// Function: precede
template <typename F>
typename BasicTaskflow<F>::Task& BasicTaskflow<F>::Task::precede(Task tgt) {
  _node->precede(*(tgt._node));
  return *this;
}

// Function: broadcast
template <typename F>
template <typename... Bs>
typename BasicTaskflow<F>::Task& BasicTaskflow<F>::Task::broadcast(Bs&&... tgts) {
  (_node->precede(*(tgts._node)), ...);
  return *this;
}

// Procedure: _broadcast
template <typename F>
template <typename S>
void BasicTaskflow<F>::Task::_broadcast(S& tgts) {
  for(auto& to : tgts) {
    _node->precede(*(to._node));
  }
}
      
// Function: broadcast
template <typename F>
typename BasicTaskflow<F>::Task& BasicTaskflow<F>::Task::broadcast(std::vector<Task>& tgts) {
  _broadcast(tgts);
  return *this;
}

// Function: broadcast
template <typename F>
typename BasicTaskflow<F>::Task& BasicTaskflow<F>::Task::broadcast(std::initializer_list<Task> tgts) {
  _broadcast(tgts);
  return *this;
}

// Function: gather
template <typename F>
template <typename... Bs>
typename BasicTaskflow<F>::Task& BasicTaskflow<F>::Task::gather(Bs&&... tgts) {
  (tgts->precede(*_node), ...);
  return *this;
}

// Procedure: _gather
template <typename F>
template <typename S>
void BasicTaskflow<F>::Task::_gather(S& tgts) {
  for(auto& from : tgts) {
    from._node->precede(*_node);
  }
}

// Function: gather
template <typename F>
typename BasicTaskflow<F>::Task& BasicTaskflow<F>::Task::gather(std::vector<Task>& tgts) {
  _gather(tgts);
  return *this;
}

// Function: gather
template <typename F>
typename BasicTaskflow<F>::Task& BasicTaskflow<F>::Task::gather(std::initializer_list<Task> tgts) {
  _gather(tgts);
  return *this;
}

// Operator =
template <typename F>
typename BasicTaskflow<F>::Task& BasicTaskflow<F>::Task::operator = (const Task& rhs) {
  _node = rhs._node;
  return *this;
}

// Constructor
template <typename F>
BasicTaskflow<F>::Task::Task(Task&& rhs) : _node{rhs._node} { 
  rhs._node = nullptr; 
}

// Function: work
template <typename F>
template <typename C>
typename BasicTaskflow<F>::Task& BasicTaskflow<F>::Task::work(C&& c) {
  _node->_work = std::forward<C>(c);
  return *this;
}

// Function: name
template <typename F>
typename BasicTaskflow<F>::Task& BasicTaskflow<F>::Task::name(const std::string& name) {
  _node->_name = name;
  return *this;
}

// Function: name
template <typename F>
const std::string& BasicTaskflow<F>::Task::name() const {
  return _node->_name;
}

// Function: num_dependents
template <typename F>
size_t BasicTaskflow<F>::Task::num_dependents() const {
  return _node->_dependents;
}

// Function: num_successors
template <typename F>
size_t BasicTaskflow<F>::Task::num_successors() const {
  return _node->_successors.size();
}

// Constructor
template <typename F>
BasicTaskflow<F>::Task::Task(Node* t) : _node {t} {
}

//---------------------------------------------------------
// BasicTaskflow
//---------------------------------------------------------

// Constructor
template <typename F>
BasicTaskflow<F>::BasicTaskflow() : _threadpool{std::thread::hardware_concurrency()} {
}

// Constructor
template <typename F>
BasicTaskflow<F>::BasicTaskflow(unsigned N) : _threadpool{N} {
}

// Destructor
template <typename F>
BasicTaskflow<F>::~BasicTaskflow() {
  _wait_for_topologies();
}

// Procedure: num_workers
template <typename F>
void BasicTaskflow<F>::num_workers(size_t W) {
  _threadpool.shutdown();
  _threadpool.spawn(W);
}

// Function: num_nodes
template <typename F>
size_t BasicTaskflow<F>::num_nodes() const {
  //return _nodes.size();
  return std::distance(_nodes.begin(), _nodes.end());
}

// Function: num_workers
template <typename F>
size_t BasicTaskflow<F>::num_workers() const {
  return _threadpool.num_workers();
}

// Function: num_topologies
template <typename F>
size_t BasicTaskflow<F>::num_topologies() const {
  return _topologies.size();
}

// Procedure: precede
template <typename F>
auto BasicTaskflow<F>::precede(Task from, Task to) {
  from._node->precede(*(to._node));
}

// Procedure: _linearize
template <typename F>
template <typename L>
void BasicTaskflow<F>::_linearize(L& keys) {
  std::adjacent_find(
    keys.begin(), keys.end(), 
    [] (auto& from, auto& to) {
      from._node->precede(*(to._node));
      return false;
    }
  );
}

// Procedure: linearize
template <typename F>
auto BasicTaskflow<F>::linearize(std::vector<Task>& keys) {
  _linearize(keys); 
}

// Procedure: linearize
template <typename F>
auto BasicTaskflow<F>::linearize(std::initializer_list<Task> keys) {
  _linearize(keys);
}

// Procedure: broadcast
template <typename F>
auto BasicTaskflow<F>::broadcast(Task from, std::vector<Task>& keys) {
  from.broadcast(keys);
}

// Procedure: broadcast
template <typename F>
auto BasicTaskflow<F>::broadcast(Task from, std::initializer_list<Task> keys) {
  from.broadcast(keys);
}

// Function: gather
template <typename F>
auto BasicTaskflow<F>::gather(std::vector<Task>& keys, Task to) {
  to.gather(keys);
}

// Function: gather
template <typename F>
auto BasicTaskflow<F>::gather(std::initializer_list<Task> keys, Task to) {
  to.gather(keys);
}

// Procedure: silent_dispatch 
template <typename F>
auto BasicTaskflow<F>::silent_dispatch() {

  if(_nodes.empty()) return;

  auto& topology = _topologies.emplace_front(std::move(_nodes));

  // Start the taskflow
  _schedule(topology.source);
}

// Procedure: dispatch 
template <typename F>
auto BasicTaskflow<F>::dispatch() {

  if(_nodes.empty()) {
    return std::async(std::launch::deferred, [](){}).share();
  }

  auto& topology = _topologies.emplace_front(std::move(_nodes));

  // Start the taskflow
  _schedule(topology.source);
  
  return topology.future;
}

// Procedure: wait_for_all
template <typename F>
auto BasicTaskflow<F>::wait_for_all() {
  if(!_nodes.empty()) {
    silent_dispatch();
  }
  _wait_for_topologies();
}

// Procedure: _wait_for_topologies
template <typename F>
void BasicTaskflow<F>::_wait_for_topologies() {
  for(auto& t: _topologies){
    t.future.get();
  }
  _topologies.clear();
}

// Function: placeholder
template <typename F>
auto BasicTaskflow<F>::placeholder() {
  auto& node = _nodes.emplace_front();
  return Task(&node);
}

// Function: silent_emplace
template <typename F>
template <typename C>
auto BasicTaskflow<F>::silent_emplace(C&& c) {
  auto& node = _nodes.emplace_front(std::forward<C>(c));
  return Task(&node);
}

// Function: silent_emplace
template <typename F>
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto BasicTaskflow<F>::silent_emplace(C&&... cs) {
  return std::make_tuple(silent_emplace(std::forward<C>(cs))...);
}

// Function: emplace
template <typename F>
template <typename C>
auto BasicTaskflow<F>::emplace(C&& c) {
  
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
template <typename F>
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto BasicTaskflow<F>::emplace(C&&... cs) {
  return std::make_tuple(emplace(std::forward<C>(cs))...);
}

// Function: parallel_for    
template <typename F>
template <typename I, typename C>
auto BasicTaskflow<F>::parallel_for(I beg, I end, C&& c, size_t g) {

  using category = typename std::iterator_traits<I>::iterator_category;

  if(g == 0) {
    auto d = std::distance(beg, end);
    auto w = std::max(size_t{1}, num_workers());
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
template <typename F>
template <typename T, typename C, std::enable_if_t<is_iterable_v<T>, void>*>
auto BasicTaskflow<F>::parallel_for(T& t, C&& c, size_t group) {
  return parallel_for(t.begin(), t.end(), std::forward<C>(c), group);
}

/*// Function: _reduce
// beg: begining position of this segment
// n  : size of this segment
// g  : chunk (group) size
// op : operator
// S  : source task
template <typename F>
template <typename I, typename O>
auto BasicTaskflow<F>::_reduce(I beg, size_t n, O op, size_t g, Task& S) {

  //assert(n > 0);

  // base case
  if(n <= g) {  
    auto kvp = emplace([beg, n, op] () mutable { 
      auto init = *beg++;
      for(size_t i=1; i<n; ++i, ++beg) {
        init = op(std::move(init), *beg);
      }
      return init;
    });
    S.precede(std::get<Task>(kvp));
    return kvp;
  }
  // recursion
  else {
    auto llen = n / 2;
    auto rlen = n - llen;
    auto [ltask, lfu] = _reduce(beg, llen, op, g, S);
    auto [rtask, rfu] = _reduce(std::next(beg, llen), rlen, op, g, S);
    auto kvp = emplace(
      [op, l=MoveOnCopy{std::move(lfu)}, r=MoveOnCopy{std::move(rfu)}] () {
        return op(l.object.get(), r.object.get()); 
      }
    );
    ltask.precede(std::get<Task>(kvp));
    rtask.precede(std::get<Task>(kvp));
    return kvp;
  }
} */

// Function: reduce 
template <typename F>
template <typename I, typename T, typename B>
auto BasicTaskflow<F>::reduce(I beg, I end, T& result, B&& op, size_t g) {
  
  using category = typename std::iterator_traits<I>::iterator_category;
  
  // Evenly partition
  if(g == 0) {
    auto d = std::distance(beg, end);
    auto w = std::max(size_t{1}, num_workers());
    g = (d + w - 1) / w;
  }

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

  /*if(g == 0) {
    g = 1;
  }

  if(auto n = std::distance(beg, end); n==0) {
    return std::make_pair(placeholder(), placeholder());
  }
  else {
    auto src = placeholder();
    auto [root, fu] = _reduce(beg, n, op, g, src);
    auto tgt = silent_emplace([op, fu=MoveOnCopy{std::move(fu)}, &result] () {
      result = op(std::move(result), fu.object.get());
    });  
    root.precede(tgt);
    return std::make_pair(src, tgt);
  }*/
}

// Function: reduce 
template <typename F>
template <typename I, typename T, typename B>
auto BasicTaskflow<F>::reduce(I beg, I end, T& result, B&& op) {
  return reduce(beg, end, result, std::forward<B>(op), 0);
}

// Function: reduce_min
// Find the minimum element over a range of items.
template <typename F>
template <typename I, typename T>
auto BasicTaskflow<F>::reduce_min(I beg, I end, T& result) {
  return reduce(beg, end, result, [] (const auto& l, const auto& r) {
    return std::min(l, r);
  }, 0);
}

// Function: reduce_max
// Find the maximum element over a range of items.
template <typename F>
template <typename I, typename T>
auto BasicTaskflow<F>::reduce_max(I beg, I end, T& result) {
  return reduce(beg, end, result, [] (const auto& l, const auto& r) {
    return std::max(l, r);
  }, 0);
}

// Function: transform_reduce    
template <typename F>
template <typename I, typename T, typename B, typename U>
auto BasicTaskflow<F>::transform_reduce(I beg, I end, T& result, B&& bop, U&& uop, size_t g) {

  using category = typename std::iterator_traits<I>::iterator_category;
  
  // Even partition
  if(g == 0) {
    auto d = std::distance(beg, end);
    auto w = std::max(size_t{1}, num_workers());
    g = (d + w - 1) / w;
  }

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
template <typename F>
template <typename I, typename T, typename B, typename U>
auto BasicTaskflow<F>::transform_reduce(I beg, I end, T& result, B&& bop, U&& uop) {
  return transform_reduce(beg, end, result, std::forward<B>(bop), std::forward<U>(uop), 0);
}


/*// Function: parallel_range    
template <typename F>
template <typename I, typename C>
auto BasicTaskflow<F>::parallel_range(const I beg, const I end, C&& c, ssize_t group) {

  if(group <= 0){
    group = 1;
  }

  auto source = placeholder();
  auto target = placeholder();
  
  for(auto i=beg; i<end; i+=group){
    auto b = i;
    auto e = std::min(b+group, end);
    auto task = silent_emplace([c, b, e] (){ 
      for(auto j=b;j<e; j++){
        c(j);
      }
    });
    source.precede(task);
    task.precede(target);
  }

  return std::make_pair(source, target); 
}*/


// Procedure: _schedule
template <typename F>
void BasicTaskflow<F>::_schedule(Node& task) {
  _threadpool.silent_async([this, &task](){
    // Here we need to fetch the num_successors first to avoid the invalid memory
    // access caused by topology clear.
    const auto num_successors = task.num_successors();
    if(task._work) {
      task._work();
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
template <typename F>
std::string BasicTaskflow<F>::dump() const {

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
      
      if(s->name().empty()) os << '\"' << &node << '\"';
      else os << std::quoted(s->name());

      os << ";\n";
    }
  }

  os << "}";
  
  return os.str();
}

//-------------------------------------------------------------------------------------------------

using Taskflow = BasicTaskflow<std::function<void()>>;


};  // end of namespace tf. -----------------------------------------------------------------------


#endif

