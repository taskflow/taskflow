// MIT License
// 
// Copyright (c) 2018 Dr. Tsung-Wei Huang, Chun-Xun Lin, and Martin Wong
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

namespace tf {

template <typename... ArgsT>
inline void __throw__(const char* fname, const size_t line, ArgsT&&... args) {
  std::ostringstream oss;
  oss << "[" << fname << ":" << line << "] ";
  (oss << ... << std::forward<ArgsT>(args));
  throw std::runtime_error(oss.str());
}

#define TF_THROW(...) __throw__(__FILE__, __LINE__, __VA_ARGS__);

//-------------------------------------------------------------------------------------------------
// Threadpool definition
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

// ------------------------------------------------------------------------------------------------

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

//-------------------------------------------------------------------------------------------------
// Taskflow definition
//-------------------------------------------------------------------------------------------------

// Class: BasicTaskflow
template <typename F>
class BasicTaskflow {
  
  template <typename G>
  friend std::ostream& operator << (std::ostream&, const BasicTaskflow<G>&);
  
  // Struct: Node
  struct Node {
  
    friend class BasicTaskflow;
    friend class Topology;
    friend class Task;
  
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

    Topology(std::list<Node>&&);

    std::list<Node> nodes;
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

    auto placeholder();
    auto dispatch();
    auto silent_dispatch();
    auto precede(Task, Task);
    auto linearize(std::vector<Task>&);
    auto linearize(std::initializer_list<Task>);
    auto broadcast(Task, std::vector<Task>&);
    auto broadcast(Task, std::initializer_list<Task>);
    auto gather(std::vector<Task>&, Task);
    auto gather(std::initializer_list<Task>, Task);
    auto wait_for_all();

    template<typename I, class C>
    auto parallel_for(I, I, C&&);

    size_t num_nodes() const;
    size_t num_workers() const;
    size_t num_topologies() const;

    std::string dump() const;

  private:

    Threadpool _threadpool;

    std::list<Node> _nodes;
    std::list<Topology> _topologies;

    void _schedule(Node&);
    void _wait_for_topologies();

    template <typename L>
    void _linearize(L&);
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
BasicTaskflow<F>::Topology::Topology(std::list<Node>&& t) : 
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

// Function: broadcast
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

// Function: num_nodes
template <typename F>
size_t BasicTaskflow<F>::num_nodes() const {
  return _nodes.size();
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

  auto& topology = _topologies.emplace_back(std::move(_nodes));

  // Start the taskflow
  _schedule(topology.source);
}

// Procedure: dispatch 
template <typename F>
auto BasicTaskflow<F>::dispatch() {

  if(_nodes.empty()) {
    return std::async(std::launch::deferred, [](){}).share();
  }

  //_topologies.remove_if([](auto &t){ 
  //   auto status = t.future.wait_for(std::chrono::seconds(0));
  //   if(status == std::future_status::ready){
  //     return true;
  //   }
  //  return false; 
  //});

  auto& topology = _topologies.emplace_back(std::move(_nodes));

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
  auto& node = _nodes.emplace_back();
  return Task(&node);
}

// Function: silent_emplace
template <typename F>
template <typename C>
auto BasicTaskflow<F>::silent_emplace(C&& c) {
  auto& node = _nodes.emplace_back(std::forward<C>(c));
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

  auto& node = _nodes.emplace_back([p=MoveOnCopy(std::move(p)), c=std::forward<C>(c)] () mutable { 
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
template <typename I, class C>
auto BasicTaskflow<F>::parallel_for(I beg, I end, C&& c) {

  auto source = placeholder();
  auto target = placeholder();

  for(; beg != end; ++beg) {
    auto task = silent_emplace([&, itr=beg] (){ c(*itr); });
    source.precede(task);
    task.precede(target);
  }

  return std::make_pair(source, target); 
}

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

    os << "  \"";
    if(!node.name().empty()) os << node.name();
    else os << &node;
    os << "\";\n";

    for(const auto s : node._successors) {
      os << "  \"";
      if(!node.name().empty()) os << node.name();
      else os << &node;
      os << "\" -> \"";
      if(s->name() != "") os << s->name();
      else os << s;
      os << "\";\n";  
    }
  }

  os << "}\n";
  
  return os.str();
}

//-------------------------------------------------------------------------------------------------

using Taskflow = BasicTaskflow<std::function<void()>>;


};  // end of namespace tf. -----------------------------------------------------------------------


#endif


