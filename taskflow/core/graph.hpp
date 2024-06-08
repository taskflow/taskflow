#pragma once

#include "../utility/traits.hpp"
#include "../utility/iterator.hpp"
#include "../utility/object_pool.hpp"
#include "../utility/os.hpp"
#include "../utility/math.hpp"
#include "../utility/small_vector.hpp"
#include "../utility/serializer.hpp"
#include "error.hpp"
#include "declarations.hpp"
#include "semaphore.hpp"
#include "environment.hpp"
#include "topology.hpp"
#include "tsq.hpp"

/**
@file graph.hpp
@brief graph include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// Class: Graph
// ----------------------------------------------------------------------------


/**
@class Graph

@brief class to create a graph object

A graph is the ultimate storage for a task dependency graph and is the main
gateway to interact with an executor.
A graph manages a set of nodes in a global object pool that animates and
recycles node objects efficiently without going through repetitive and
expensive memory allocations and deallocations.
This class is mainly used for creating an opaque graph object in a custom
class to interact with the executor through taskflow composition.

A graph object is move-only.
*/
class Graph {

  friend class Node;
  friend class FlowBuilder;
  friend class Subflow;
  friend class Taskflow;
  friend class Executor;

  public:

    /**
    @brief constructs a graph object
    */
    Graph() = default;

    /**
    @brief disabled copy constructor
    */
    Graph(const Graph&) = delete;

    /**
    @brief constructs a graph using move semantics
    */
    Graph(Graph&&);

    /**
    @brief destructs the graph object
    */
    ~Graph();

    /**
    @brief disabled copy assignment operator
    */
    Graph& operator = (const Graph&) = delete;

    /**
    @brief assigns a graph using move semantics
    */
    Graph& operator = (Graph&&);

    /**
    @brief queries if the graph is empty
    */
    bool empty() const;

    /**
    @brief queries the number of nodes in the graph
    */
    size_t size() const;

    /**
    @brief clears the graph
    */
    void clear();

  private:

    std::vector<Node*> _nodes;

    void _clear();
    void _clear_detached();
    void _merge(Graph&&);
    void _erase(Node*);
    
    /**
    @private
    */
    template <typename ...ArgsT>
    Node* _emplace_back(ArgsT&&...);
};

// ----------------------------------------------------------------------------

/**
@class Runtime

@brief class to include a runtime object in a task

A runtime object allows users to interact with the
scheduling runtime inside a task, such as scheduling an active task,
spawning a subflow, and so on.

@code{.cpp}
tf::Task A, B, C, D;
std::tie(A, B, C, D) = taskflow.emplace(
  [] () { return 0; },
  [&C] (tf::Runtime& rt) {  // C must be captured by reference
    std::cout << "B\n";
    rt.schedule(C);
  },
  [] () { std::cout << "C\n"; },
  [] () { std::cout << "D\n"; }
);
A.precede(B, C, D);
executor.run(taskflow).wait();
@endcode

A runtime object is associated with the worker and the executor
that runs the task.

*/
class Runtime {

  friend class Executor;
  friend class FlowBuilder;

  public:
  
  /**
  @brief destroys the runtime object

  Issues a tf::Runtime::corun_all to finish all spawned asynchronous tasks
  and then destroys the runtime object.
  */
  ~Runtime();

  /**
  @brief obtains the running executor

  The running executor of a runtime task is the executor that runs
  the parent taskflow of that runtime task.

  @code{.cpp}
  tf::Executor executor;
  tf::Taskflow taskflow;
  taskflow.emplace([&](tf::Runtime& rt){
    assert(&(rt.executor()) == &executor);
  });
  executor.run(taskflow).wait();
  @endcode
  */
  Executor& executor();

  /**
  @brief schedules an active task immediately to the worker's queue

  @param task the given active task to schedule immediately

  This member function immediately schedules an active task to the
  task queue of the associated worker in the runtime task.
  An active task is a task in a running taskflow.
  The task may or may not be running, and scheduling that task
  will immediately put the task into the task queue of the worker
  that is running the runtime task.
  Consider the following example:

  @code{.cpp}
  tf::Task A, B, C, D;
  std::tie(A, B, C, D) = taskflow.emplace(
    [] () { return 0; },
    [&C] (tf::Runtime& rt) {  // C must be captured by reference
      std::cout << "B\n";
      rt.schedule(C);
    },
    [] () { std::cout << "C\n"; },
    [] () { std::cout << "D\n"; }
  );
  A.precede(B, C, D);
  executor.run(taskflow).wait();
  @endcode

  The executor will first run the condition task @c A which returns @c 0
  to inform the scheduler to go to the runtime task @c B.
  During the execution of @c B, it directly schedules task @c C without
  going through the normal taskflow graph scheduling process.
  At this moment, task @c C is active because its parent taskflow is running.
  When the taskflow finishes, we will see both @c B and @c C in the output.
  */
  void schedule(Task task);
  
  /**
  @brief runs the given callable asynchronously

  @tparam F callable type
  @param f callable object
    
  The method creates an asynchronous task to launch the given
  function on the given arguments.
  The difference to tf::Executor::async is that the created asynchronous task
  pertains to the runtime object.
  Applications can explicitly issue tf::Runtime::corun_all
  to wait for all spawned asynchronous tasks to finish.
  For example:

  @code{.cpp}
  std::atomic<int> counter(0);
  taskflow.emplace([&](tf::Runtime& rt){
    auto fu1 = rt.async([&](){ counter++; });
    auto fu2 = rt.async([&](){ counter++; });
    fu1.get();
    fu2.get();
    assert(counter == 2);
    
    // spawn 100 asynchronous tasks from the worker of the runtime
    for(int i=0; i<100; i++) {
      rt.async([&](){ counter++; });
    }
    
    // wait for the 100 asynchronous tasks to finish
    rt.corun_all();
    assert(counter == 102);
  });
  @endcode

  This method is thread-safe and can be called by multiple workers
  that hold the reference to the runtime.
  For example, the code below spawns 100 tasks from the worker of
  a runtime, and each of the 100 tasks spawns another task
  that will be run by another worker.
  
  @code{.cpp}
  std::atomic<int> counter(0);
  taskflow.emplace([&](tf::Runtime& rt){
    // worker of the runtime spawns 100 tasks each spawning another task
    // that will be run by another worker
    for(int i=0; i<100; i++) {
      rt.async([&](){ 
        counter++; 
        rt.async([](){ counter++; });
      });
    }
    
    // wait for the 200 asynchronous tasks to finish
    rt.corun_all();
    assert(counter == 200);
  });
  @endcode
  */
  template <typename F>
  auto async(F&& f);
  
  /**
  @brief runs the given callable asynchronously

  @tparam F callable type
  @tparam P task parameters type

  @param params task parameters
  @param f callable

  @code{.cpp}
  taskflow.emplace([&](tf::Runtime& rt){
    auto future = rt.async("my task", [](){});
    future.get();
  });
  @endcode

  */
  template <typename P, typename F>
  auto async(P&& params, F&& f);

  /**
  @brief runs the given function asynchronously without returning any future object

  @tparam F callable type
  @param f callable

  This member function is more efficient than tf::Runtime::async
  and is encouraged to use when there is no data returned.

  @code{.cpp}
  std::atomic<int> counter(0);
  taskflow.emplace([&](tf::Runtime& rt){
    for(int i=0; i<100; i++) {
      rt.silent_async([&](){ counter++; });
    }
    rt.corun_all();
    assert(counter == 100);
  });
  @endcode

  This member function is thread-safe.
  */
  template <typename F>
  void silent_async(F&& f);
  
  /**
  @brief runs the given function asynchronously without returning any future object

  @tparam F callable type
  @param params task parameters
  @param f callable
  
  @code{.cpp}
  taskflow.emplace([&](tf::Runtime& rt){
    rt.silent_async("my task", [](){});
    rt.corun_all();
  });
  @endcode
  */
  template <typename P, typename F>
  void silent_async(P&& params, F&& f);
  
  /**
  @brief similar to tf::Runtime::silent_async but the caller must be the worker of the runtime

  @tparam F callable type

  @param f callable

  The method bypass the check of the caller worker from the executor 
  and thus can only called by the worker of this runtime.

  @code{.cpp}
  taskflow.emplace([&](tf::Runtime& rt){
    // running by the worker of this runtime
    rt.silent_async_unchecked([](){});
    rt.corun_all();
  });
  @endcode
  */
  template <typename F>
  void silent_async_unchecked(F&& f);
  
  /**
  @brief similar to tf::Runtime::silent_async but the caller must be the worker of the runtime

  @tparam F callable type
  @tparam P task parameters type

  @param params task parameters
  @param f callable

  The method bypass the check of the caller worker from the executor 
  and thus can only called by the worker of this runtime.

  @code{.cpp}
  taskflow.emplace([&](tf::Runtime& rt){
    // running by the worker of this runtime
    rt.silent_async_unchecked("my task", [](){});
    rt.corun_all();
  });
  @endcode
  */
  template <typename P, typename F>
  void silent_async_unchecked(P&& params, F&& f);

  /**
  @brief co-runs the given target and waits until it completes
  
  A target can be one of the following forms:
    + a subflow task to spawn a subflow or
    + a composable graph object with `tf::Graph& T::graph()` defined

  @code{.cpp}
  // co-run a subflow and wait until all tasks complete
  taskflow.emplace([](tf::Runtime& rt){
    rt.corun([](tf::Subflow& sf){
      tf::Task A = sf.emplace([](){});
      tf::Task B = sf.emplace([](){});
    }); 
  });
  
  // co-run a taskflow and wait until all tasks complete
  tf::Taskflow taskflow1, taskflow2;
  taskflow1.emplace([](){ std::cout << "running taskflow1\n"; });
  taskflow2.emplace([&](tf::Runtime& rt){
    std::cout << "running taskflow2\n";
    rt.corun(taskflow1);
  });
  executor.run(taskflow2).wait();
  @endcode

  Although tf::Runtime::corun blocks until the operation completes, 
  the caller thread (worker) is not blocked (e.g., sleeping or holding any lock). 
  Instead, the caller thread joins the work-stealing loop of the executor 
  and returns when all tasks in the target completes.
  
  @attention
  Only the worker of this tf::Runtime can issue corun.
  */
  template <typename T>
  void corun(T&& target);

  /**
  @brief keeps running the work-stealing loop until the predicate becomes true
  
  @tparam P predicate type
  @param predicate a boolean predicate to indicate when to stop the loop

  The method keeps the caller worker running in the work-stealing loop
  until the stop predicate becomes true.
  
  @attention
  Only the worker of this tf::Runtime can issue corun.
  */
  template <typename P>
  void corun_until(P&& predicate);
  
  /**
  @brief corun all asynchronous tasks spawned by this runtime with other workers

  Coruns all asynchronous tasks (tf::Runtime::async,
  tf::Runtime::silent_async) with other workers until all those 
  asynchronous tasks finish.
    
  @code{.cpp}
  std::atomic<size_t> counter{0};
  taskflow.emplace([&](tf::Runtime& rt){
    // spawn 100 async tasks and wait
    for(int i=0; i<100; i++) {
      rt.silent_async([&](){ counter++; });
    }
    rt.corun_all();
    assert(counter == 100);
    
    // spawn another 100 async tasks and wait
    for(int i=0; i<100; i++) {
      rt.silent_async([&](){ counter++; });
    }
    rt.corun_all();
    assert(counter == 200);
  });
  @endcode

  @attention
  Only the worker of this tf::Runtime can issue tf::Runtime::corun_all.
  */
  inline void corun_all();

  /**
  @brief acquire a reference to the underlying worker
  */
  inline Worker& worker();

  protected:
  
  /**
  @private
  */
  explicit Runtime(Executor&, Worker&, Node*);
  
  /**
  @private
  */
  Executor& _executor;
  
  /**
  @private
  */
  Worker& _worker;
  
  /**
  @private
  */
  Node* _parent;

  /**
  @private
  */
  template <typename P, typename F>
  auto _async(Worker& w, P&& params, F&& f);
  
  /**
  @private
  */
  template <typename P, typename F>
  void _silent_async(Worker& w, P&& params, F&& f);
};

// constructor
inline Runtime::Runtime(Executor& e, Worker& w, Node* p) :
  _executor{e},
  _worker  {w},
  _parent  {p}{
}

// Function: executor
inline Executor& Runtime::executor() {
  return _executor;
}

// Function: worker
inline Worker& Runtime::worker() {
  return _worker;
}

// ----------------------------------------------------------------------------
// TaskParams
// ----------------------------------------------------------------------------

/**
@struct TaskParams

@brief task parameters to use when creating an asynchronous task
*/
struct TaskParams {
  /**
  @brief name of the task
  */
  std::string name;

  /**
  @brief priority of the tassk
  */
  unsigned priority {0};

  /**
  @brief C-styled pointer to user data
  */
  void* data {nullptr};
};

/**
@struct DefaultTaskParams

@brief empty task parameter type for compile-time optimization
*/
struct DefaultTaskParams {
};

/**
@brief determines if the given type is a task parameter type

Task parameters can be specified in one of the following types:
  + tf::TaskParams: assign the struct of defined parameters
  + tf::DefaultTaskParams: assign nothing
  + std::string: assign a name to the task
*/
template <typename P>
constexpr bool is_task_params_v =
  std::is_same_v<std::decay_t<P>, TaskParams> ||
  std::is_same_v<std::decay_t<P>, DefaultTaskParams> ||
  std::is_constructible_v<std::string, P>;

// ----------------------------------------------------------------------------
// Node
// ----------------------------------------------------------------------------

/**
@private
*/
class Node {

  friend class Graph;
  friend class Task;
  friend class AsyncTask;
  friend class TaskView;
  friend class Taskflow;
  friend class Executor;
  friend class FlowBuilder;
  friend class Subflow;
  friend class Runtime;

  enum class AsyncState : int {
    UNFINISHED = 0,
    LOCKED = 1,
    FINISHED = 2
  };

  TF_ENABLE_POOLABLE_ON_THIS;

  // state bit flag
  constexpr static int CONDITIONED = 1;
  constexpr static int DETACHED    = 2;
  constexpr static int ACQUIRED    = 4;
  constexpr static int READY       = 8;
  constexpr static int EXCEPTION   = 16;

  using Placeholder = std::monostate;

  // static work handle
  struct Static {

    template <typename C>
    Static(C&&);

    std::variant<
      std::function<void()>, std::function<void(Runtime&)>
    > work;
  };

  // subflow work handle
  struct Subflow {

    template <typename C>
    Subflow(C&&);

    std::function<void(tf::Subflow&)> work;
    Graph subgraph;
  };

  // condition work handle
  struct Condition {

    template <typename C>
    Condition(C&&);
    
    std::variant<
      std::function<int()>, std::function<int(Runtime&)>
    > work;
  };

  // multi-condition work handle
  struct MultiCondition {

    template <typename C>
    MultiCondition(C&&);

    std::variant<
      std::function<SmallVector<int>()>, std::function<SmallVector<int>(Runtime&)>
    > work;
  };

  // module work handle
  struct Module {

    template <typename T>
    Module(T&);

    Graph& graph;
  };

  // Async work
  struct Async {

    template <typename T>
    Async(T&&);

    std::variant<
      std::function<void()>, std::function<void(Runtime&)>
    > work;
  };
  
  // silent dependent async
  struct DependentAsync {
    
    template <typename C>
    DependentAsync(C&&);
    
    std::variant<
      std::function<void()>, std::function<void(Runtime&)>
    > work;
   
    std::atomic<size_t> use_count {1};
    std::atomic<AsyncState> state {AsyncState::UNFINISHED};
  };

  using handle_t = std::variant<
    Placeholder,      // placeholder
    Static,           // static tasking
    Subflow,          // subflow tasking
    Condition,        // conditional tasking
    MultiCondition,   // multi-conditional tasking
    Module,           // composable tasking
    Async,            // async tasking
    DependentAsync    // dependent async tasking
  >;

  struct Semaphores {
    SmallVector<Semaphore*> to_acquire;
    SmallVector<Semaphore*> to_release;
  };

  public:

  // variant index
  constexpr static auto PLACEHOLDER     = get_index_v<Placeholder, handle_t>;
  constexpr static auto STATIC          = get_index_v<Static, handle_t>;
  constexpr static auto SUBFLOW         = get_index_v<Subflow, handle_t>;
  constexpr static auto CONDITION       = get_index_v<Condition, handle_t>;
  constexpr static auto MULTI_CONDITION = get_index_v<MultiCondition, handle_t>;
  constexpr static auto MODULE          = get_index_v<Module, handle_t>;
  constexpr static auto ASYNC           = get_index_v<Async, handle_t>;
  constexpr static auto DEPENDENT_ASYNC = get_index_v<DependentAsync, handle_t>;

  Node() = default;

  template <typename... Args>
  Node(const std::string&, unsigned, Topology*, Node*, size_t, Args&&...);
  
  template <typename... Args>
  Node(const std::string&, Topology*, Node*, size_t, Args&&...);
  
  template <typename... Args>
  Node(const TaskParams&, Topology*, Node*, size_t, Args&&...);
  
  template <typename... Args>
  Node(const DefaultTaskParams&, Topology*, Node*, size_t, Args&&...);

  ~Node();

  size_t num_successors() const;
  size_t num_dependents() const;
  size_t num_strong_dependents() const;
  size_t num_weak_dependents() const;

  const std::string& name() const;

  private:

  std::string _name;
  
  unsigned _priority {0};
  
  void* _data {nullptr};
  
  Topology* _topology {nullptr};
  Node* _parent {nullptr};

  SmallVector<Node*> _successors;
  SmallVector<Node*> _dependents;

  std::atomic<int> _state {0};
  std::atomic<size_t> _join_counter {0};

  std::unique_ptr<Semaphores> _semaphores;
  std::exception_ptr _exception_ptr {nullptr};
  
  handle_t _handle;

  void _precede(Node*);
  void _set_up_join_counter();
  void _process_exception();

  bool _is_cancelled() const;
  bool _is_conditioner() const;
  bool _acquire_all(SmallVector<Node*>&);

  SmallVector<Node*> _release_all();
};

// ----------------------------------------------------------------------------
// Node Object Pool
// ----------------------------------------------------------------------------

/**
@private
*/
inline ObjectPool<Node> _task_pool;

/**
@private
*/
template <typename... ArgsT>
TF_FORCE_INLINE Node* animate(ArgsT&&... args) {
  //return new Node(std::forward<ArgsT>(args)...);
  return _task_pool.animate(std::forward<ArgsT>(args)...);
}

/**
@private
*/
TF_FORCE_INLINE void recycle(Node* ptr) {
  //delete ptr;
  _task_pool.recycle(ptr);
}

// ----------------------------------------------------------------------------
// Definition for Node::Static
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::Static::Static(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::Subflow
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::Subflow::Subflow(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::Condition
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::Condition::Condition(C&& c) : work {std::forward<C>(c)} {
}                                        

// ----------------------------------------------------------------------------
// Definition for Node::MultiCondition
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::MultiCondition::MultiCondition(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::Module
// ----------------------------------------------------------------------------

// Constructor
template <typename T>
inline Node::Module::Module(T& obj) : graph{ obj.graph() } {
}

// ----------------------------------------------------------------------------
// Definition for Node::Async
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::Async::Async(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::DependentAsync
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::DependentAsync::DependentAsync(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node
// ----------------------------------------------------------------------------

// Constructor
template <typename... Args>
Node::Node(
  const std::string& name, 
  unsigned priority,
  Topology* topology, 
  Node* parent, 
  size_t join_counter,
  Args&&... args
) :
  _name         {name},
  _priority     {priority},
  _topology     {topology},
  _parent       {parent},
  _join_counter {join_counter},
  _handle       {std::forward<Args>(args)...} {
}

// Constructor
template <typename... Args>
Node::Node(
  const std::string& name,
  Topology* topology, 
  Node* parent, 
  size_t join_counter,
  Args&&... args
) :
  _name         {name},
  _topology     {topology},
  _parent       {parent},
  _join_counter {join_counter},
  _handle       {std::forward<Args>(args)...} {
}

// Constructor
template <typename... Args>
Node::Node(
  const TaskParams& params,
  Topology* topology, 
  Node* parent, 
  size_t join_counter,
  Args&&... args
) :
  _name         {params.name},
  _priority     {params.priority},
  _data         {params.data},
  _topology     {topology},
  _parent       {parent},
  _join_counter {join_counter},
  _handle       {std::forward<Args>(args)...} {
}

// Constructor
template <typename... Args>
Node::Node(
  const DefaultTaskParams&,
  Topology* topology, 
  Node* parent, 
  size_t join_counter,
  Args&&... args
) :
  _topology     {topology},
  _parent       {parent},
  _join_counter {join_counter},
  _handle       {std::forward<Args>(args)...} {
}

// Destructor
inline Node::~Node() {
  // this is to avoid stack overflow

  if(_handle.index() == SUBFLOW) {
    // using std::get_if instead of std::get makes this compatible
    // with older macOS versions
    // the result of std::get_if is guaranteed to be non-null
    // due to the index check above
    auto& subgraph = std::get_if<Subflow>(&_handle)->subgraph;
    std::vector<Node*> nodes;
    nodes.reserve(subgraph.size());

    std::move(
      subgraph._nodes.begin(), subgraph._nodes.end(), std::back_inserter(nodes)
    );
    subgraph._nodes.clear();

    size_t i = 0;

    while(i < nodes.size()) {

      if(nodes[i]->_handle.index() == SUBFLOW) {
        auto& sbg = std::get_if<Subflow>(&(nodes[i]->_handle))->subgraph;
        std::move(
          sbg._nodes.begin(), sbg._nodes.end(), std::back_inserter(nodes)
        );
        sbg._nodes.clear();
      }

      ++i;
    }

    //auto& np = Graph::_node_pool();
    for(i=0; i<nodes.size(); ++i) {
      recycle(nodes[i]);
    }
  }
}

// Procedure: _precede
inline void Node::_precede(Node* v) {
  _successors.push_back(v);
  v->_dependents.push_back(this);
}

// Function: num_successors
inline size_t Node::num_successors() const {
  return _successors.size();
}

// Function: dependents
inline size_t Node::num_dependents() const {
  return _dependents.size();
}

// Function: num_weak_dependents
inline size_t Node::num_weak_dependents() const {
  size_t n = 0;
  for(size_t i=0; i<_dependents.size(); i++) {
    //if(_dependents[i]->_handle.index() == Node::CONDITION) {
    if(_dependents[i]->_is_conditioner()) {
      n++;
    }
  }
  return n;
}

// Function: num_strong_dependents
inline size_t Node::num_strong_dependents() const {
  size_t n = 0;
  for(size_t i=0; i<_dependents.size(); i++) {
    //if(_dependents[i]->_handle.index() != Node::CONDITION) {
    if(!_dependents[i]->_is_conditioner()) {
      n++;
    }
  }
  return n;
}

// Function: name
inline const std::string& Node::name() const {
  return _name;
}

// Function: _is_conditioner
inline bool Node::_is_conditioner() const {
  return _handle.index() == Node::CONDITION ||
         _handle.index() == Node::MULTI_CONDITION;
}

// Function: _is_cancelled
// we currently only support cancellation of taskflow (no async task)
inline bool Node::_is_cancelled() const {
  //return _topology && _topology->_is_cancelled.load(std::memory_order_relaxed);
  return _topology &&
         (_topology->_state.load(std::memory_order_relaxed) & Topology::CANCELLED);
}

// Procedure: _set_up_join_counter
inline void Node::_set_up_join_counter() {
  size_t c = 0;
  for(auto p : _dependents) {
    //if(p->_handle.index() == Node::CONDITION) {
    if(p->_is_conditioner()) {
      _state.fetch_or(Node::CONDITIONED, std::memory_order_relaxed);
    }
    else {
      c++;
    }
  }
  _join_counter.store(c, std::memory_order_relaxed);
}

// Procedure: _process_exception
inline void Node::_process_exception() {
  if(_exception_ptr) {
    auto e = _exception_ptr;
    _exception_ptr = nullptr;
    std::rethrow_exception(e);
  }
}

// Function: _acquire_all
inline bool Node::_acquire_all(SmallVector<Node*>& nodes) {

  auto& to_acquire = _semaphores->to_acquire;

  for(size_t i = 0; i < to_acquire.size(); ++i) {
    if(!to_acquire[i]->_try_acquire_or_wait(this)) {
      for(size_t j = 1; j <= i; ++j) {
        auto r = to_acquire[i-j]->_release();
        nodes.insert(std::end(nodes), std::begin(r), std::end(r));
      }
      return false;
    }
  }
  return true;
}

// Function: _release_all
inline SmallVector<Node*> Node::_release_all() {

  auto& to_release = _semaphores->to_release;

  SmallVector<Node*> nodes;
  for(const auto& sem : to_release) {
    auto r = sem->_release();
    nodes.insert(std::end(nodes), std::begin(r), std::end(r));
  }

  return nodes;
}

// ----------------------------------------------------------------------------
// Graph definition
// ----------------------------------------------------------------------------

// Destructor
inline Graph::~Graph() {
  _clear();
}

// Move constructor
inline Graph::Graph(Graph&& other) :
  _nodes {std::move(other._nodes)} {
}

// Move assignment
inline Graph& Graph::operator = (Graph&& other) {
  _clear();
  _nodes = std::move(other._nodes);
  return *this;
}

// Procedure: clear
inline void Graph::clear() {
  _clear();
}

// Procedure: clear
inline void Graph::_clear() {
  for(auto node : _nodes) {
    recycle(node);
  }
  _nodes.clear();
}

// Procedure: clear_detached
inline void Graph::_clear_detached() {

  auto mid = std::partition(_nodes.begin(), _nodes.end(), [] (Node* node) {
    return !(node->_state.load(std::memory_order_relaxed) & Node::DETACHED);
  });

  for(auto itr = mid; itr != _nodes.end(); ++itr) {
    recycle(*itr);
  }
  _nodes.resize(std::distance(_nodes.begin(), mid));
}

// Procedure: merge
inline void Graph::_merge(Graph&& g) {
  for(auto n : g._nodes) {
    _nodes.push_back(n);
  }
  g._nodes.clear();
}

// Function: erase
inline void Graph::_erase(Node* node) {
  if(auto I = std::find(_nodes.begin(), _nodes.end(), node); I != _nodes.end()) {
    _nodes.erase(I);
    recycle(node);
  }
}

// Function: size
inline size_t Graph::size() const {
  return _nodes.size();
}

// Function: empty
inline bool Graph::empty() const {
  return _nodes.empty();
}

/**
@private
*/
template <typename ...ArgsT>
Node* Graph::_emplace_back(ArgsT&&... args) {
  _nodes.push_back(animate(std::forward<ArgsT>(args)...));
  return _nodes.back();
}


}  // end of namespace tf. ---------------------------------------------------










