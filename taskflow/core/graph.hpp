#pragma once

#include "../utility/macros.hpp"
#include "../utility/traits.hpp"
#include "../utility/iterator.hpp"

#ifdef TF_ENABLE_TASK_POOL
#include "../utility/object_pool.hpp"
#endif

#include "../utility/os.hpp"
#include "../utility/math.hpp"
#include "../utility/small_vector.hpp"
#include "../utility/serializer.hpp"
#include "../utility/latch.hpp"
#include "../utility/mpmc.hpp"
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
class Graph : public std::vector<std::unique_ptr<Node>> {

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
  Graph(Graph&&) = default;

  /**
  @brief disabled copy assignment operator
  */
  Graph& operator = (const Graph&) = delete;

  /**
  @brief assigns a graph using move semantics
  */
  Graph& operator = (Graph&&) = default;
  

  private:

  void _erase(Node*);
  
  /**
  @private
  */
  template <typename ...ArgsT>
  Node* _emplace_back(ArgsT&&...);
};

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
  friend class AnchorGuard;
  friend class PreemptionGuard;

  //template <typename T>
  //friend class Freelist;

#ifdef TF_ENABLE_TASK_POOL
  TF_ENABLE_POOLABLE_ON_THIS;
#endif

  using Placeholder = std::monostate;

  // static work handle
  struct Static {

    template <typename C>
    Static(C&&);

    std::function<void()> work;
  };
  
  // runtime work handle
  struct Runtime {

    template <typename C>
    Runtime(C&&);

    std::function<void(tf::Runtime&)> work;
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
    
    std::function<int()> work;
  };

  // multi-condition work handle
  struct MultiCondition {

    template <typename C>
    MultiCondition(C&&);

    std::function<SmallVector<int>()> work;
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
      std::function<void()>, 
      std::function<void(tf::Runtime&)>,       // silent async
      std::function<void(tf::Runtime&, bool)>  // async
    > work;
  };
  
  // silent dependent async
  struct DependentAsync {
    
    template <typename C>
    DependentAsync(C&&);
    
    std::variant<
      std::function<void()>, 
      std::function<void(tf::Runtime&)>,       // silent async
      std::function<void(tf::Runtime&, bool)>  // async
    > work;
   
    std::atomic<size_t> use_count {1};
    std::atomic<ASTATE::underlying_type> state {ASTATE::UNFINISHED};
  };

  using handle_t = std::variant<
    Placeholder,      // placeholder
    Static,           // static tasking
    Runtime,          // runtime tasking
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
  constexpr static auto RUNTIME         = get_index_v<Runtime, handle_t>;
  constexpr static auto SUBFLOW         = get_index_v<Subflow, handle_t>;
  constexpr static auto CONDITION       = get_index_v<Condition, handle_t>;
  constexpr static auto MULTI_CONDITION = get_index_v<MultiCondition, handle_t>;
  constexpr static auto MODULE          = get_index_v<Module, handle_t>;
  constexpr static auto ASYNC           = get_index_v<Async, handle_t>;
  constexpr static auto DEPENDENT_ASYNC = get_index_v<DependentAsync, handle_t>;

  Node() = default;
  
  template <typename... Args>
  Node(const std::string&, Topology*, Node*, size_t, Args&&...);
  
  template <typename... Args>
  Node(nstate_t, estate_t, const std::string&, Topology*, Node*, size_t, Args&&...);
  
  template <typename... Args>
  Node(const TaskParams&, Topology*, Node*, size_t, Args&&...);
  
  template <typename... Args>
  Node(nstate_t, estate_t, const TaskParams&, Topology*, Node*, size_t, Args&&...);
  
  template <typename... Args>
  Node(const DefaultTaskParams&, Topology*, Node*, size_t, Args&&...);
  
  template <typename... Args>
  Node(nstate_t, estate_t, const DefaultTaskParams&, Topology*, Node*, size_t, Args&&...);

  //~Node();

  size_t num_successors() const;
  size_t num_dependents() const;
  size_t num_strong_dependents() const;
  size_t num_weak_dependents() const;

  const std::string& name() const;

  private:
  
  nstate_t _nstate              {NSTATE::NONE};
  std::atomic<estate_t> _estate {ESTATE::NONE};

  std::string _name;
  
  void* _data {nullptr};
  
  Topology* _topology {nullptr};
  Node* _parent {nullptr};

  SmallVector<Node*> _successors;
  SmallVector<Node*> _dependents;

  std::atomic<size_t> _join_counter {0};
  
  handle_t _handle;
  
  std::unique_ptr<Semaphores> _semaphores;
  
  std::exception_ptr _exception_ptr {nullptr};

  // free list
  //Node* _freelist_next{nullptr};

  bool _is_cancelled() const;
  bool _is_conditioner() const;
  bool _is_preempted() const;
  bool _acquire_all(SmallVector<Node*>&);
  void _release_all(SmallVector<Node*>&);
  void _precede(Node*);
  void _set_up_join_counter();
  void _rethrow_exception();
};

// ----------------------------------------------------------------------------
// Node Object Pool
// ----------------------------------------------------------------------------

/**
@private
*/
#ifdef TF_ENABLE_TASK_POOL
inline ObjectPool<Node> _task_pool;
#endif

/**
@private
*/
template <typename... ArgsT>
TF_FORCE_INLINE Node* animate(ArgsT&&... args) {
#ifdef TF_ENABLE_TASK_POOL
  return _task_pool.animate(std::forward<ArgsT>(args)...);
#else
  return new Node(std::forward<ArgsT>(args)...);
#endif
}

/**
@private
*/
TF_FORCE_INLINE void recycle(Node* ptr) {
#ifdef TF_ENABLE_TASK_POOL
  _task_pool.recycle(ptr);
#else
  delete ptr;
#endif
}

// ----------------------------------------------------------------------------
// Definition for Node::Static
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::Static::Static(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::Runtime
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::Runtime::Runtime(C&& c) : work {std::forward<C>(c)} {
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
  nstate_t nstate,
  estate_t estate,
  const std::string& name,
  Topology* topology, 
  Node* parent, 
  size_t join_counter,
  Args&&... args
) :
  _nstate       {nstate},
  _estate       {estate},
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
  _data         {params.data},
  _topology     {topology},
  _parent       {parent},
  _join_counter {join_counter},
  _handle       {std::forward<Args>(args)...} {
}

// Constructor
template <typename... Args>
Node::Node(
  nstate_t nstate,
  estate_t estate,
  const TaskParams& params,
  Topology* topology, 
  Node* parent, 
  size_t join_counter,
  Args&&... args
) :
  _nstate       {nstate},
  _estate       {estate},
  _name         {params.name},
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

// Constructor
template <typename... Args>
Node::Node(
  nstate_t nstate,
  estate_t estate,
  const DefaultTaskParams&,
  Topology* topology, 
  Node* parent, 
  size_t join_counter,
  Args&&... args
) :
  _nstate       {nstate},
  _estate       {estate},
  _topology     {topology},
  _parent       {parent},
  _join_counter {join_counter},
  _handle       {std::forward<Args>(args)...} {
}

// Destructor
//inline Node::~Node() {
//  // this is to avoid stack overflow
//  if(_handle.index() == SUBFLOW) {
//    auto& subgraph = std::get_if<Subflow>(&_handle)->subgraph;
//    std::vector<Node*> nodes;
//    nodes.reserve(subgraph.size());
//
//    std::move(
//      subgraph._nodes.begin(), subgraph._nodes.end(), std::back_inserter(nodes)
//    );
//    subgraph._nodes.clear();
//
//    size_t i = 0;
//
//    while(i < nodes.size()) {
//
//      if(nodes[i]->_handle.index() == SUBFLOW) {
//        auto& sbg = std::get_if<Subflow>(&(nodes[i]->_handle))->subgraph;
//        std::move(
//          sbg._nodes.begin(), sbg._nodes.end(), std::back_inserter(nodes)
//        );
//        sbg._nodes.clear();
//      }
//
//      ++i;
//    }
//
//    //auto& np = Graph::_node_pool();
//    for(i=0; i<nodes.size(); ++i) {
//      recycle(nodes[i]);
//    }
//  }
//}

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

// Function: _is_preempted
inline bool Node::_is_preempted() const {
  return _nstate & NSTATE::PREEMPTED;
}

// Function: _is_cancelled
// we currently only support cancellation of taskflow (no async task)
inline bool Node::_is_cancelled() const {
  return (_topology && (_topology->_estate.load(std::memory_order_relaxed) & ESTATE::CANCELLED)) 
         ||
         (_parent && (_parent->_estate.load(std::memory_order_relaxed) & ESTATE::CANCELLED));
}

// Procedure: _set_up_join_counter
inline void Node::_set_up_join_counter() {
  size_t c = 0;
  for(auto p : _dependents) {
    if(p->_is_conditioner()) {
      //_nstate |= NSTATE::CONDITIONED;
      _nstate = (_nstate + 1) | NSTATE::CONDITIONED;
    }
    else {
      c++;
    }
  }
  _join_counter.store(c, std::memory_order_relaxed);
}

// Procedure: _rethrow_exception
inline void Node::_rethrow_exception() {
  if(_exception_ptr) {
    auto e = _exception_ptr;
    _exception_ptr = nullptr;
    std::rethrow_exception(e);
  }
}

// Function: _acquire_all
inline bool Node::_acquire_all(SmallVector<Node*>& nodes) {
  // assert(_semaphores != nullptr);
  auto& to_acquire = _semaphores->to_acquire;
  for(size_t i = 0; i < to_acquire.size(); ++i) {
    if(!to_acquire[i]->_try_acquire_or_wait(this)) {
      for(size_t j = 1; j <= i; ++j) {
        to_acquire[i-j]->_release(nodes);
      }
      return false;
    }
  }
  return true;
}

// Function: _release_all
inline void Node::_release_all(SmallVector<Node*>& nodes) {
  // assert(_semaphores != nullptr);
  auto& to_release = _semaphores->to_release;
  for(const auto& sem : to_release) {
    sem->_release(nodes);
  }
}



// ----------------------------------------------------------------------------
// AnchorGuard
// ----------------------------------------------------------------------------

/**
@private
*/
class AnchorGuard {

  public:
  
  // anchor is at estate as it may be accessed by multiple threads (e.g., corun's
  // parent with tear_down_async's parent).
  AnchorGuard(Node* node) : _node{node} { 
    _node->_estate.fetch_or(ESTATE::ANCHORED, std::memory_order_relaxed);
  }

  ~AnchorGuard() {
    _node->_estate.fetch_and(~ESTATE::ANCHORED, std::memory_order_relaxed);
  }
  
  private:

  Node* _node;
};


// ----------------------------------------------------------------------------
// Graph definition
// ----------------------------------------------------------------------------

// Function: erase
inline void Graph::_erase(Node* node) {
  erase(
    std::remove_if(begin(), end(), [&](auto& p){ return p.get() == node; }),
    end()
  );
}

/**
@private
*/
template <typename ...ArgsT>
Node* Graph::_emplace_back(ArgsT&&... args) {
  push_back(std::make_unique<Node>(std::forward<ArgsT>(args)...));
  return back().get();
}

// ----------------------------------------------------------------------------
// Graph checker
// ----------------------------------------------------------------------------

/**
@private
 */
template <typename T, typename = void>
struct has_graph : std::false_type {};

/**
@private
 */
template <typename T>
struct has_graph<T, std::void_t<decltype(std::declval<T>().graph())>>
    : std::is_same<decltype(std::declval<T>().graph()), Graph&> {};

/**
 * @brief determines if the given type has a member function `Graph& graph()`
 *
 * This trait determines if the provided type `T` contains a member function
 * with the exact signature `tf::Graph& graph()`. It uses SFINAE and `std::void_t`
 * to detect the presence of the member function and its return type.
 *
 * @tparam T The type to inspect.
 * @retval true If the type `T` has a member function `tf::Graph& graph()`.
 * @retval false Otherwise.
 *
 * Example usage:
 * @code
 *
 * struct A {
 *   tf::Graph& graph() { return my_graph; };
 *   tf::Graph my_graph;
 *
 *   // other custom members to alter my_graph
 * };
 *
 * struct C {}; // No graph function
 *
 * static_assert(has_graph_v<A>, "A has graph()");
 * static_assert(!has_graph_v<C>, "C does not have graph()");
 * @endcode
 */
template <typename T>
constexpr bool has_graph_v = has_graph<T>::value;

// ----------------------------------------------------------------------------
// detailed helper functions
// ----------------------------------------------------------------------------

namespace detail {

/**
@private
*/
template <typename T>
TF_FORCE_INLINE Node* get_node_ptr(T& node) {
  using U = std::decay_t<T>;
  if constexpr (std::is_same_v<U, Node*>) {
    return node;
  } 
  else if constexpr (std::is_same_v<U, std::unique_ptr<Node>>) {
    return node.get();
  } 
  else {
    static_assert(dependent_false_v<T>, "Unsupported type for get_node_ptr");
  }
} 

}  // end of namespace tf::detail ---------------------------------------------


}  // end of namespace tf. ----------------------------------------------------



