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
#include "../utility/lazy_string.hpp"
#include "../utility/list.hpp"
#include "error.hpp"
#include "declarations.hpp"
#include "semaphore.hpp"
#include "environment.hpp"
#include "topology.hpp"
#include "wsq.hpp"


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
  
  // --- Iterator Subclass ---
  class Iterator {

    public:
    using iterator_category = std::forward_iterator_tag;

    explicit Iterator(Node* ptr) : _ptr(ptr) {}

    auto operator*() const { return _ptr; }
    auto operator*() { return _ptr; }
    //pointer operator->() { return _ptr; }

    Iterator& operator++();
    Iterator operator++(int);

    bool operator==(const Iterator& other) const { return _ptr == other._ptr; }
    bool operator!=(const Iterator& other) const { return _ptr != other._ptr; }

  private:
    Node* _ptr;
  };

  /**
  @brief constructs the graph object
  */
  Graph() = default;

  /**
  @brief destroys the graph object
  */
  ~Graph();

  /**
  @brief disabled copy constructor
  */
  Graph(const Graph&) = delete;

  /**
  @brief constructs a graph using move semantics
  */
  Graph(Graph&&);

  /**
  @brief disabled copy assignment operator
  */
  Graph& operator = (const Graph&) = delete;

  /**
  @brief assigns a graph using move semantics
  */
  Graph& operator = (Graph&&);

  /**
  @brief clears the graph
  */
  void clear();

  /**
  @brief returns the number of nodes in the graph
  */
  //size_t size() const;
  
  /**
  @brief queries the emptiness of the graph
  */
  bool empty() const;
  
  /**
  @brief returns an iterator to the first node of this graph
  */
  auto begin();

  /**
  @brief returns an iterator past the last element of this graph
  */
  auto end();

  /**
  @brief returns an iterator to the first node of this graph
  */
  auto begin() const;
  
  /**
  @brief returns an iterator past the last element of this graph
  */
  auto end() const;



  private:

  //std::vector<Node*> _nodes;

  Node* _head {nullptr};

  void _erase(Node*);

  size_t _set_up(Topology*, NodeBase*);
  
  /**
  @private
  */
  template <typename ...ArgsT>
  Node* _emplace(ArgsT&&...);
};

// ----------------------------------------------------------------------------
// TaskParams
// ----------------------------------------------------------------------------

/**
@class TaskParams

@brief class to create a task parameter object 

tf::TaskParams is primarily used by asynchronous tasking.
*/
class TaskParams {

  public:

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
@class DefaultTaskParams

@brief class to create an empty task parameter for compile-time optimization
*/
class DefaultTaskParams {};

/**
@brief determines if the given type is a task parameter type

Task parameters can be specified in one of the following types:
  + tf::TaskParams
  + tf::DefaultTaskParams
  + std::string
*/
template <typename P>
constexpr bool is_task_params_v =
  std::is_same_v<std::decay_t<P>, TaskParams> ||
  std::is_same_v<std::decay_t<P>, DefaultTaskParams> ||
  std::is_constructible_v<std::string, P>;

// ----------------------------------------------------------------------------
// NodeBase
// ----------------------------------------------------------------------------

/**
@private
*/
class NodeBase {

  friend class Node;
  friend class Graph;
  friend class Task;
  friend class AsyncTask;
  friend class TaskView;
  friend class Taskflow;
  friend class Executor;
  friend class FlowBuilder;
  friend class Subflow;
  friend class Runtime;
  friend class NonpreemptiveRuntime;
  friend class ExplicitAnchorGuard;
  friend class TaskGroup;
  friend class Algorithm;
  
  protected:
  
  nstate_t _nstate              {NSTATE::NONE};
  std::atomic<estate_t> _estate {ESTATE::NONE};

  NodeBase* _parent {nullptr};
  std::atomic<size_t> _join_counter {0};
  
  std::exception_ptr _exception_ptr {nullptr};

  NodeBase() = default;

  NodeBase(nstate_t nstate, estate_t estate, NodeBase* parent, size_t join_counter) :
    _nstate {nstate}, 
    _estate {estate},
    _parent {parent},
    _join_counter {join_counter} {
  }
  
  void _rethrow_exception() {
    if(_exception_ptr) {
      auto e = _exception_ptr;
      _exception_ptr = nullptr;
      _estate.fetch_and(~(ESTATE::EXCEPTION | ESTATE::CAUGHT), std::memory_order_relaxed);
      std::rethrow_exception(e);
    }
  }
};

// ----------------------------------------------------------------------------
// Topology
// ----------------------------------------------------------------------------

/**
@private
*/
class Topology : public NodeBase {

  friend class Executor;
  friend class Subflow;
  friend class Runtime;
  friend class NonpreemptiveRuntime;
  friend class Node;

  template <typename T>
  friend class Future;
  
  public:

  template <typename Predicate, typename OnFinish>
  Topology(Taskflow&, Predicate&&, OnFinish&&);

  bool cancelled() const;

  private:

  Taskflow& _taskflow;

  std::promise<void> _promise;

  std::function<bool()> _predicate;
  std::function<void()> _on_finish;
  
  void _carry_out_promise();
};

// Constructor
template <typename Predicate, typename OnFinish>
Topology::Topology(Taskflow& tf, Predicate&& predicate, OnFinish&& on_finish):
  NodeBase(NSTATE::NONE, ESTATE::EXPLICITLY_ANCHORED, nullptr, 0),
  _taskflow(tf),
  _predicate(std::forward<Predicate>(predicate)),
  _on_finish(std::forward<OnFinish> (on_finish)) {
}

// Procedure
inline void Topology::_carry_out_promise() {
  if(_exception_ptr) {
    auto e = _exception_ptr;
    _exception_ptr = nullptr;
    _promise.set_exception(e);
  }
  else {
    _promise.set_value();
  }
}

// Function: cancelled
inline bool Topology::cancelled() const {
  return _estate.load(std::memory_order_relaxed) & (ESTATE::CANCELLED | ESTATE::EXCEPTION);
}


// ----------------------------------------------------------------------------
// Node
// ----------------------------------------------------------------------------

/**
@private
*/
class Node : public NodeBase {

  friend class Graph;
  friend class Task;
  friend class AsyncTask;
  friend class TaskView;
  friend class Taskflow;
  friend class Executor;
  friend class FlowBuilder;
  friend class Subflow;
  friend class Runtime;
  friend class NonpreemptiveRuntime;
  friend class ExplicitAnchorGuard;
  friend class TaskGroup;
  friend class Algorithm;

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
  
  struct NonpreemptiveRuntime {
    
    template <typename C>
    NonpreemptiveRuntime(C&&);

    std::function<void(tf::NonpreemptiveRuntime&)> work;
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
  };

  using handle_t = std::variant<
    Placeholder,          // placeholder
    Static,               // static tasking
    Runtime,              // runtime tasking
    NonpreemptiveRuntime, // runtime (non-preemptive) tasking
    Subflow,              // subflow tasking
    Condition,            // conditional tasking
    MultiCondition,       // multi-conditional tasking
    Module,               // composable tasking
    Async,                // async tasking
    DependentAsync        // dependent async tasking
  >;

  struct Semaphores {
    SmallVector<Semaphore*> to_acquire;
    SmallVector<Semaphore*> to_release;
  };

  public:

  // variant index
  constexpr static auto PLACEHOLDER           = get_index_v<Placeholder, handle_t>;
  constexpr static auto STATIC                = get_index_v<Static, handle_t>;
  constexpr static auto RUNTIME               = get_index_v<Runtime, handle_t>;
  constexpr static auto NONPREEMPTIVE_RUNTIME = get_index_v<NonpreemptiveRuntime, handle_t>;
  constexpr static auto SUBFLOW               = get_index_v<Subflow, handle_t>;
  constexpr static auto CONDITION             = get_index_v<Condition, handle_t>;
  constexpr static auto MULTI_CONDITION       = get_index_v<MultiCondition, handle_t>;
  constexpr static auto MODULE                = get_index_v<Module, handle_t>;
  constexpr static auto ASYNC                 = get_index_v<Async, handle_t>;
  constexpr static auto DEPENDENT_ASYNC       = get_index_v<DependentAsync, handle_t>;

  Node() = default;
  
  template <typename... Args>
  Node(nstate_t, estate_t, const TaskParams&, Topology*, NodeBase*, size_t, Args&&...);
  
  template <typename... Args>
  Node(nstate_t, estate_t, const DefaultTaskParams&, Topology*, NodeBase*, size_t, Args&&...);

  size_t num_successors() const;
  size_t num_predecessors() const;
  size_t num_strong_dependencies() const;
  size_t num_weak_dependencies() const;

  const std::string& name() const;
  
  private:
  
  std::string _name;
  
  void* _data {nullptr};
  
  Topology* _topology {nullptr};

  handle_t _handle;
  
  size_t _num_successors {0};
  SmallVector<Node*, 4> _edges;
  
  std::unique_ptr<Semaphores> _semaphores;

  Node* next {nullptr};

  bool _is_parent_cancelled() const;
  bool _is_conditioner() const;
  bool _acquire_all(SmallVector<Node*>&);
  void _release_all(SmallVector<Node*>&);
  void _precede(Node*);
  void _set_up_join_counter();

  void _remove_successors(Node*);
  void _remove_predecessors(Node*);
};


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

// Constructor
template <typename C>
Node::NonpreemptiveRuntime::NonpreemptiveRuntime(C&& c) : work {std::forward<C>(c)} {
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
  nstate_t nstate,
  estate_t estate,
  const TaskParams& params,
  Topology* topology, 
  NodeBase* parent, 
  size_t join_counter,
  Args&&... args
) :
  NodeBase(nstate, estate, parent, join_counter),
  _name     {params.name},
  _data     {params.data},
  _topology {topology},
  _handle   {std::forward<Args>(args)...} {
}

// Constructor
template <typename... Args>
Node::Node(
  nstate_t nstate,
  estate_t estate,
  const DefaultTaskParams&,
  Topology* topology, 
  NodeBase* parent, 
  size_t join_counter,
  Args&&... args
) :
  NodeBase(nstate, estate, parent, join_counter),
  _topology {topology},
  _handle   {std::forward<Args>(args)...} {
}

//// Constructor
//template <typename T, typename... Args>
//void Node::reset(
//  nstate_t nstate,
//  estate_t estate,
//  const TaskParams& params,
//  Topology* topology, 
//  NodeBase* parent, 
//  size_t join_counter,
//  std::in_place_type_t<T>,
//  Args&&... args
//) {
//  _nstate   = nstate;
//  _estate   = estate;
//  _parent   = parent;
//  _join_counter.store(join_counter, std::memory_order_relaxed);
//  _exception_ptr = nullptr;
//  _name     = params.name;
//  _data     = params.data;
//  _topology = topology;
//  _handle.emplace<T>(std::forward<Args>(args)...);
//  _num_successors = 0;
//  _edges.clear();
//  _semaphores.reset();
//}
//
//// Constructor
//template <typename T, typename... Args>
//void Node::reset(
//  nstate_t nstate,
//  estate_t estate,
//  const DefaultTaskParams&,
//  Topology* topology, 
//  NodeBase* parent, 
//  size_t join_counter,
//  std::in_place_type_t<T>,
//  Args&&... args
//) {
//  _nstate = nstate;
//  _estate = estate;
//  _parent = parent;
//  _join_counter.store(join_counter, std::memory_order_relaxed);
//  _exception_ptr = nullptr;
//  _name.clear();
//  _data = nullptr;
//  _topology = topology;
//  _handle.emplace<T>(std::forward<Args>(args)...);
//  _num_successors = 0;
//  _edges.clear();
//  _semaphores.reset();
//}

// Procedure: _precede
/*
u successor   layout: s1, s2, s3, p1, p2 (num_successors = 3)
v predecessor layout: s1, p1, p2

add a new successor: u->v
u successor   layout: 
  s1, s2, s3, p1, p2, v (push_back v)
  s1, s2, s3, v, p2, p1 (swap adj[num_successors] with adj[n-1])
v predecessor layout: 
  s1, p1, p2, u         (push_back u)
*/ 
inline void Node::_precede(Node* v) {
  _edges.push_back(v);
  std::swap(_edges[_num_successors++], _edges[_edges.size() - 1]);
  v->_edges.push_back(this);
}

// Function: _remove_successors
inline void Node::_remove_successors(Node* node) {
  auto sit = std::remove(_edges.begin(), _edges.begin() + _num_successors, node);
  size_t new_num_successors = std::distance(_edges.begin(), sit);
  std::move(_edges.begin() + _num_successors, _edges.end(), sit);
  _edges.resize(_edges.size() - (_num_successors - new_num_successors));
  _num_successors = new_num_successors;
}

// Function: _remove_predecessors
inline void Node::_remove_predecessors(Node* node) {
  _edges.erase( 
    std::remove(_edges.begin() + _num_successors, _edges.end(), node), _edges.end()
  );
}

// Function: num_successors
inline size_t Node::num_successors() const {
  return _num_successors;
}

// Function: predecessors
inline size_t Node::num_predecessors() const {
  return _edges.size() - _num_successors;
}

// Function: num_weak_dependencies
inline size_t Node::num_weak_dependencies() const {
  size_t n = 0;
  for(size_t i=_num_successors; i<_edges.size(); i++) {
    n += _edges[i]->_is_conditioner();
  }
  return n;
}

// Function: num_strong_dependencies
inline size_t Node::num_strong_dependencies() const {
  size_t n = 0;
  for(size_t i=_num_successors; i<_edges.size(); i++) {
    n += !_edges[i]->_is_conditioner();
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

// Function: _is_parent_cancelled
inline bool Node::_is_parent_cancelled() const {
  return (_topology && (_topology->_estate.load(std::memory_order_relaxed) & (ESTATE::CANCELLED | ESTATE::EXCEPTION))) 
         ||
         (_parent && (_parent->_estate.load(std::memory_order_relaxed) & (ESTATE::CANCELLED | ESTATE::EXCEPTION)));
}

// Procedure: _set_up_join_counter
inline void Node::_set_up_join_counter() {
  size_t c = 0;
  //for(auto p : _predecessors) {
  for(size_t i=_num_successors; i<_edges.size(); i++) {
    bool is_cond = _edges[i]->_is_conditioner();
    _nstate = (_nstate + is_cond) | (is_cond * NSTATE::CONDITIONED);  // weak dependency
    c += !is_cond;  // strong dependency
  }
  _join_counter.store(c, std::memory_order_relaxed);
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
// ExplicitAnchorGuard
// ----------------------------------------------------------------------------

/**
@private
*/
class ExplicitAnchorGuard {

  public:
  
  // Explicit anchor must sit in estate as it may be accessed by multiple threads 
  // (e.g., corun's parent with tear_down_async's parent).
  ExplicitAnchorGuard(NodeBase* node_base) : _node_base{node_base} { 
    _node_base->_estate.fetch_or(ESTATE::EXPLICITLY_ANCHORED, std::memory_order_relaxed);
  }

  ~ExplicitAnchorGuard() {
    _node_base->_estate.fetch_and(~ESTATE::EXPLICITLY_ANCHORED, std::memory_order_relaxed);
  }
  
  private:

  NodeBase* _node_base;
};

// ----------------------------------------------------------------------------
// Node Object Pool
// ----------------------------------------------------------------------------

/**
@private
*/
#ifdef TF_ENABLE_TASK_POOL
inline ObjectPool<Node> _node_pool;
#endif

/**
@private
*/
template <typename... ArgsT>
TF_FORCE_INLINE Node* animate(ArgsT&&... args) {
#ifdef TF_ENABLE_TASK_POOL
  return _node_pool.animate(std::forward<ArgsT>(args)...);
#else
  return new Node(std::forward<ArgsT>(args)...);
#endif
}

/**
@private
*/
TF_FORCE_INLINE void recycle(Node* ptr) {
#ifdef TF_ENABLE_TASK_POOL
  _node_pool.recycle(ptr);
#else
  delete ptr;
#endif
}


// ----------------------------------------------------------------------------
// Graph definition
// ----------------------------------------------------------------------------
    
inline Graph::Iterator& Graph::Iterator::operator++() {
  if (_ptr) _ptr = _ptr->next;
  return *this;
}

inline Graph::Iterator Graph::Iterator::operator++(int) {
  Iterator tmp = *this;
  ++(*this);
  return tmp;
}

// Destructor
inline Graph::~Graph() {
  clear();
}

// Move constructor
inline Graph::Graph(Graph&& other) :
  _head {other._head} {
  other._head = nullptr;
}

// Move assignment
inline Graph& Graph::operator = (Graph&& other) {
  clear();
  _head = other._head;
  other._head = nullptr;
  return *this;
}

// Procedure: clear
inline void Graph::clear() {
  while(_head) {
    auto old_head = _head;
    _head = _head->next;
    recycle(old_head);
  }
}

// Function: size
//inline size_t Graph::size() const {
//  size_t count = 0;
//  auto curr = _head;
//  while (curr) {
//    count++;
//    curr = curr->next;
//  }
//  return count;
//}

// Function: empty
inline bool Graph::empty() const {
  return _head == nullptr;
}

// Function: begin
inline auto Graph::begin() {
  return Iterator(_head);
}

// Function: end
inline auto Graph::end() {
  return Iterator(nullptr);
}

// Function: begin
inline auto Graph::begin() const {
  return Iterator(_head);
}

// Function: end
inline auto Graph::end() const {
  return Iterator(nullptr);
}

// Function: erase
inline void Graph::_erase(Node* tgt) {

  if (!_head || !tgt) {
    return;
  }

  // Case: Target is the head
  if (_head == tgt) {
    _head = _head->next;
    //tgt->next = nullptr;
    recycle(tgt);
    return;
  }

  // Case: Search for predecessor
  auto current = _head;
  while (current->next && current->next != tgt) {
    current = current->next;
  }

  // Bridge the gap
  if (current->next == tgt) {
    current->next = tgt->next;
    //tgt->next = nullptr;
    recycle(tgt);
  }  
}

// Function: _set_up
inline size_t Graph::_set_up(Topology* tpg, NodeBase* parent) {

  if (!_head) {
    return 0;
  }
  
  size_t num_srcs = 0;

  Node* zeroHead = nullptr; // Head of the "ready" list
  Node* zeroTail = nullptr; // Tail of the "ready" list
  Node* restHead = nullptr; // Head of the "busy" list
  Node* restTail = nullptr; // Tail of the "busy" list

  Node* current = _head;

  while (current) {

    current->_topology = tpg;
    current->_parent = parent;
    current->_nstate = NSTATE::NONE;
    current->_estate.store(ESTATE::NONE, std::memory_order_relaxed);
    current->_set_up_join_counter();
    current->_exception_ptr = nullptr;

    auto nextNode = current->next; // Save next pointer before re-linking
    current->next = nullptr;     // Disconnect the current node

    if (current->num_predecessors() == 0) {
      ++num_srcs;
      // Append to the "zero" chain
      if (!zeroHead) {
        zeroHead = zeroTail = current;
      } else {
        zeroTail->next = current;
        zeroTail = current;
      }
    } else {
      // Append to the "rest" chain
      if (!restHead) {
        restHead = restTail = current;
      } else {
        restTail->next = current;
        restTail = current;
      }
    }
    current = nextNode;
  }

  // Stitch the two chains together
  if (zeroHead) {
    _head = zeroHead;
    zeroTail->next = restHead; // Connect end of zeros to start of rest
  } else {
    _head = restHead; // No zeros found, head is just the rest
  }
  
  return num_srcs;
}

/**
@private
*/
template <typename ...ArgsT>
Node* Graph::_emplace(ArgsT&&... args) {
  //_nodes.push_back(animate(std::forward<ArgsT>(args)...));
  //return _nodes.back();
  auto node = animate(std::forward<ArgsT>(args)...); 
  node->next = _head;
  _head = node;
  return _head;
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


}  // end of namespace tf. ----------------------------------------------------



