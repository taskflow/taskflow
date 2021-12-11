#pragma once

#include "../utility/iterator.hpp"
#include "../utility/object_pool.hpp"
#include "../utility/traits.hpp"
#include "../utility/os.hpp"
#include "../utility/math.hpp"
#include "../utility/small_vector.hpp"
#include "../utility/serializer.hpp"
#include "error.hpp"
#include "declarations.hpp"
#include "semaphore.hpp"
#include "environment.hpp"
#include "topology.hpp"

/** 
@file graph.hpp
@brief graph include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// Class: CustomGraphBase
// ----------------------------------------------------------------------------

/**
@private
*/
class CustomGraphBase {

  public:
  
  virtual void dump(std::ostream&, const void*, const std::string&) const = 0;
  virtual ~CustomGraphBase() = default;  
};

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

  private:

    std::vector<Node*> _nodes;
    
    void _clear();
    void _clear_detached();
    void _merge(Graph&&);
    void _erase(Node*);
    
    template <typename ...Args>
    Node* _emplace_back(Args&& ...); 

    Node* _emplace_back();
};

// ----------------------------------------------------------------------------

class Runtime {

  public:

  explicit Runtime(Executor& e, Worker& w) : _executor{e}, _worker{w} {
  }

  Executor& executor() { return _executor; }

  Worker& worker() { return _worker; }

  private:

  Executor& _executor;

  Worker& _worker;
};


// ----------------------------------------------------------------------------
// Node
// ----------------------------------------------------------------------------

/**
@private
*/
class Node {
  
  friend class Graph;
  friend class Task;
  friend class TaskView;
  friend class Taskflow;
  friend class Executor;
  friend class FlowBuilder;
  friend class Subflow;
  friend class Sanitizer;


  TF_ENABLE_POOLABLE_ON_THIS;

  // state bit flag
  constexpr static int CONDITIONED = 0x1;
  constexpr static int DETACHED    = 0x2;
  constexpr static int ACQUIRED    = 0x4;
  constexpr static int READY       = 0x8;
  
  // static work handle
  struct Static {

    template <typename C> 
    Static(C&&);

    std::function<void()> work;
  };

  // TODO: runtime work handle
  struct Runtime {

    template <typename C>
    Runtime(C&&);
     
    std::function<void(tf::Runtime&)> work;
  };

  // dynamic work handle
  struct Dynamic {

    template <typename C> 
    Dynamic(C&&);

    std::function<void(Subflow&)> work;
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
    Async(T&&, std::shared_ptr<AsyncTopology>);

    std::function<void(bool)> work;

    std::shared_ptr<AsyncTopology> topology;
  };
  
  // Silent async work
  struct SilentAsync {
    
    template <typename C>
    SilentAsync(C&&);

    std::function<void()> work;
  };
  
  // cudaFlow work handle
  struct cudaFlow {
    
    template <typename C, typename G> 
    cudaFlow(C&& c, G&& g);

    std::function<void(Executor&, Node*)> work;

    std::unique_ptr<CustomGraphBase> graph;
  };
  
  // syclFlow work handle
  struct syclFlow {
    
    template <typename C, typename G> 
    syclFlow(C&& c, G&& g);

    std::function<void(Executor&, Node*)> work;

    std::unique_ptr<CustomGraphBase> graph;
  };
    
  using handle_t = std::variant<
    std::monostate,  // placeholder
    Static,          // static tasking
    Dynamic,         // dynamic tasking
    Condition,       // conditional tasking
    MultiCondition,  // multi-conditional tasking
    Module,          // composable tasking
    Async,           // async tasking
    SilentAsync,     // async tasking (no future)
    cudaFlow,        // cudaFlow
    syclFlow,        // syclFlow
    Runtime          // runtime tasking
  >;
    
  struct Semaphores {  
    SmallVector<Semaphore*> to_acquire;
    SmallVector<Semaphore*> to_release;
  };

  public:
  
  // variant index
  constexpr static auto PLACEHOLDER     = get_index_v<std::monostate, handle_t>;
  constexpr static auto STATIC          = get_index_v<Static, handle_t>;
  constexpr static auto DYNAMIC         = get_index_v<Dynamic, handle_t>;
  constexpr static auto CONDITION       = get_index_v<Condition, handle_t>; 
  constexpr static auto MULTI_CONDITION = get_index_v<MultiCondition, handle_t>; 
  constexpr static auto MODULE          = get_index_v<Module, handle_t>; 
  constexpr static auto ASYNC           = get_index_v<Async, handle_t>; 
  constexpr static auto SILENT_ASYNC    = get_index_v<SilentAsync, handle_t>; 
  constexpr static auto CUDAFLOW        = get_index_v<cudaFlow, handle_t>; 
  constexpr static auto SYCLFLOW        = get_index_v<syclFlow, handle_t>; 
  constexpr static auto RUNTIME         = get_index_v<Runtime, handle_t>;

  template <typename... Args>
  Node(Args&&... args);

  ~Node();

  size_t num_successors() const;
  size_t num_dependents() const;
  size_t num_strong_dependents() const;
  size_t num_weak_dependents() const;

  const std::string& name() const;

  private:

  std::string _name;

  void* _data {nullptr};

  handle_t _handle;

  SmallVector<Node*> _successors;
  SmallVector<Node*> _dependents;

  Topology* _topology {nullptr};
  
  Node* _parent {nullptr};

  std::atomic<int> _state {0};
  std::atomic<size_t> _join_counter {0};
  
  std::unique_ptr<Semaphores> _semaphores;
  
  void _precede(Node*);
  void _set_up_join_counter();

  bool _has_state(int) const;
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
inline ObjectPool<Node> node_pool;

// ----------------------------------------------------------------------------
// Definition for Node::Static
// ----------------------------------------------------------------------------
    
// Constructor
template <typename C> 
Node::Static::Static(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::Dynamic
// ----------------------------------------------------------------------------
    
// Constructor
template <typename C> 
Node::Dynamic::Dynamic(C&& c) : work {std::forward<C>(c)} {
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
// Definition for Node::cudaFlow
// ----------------------------------------------------------------------------

template <typename C, typename G>
Node::cudaFlow::cudaFlow(C&& c, G&& g) :
  work  {std::forward<C>(c)},
  graph {std::forward<G>(g)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::syclFlow
// ----------------------------------------------------------------------------

template <typename C, typename G>
Node::syclFlow::syclFlow(C&& c, G&& g) :
  work  {std::forward<C>(c)},
  graph {std::forward<G>(g)} {
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
Node::Async::Async(C&& c, std::shared_ptr<AsyncTopology>tpg) : 
  work     {std::forward<C>(c)},
  topology {std::move(tpg)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::SilentAsync
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::SilentAsync::SilentAsync(C&& c) :
  work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::Runtime
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::Runtime::Runtime(C&& c) :
  work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node
// ----------------------------------------------------------------------------

// Constructor
template <typename... Args>
Node::Node(Args&&... args): _handle{std::forward<Args>(args)...} {
} 

// Destructor
inline Node::~Node() {
  // this is to avoid stack overflow

  if(_handle.index() == DYNAMIC) {
    // using std::get_if instead of std::get makes this compatible 
    // with older macOS versions
    // the result of std::get_if is guaranteed to be non-null 
    // due to the index check above
    auto& subgraph = std::get_if<Dynamic>(&_handle)->subgraph;
    std::vector<Node*> nodes;
    nodes.reserve(subgraph.size());

    std::move(
      subgraph._nodes.begin(), subgraph._nodes.end(), std::back_inserter(nodes)
    );
    subgraph._nodes.clear();

    size_t i = 0;

    while(i < nodes.size()) {

      if(nodes[i]->_handle.index() == DYNAMIC) {
        auto& sbg = std::get_if<Dynamic>(&(nodes[i]->_handle))->subgraph;
        std::move(
          sbg._nodes.begin(), sbg._nodes.end(), std::back_inserter(nodes)
        );
        sbg._nodes.clear();
      }

      ++i;
    }
      
    //auto& np = Graph::_node_pool();
    for(i=0; i<nodes.size(); ++i) {
      node_pool.recycle(nodes[i]);
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
inline bool Node::_is_cancelled() const {
  if(_handle.index() == Node::ASYNC) {
    auto h = std::get_if<Node::Async>(&_handle);
    if(h->topology && h->topology->_is_cancelled) {
      return true;
    }
    // async tasks spawned from subflow does not have topology
  }
  return _topology && _topology->_is_cancelled;
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
  _join_counter.store(c, std::memory_order_release);
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
    
//// Function: _node_pool
//inline ObjectPool<Node>& Graph::_node_pool() {
//  static ObjectPool<Node> pool;
//  return pool;
//}

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
inline void Graph::_clear() {
  for(auto node : _nodes) {
    node_pool.recycle(node);
  }
  _nodes.clear();
}

// Procedure: clear_detached
inline void Graph::_clear_detached() {

  auto mid = std::partition(_nodes.begin(), _nodes.end(), [] (Node* node) {
    return !(node->_state.load(std::memory_order_relaxed) & Node::DETACHED);
  });
  
  for(auto itr = mid; itr != _nodes.end(); ++itr) {
    node_pool.recycle(*itr);
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
    node_pool.recycle(node);
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
    
// Function: emplace_back
// create a node from a give argument; constructor is called if necessary
template <typename ...ArgsT>
Node* Graph::_emplace_back(ArgsT&&... args) {
  _nodes.push_back(node_pool.animate(std::forward<ArgsT>(args)...));
  return _nodes.back();
}

// Function: emplace_back
// create a node from a give argument; constructor is called if necessary
inline Node* Graph::_emplace_back() {
  _nodes.push_back(node_pool.animate());
  return _nodes.back();
}



}  // end of namespace tf. ---------------------------------------------------
