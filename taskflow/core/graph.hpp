#pragma once

#include "../utility/iterator.hpp"
#include "../utility/object_pool.hpp"
#include "../utility/traits.hpp"
#include "../utility/singleton.hpp"
#include "../utility/os.hpp"
#include "../utility/math.hpp"
#include "../utility/serializer.hpp"
#include "error.hpp"
#include "declarations.hpp"
#include "semaphore.hpp"
#include "environment.hpp"
#include "topology.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// Class: CustomGraphBase
// ----------------------------------------------------------------------------
class CustomGraphBase {

  public:
  
  virtual void dump(std::ostream&, const void*, const std::string&) const = 0;
  virtual ~CustomGraphBase() = default;  
};

// ----------------------------------------------------------------------------
// Class: Graph
// ----------------------------------------------------------------------------
class Graph {

  friend class Node;
  friend class Taskflow;
  friend class Executor;

  public:

    Graph() = default;
    Graph(const Graph&) = delete;
    Graph(Graph&&);

    ~Graph();

    Graph& operator = (const Graph&) = delete;
    Graph& operator = (Graph&&);
    
    void clear();
    void clear_detached();
    void merge(Graph&&);

    bool empty() const;

    size_t size() const;
    
    template <typename ...Args>
    Node* emplace_back(Args&& ...); 

    Node* emplace_back();

  private:

    std::vector<Node*> _nodes;
};

// ----------------------------------------------------------------------------

// Class: Node
class Node {
  
  friend class Graph;
  friend class Task;
  friend class TaskView;
  friend class Taskflow;
  friend class Executor;
  friend class FlowBuilder;
  friend class Subflow;

  TF_ENABLE_POOLABLE_ON_THIS;

  // state bit flag
  constexpr static int BRANCHED = 0x1;
  constexpr static int DETACHED = 0x2;
  constexpr static int ACQUIRED = 0x4;
  
  // static work handle
  struct Static {

    template <typename C> 
    Static(C&&);

    std::function<void()> work;
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

  // module work handle
  struct Module {

    template <typename T>
    Module(T&&);

    Taskflow* module {nullptr};
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
    Module,          // composable tasking
    Async,           // async tasking
    SilentAsync,     // async tasking (no future)
    cudaFlow,        // cudaFlow
    syclFlow         // syclFlow
  >;
    
  struct Semaphores {  
    std::vector<Semaphore*> to_acquire;
    std::vector<Semaphore*> to_release;
  };

  public:
  
  // variant index
  constexpr static auto PLACEHOLDER  = get_index_v<std::monostate, handle_t>;
  constexpr static auto STATIC       = get_index_v<Static, handle_t>;
  constexpr static auto DYNAMIC      = get_index_v<Dynamic, handle_t>;
  constexpr static auto CONDITION    = get_index_v<Condition, handle_t>; 
  constexpr static auto MODULE       = get_index_v<Module, handle_t>; 
  constexpr static auto ASYNC        = get_index_v<Async, handle_t>; 
  constexpr static auto SILENT_ASYNC = get_index_v<SilentAsync, handle_t>; 
  constexpr static auto CUDAFLOW     = get_index_v<cudaFlow, handle_t>; 
  constexpr static auto SYCLFLOW     = get_index_v<syclFlow, handle_t>; 

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

    handle_t _handle;

    std::vector<Node*> _successors;
    std::vector<Node*> _dependents;

    //std::optional<Semaphores> _semaphores;
    std::unique_ptr<Semaphores> _semaphores;

    Topology* _topology {nullptr};
    
    Node* _parent {nullptr};

    int _state {0};

    std::atomic<size_t> _join_counter {0};
    
    void _precede(Node*);
    void _set_state(int);
    void _unset_state(int);
    void _clear_state();
    void _set_up_join_counter();

    bool _has_state(int) const;
    bool _is_cancelled() const;
    bool _acquire_all(std::vector<Node*>&);

    std::vector<Node*> _release_all();
};

// ----------------------------------------------------------------------------
// Node Object Pool
// ----------------------------------------------------------------------------
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
Node::Module::Module(T&& tf) : module {tf} {
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

    auto& subgraph = std::get<Dynamic>(_handle).subgraph;

    std::vector<Node*> nodes;

    std::move(
      subgraph._nodes.begin(), subgraph._nodes.end(), std::back_inserter(nodes)
    );
    subgraph._nodes.clear();

    size_t i = 0;

    while(i < nodes.size()) {

      if(nodes[i]->_handle.index() == DYNAMIC) {

        auto& sbg = std::get<Dynamic>(nodes[i]->_handle).subgraph;
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
    if(_dependents[i]->_handle.index() == Node::CONDITION) {
      n++;
    }
  }
  return n;
}

// Function: num_strong_dependents
inline size_t Node::num_strong_dependents() const {
  size_t n = 0;
  for(size_t i=0; i<_dependents.size(); i++) {
    if(_dependents[i]->_handle.index() != Node::CONDITION) {
      n++;
    }
  }
  return n;
}

// Function: name
inline const std::string& Node::name() const {
  return _name;
}

// Procedure: _set_state
inline void Node::_set_state(int flag) { 
  _state |= flag; 
}

// Procedure: _unset_state
inline void Node::_unset_state(int flag) { 
  _state &= ~flag; 
}

// Procedure: _clear_state
inline void Node::_clear_state() { 
  _state = 0; 
}

// Function: _has_state
inline bool Node::_has_state(int flag) const {
  return _state & flag;
}

// Function: _is_cancelled
inline bool Node::_is_cancelled() const {
  if(_handle.index() == Node::ASYNC) {
    auto& h = std::get<Node::Async>(_handle);
    if(h.topology && h.topology->_is_cancelled) {
      return true;
    }
  }
  // async tasks spawned from subflow does not have topology
  return _topology && _topology->_is_cancelled;
}

// Procedure: _set_up_join_counter
inline void Node::_set_up_join_counter() {

  size_t c = 0;

  for(auto p : _dependents) {
    if(p->_handle.index() == Node::CONDITION) {
      _set_state(Node::BRANCHED);
    }
    else {
      c++;
    }
  }

  _join_counter.store(c, std::memory_order_relaxed);
}


// Function: _acquire_all
inline bool Node::_acquire_all(std::vector<Node*>& nodes) {

  auto& to_acquire = _semaphores->to_acquire;

  for(size_t i = 0; i < to_acquire.size(); ++i) {
    if(!to_acquire[i]->_try_acquire_or_wait(this)) {
      for(size_t j = 1; j <= i; ++j) {
        auto r = to_acquire[i-j]->_release();
        nodes.insert(end(nodes), begin(r), end(r));
      }
      return false;
    }
  }
  return true;
}

// Function: _release_all
inline std::vector<Node*> Node::_release_all() {

  auto& to_release = _semaphores->to_release;

  std::vector<Node*> nodes;
  for(const auto& sem : to_release) {
    auto r = sem->_release();
    nodes.insert(end(nodes), begin(r), end(r));
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
  //auto& np = _node_pool();
  for(auto node : _nodes) {
    //np.recycle(node);
    node_pool.recycle(node);
  }
}

// Move constructor
inline Graph::Graph(Graph&& other) : 
  _nodes {std::move(other._nodes)} {
}

// Move assignment
inline Graph& Graph::operator = (Graph&& other) {
  _nodes = std::move(other._nodes);
  return *this;
}

// Procedure: clear
inline void Graph::clear() {
  //auto& np = _node_pool();
  for(auto node : _nodes) {
    //node->~Node();
    //np.deallocate(node);
    node_pool.recycle(node);
  }
  _nodes.clear();
}

// Procedure: clear_detached
inline void Graph::clear_detached() {

  auto mid = std::partition(_nodes.begin(), _nodes.end(), [] (Node* node) {
    return !(node->_has_state(Node::DETACHED));
  });
  
  //auto& np = _node_pool();
  for(auto itr = mid; itr != _nodes.end(); ++itr) {
    node_pool.recycle(*itr);
  }
  _nodes.resize(std::distance(_nodes.begin(), mid));
}

// Procedure: merge
inline void Graph::merge(Graph&& g) {
  for(auto n : g._nodes) {
    _nodes.push_back(n);
  }
  g._nodes.clear();
}

// Function: size
// query the size
inline size_t Graph::size() const {
  return _nodes.size();
}

// Function: empty
// query the emptiness
inline bool Graph::empty() const {
  return _nodes.empty();
}
    
// Function: emplace_back
// create a node from a give argument; constructor is called if necessary
template <typename ...ArgsT>
Node* Graph::emplace_back(ArgsT&&... args) {
  //auto node = _node_pool().allocate();
  //new (node) Node(std::forward<ArgsT>(args)...);
  //_nodes.push_back(node);
  _nodes.push_back(node_pool.animate(std::forward<ArgsT>(args)...));
  return _nodes.back();
}

// Function: emplace_back
// create a node from a give argument; constructor is called if necessary
inline Node* Graph::emplace_back() {
  //auto node = _node_pool().allocate();
  //new (node) Node();
  //_nodes.push_back(node);
  _nodes.push_back(node_pool.animate());
  return _nodes.back();
}


}  // end of namespace tf. ---------------------------------------------------





