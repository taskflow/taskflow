#pragma once

#include "error.hpp"
#include "declarations.hpp"
#include "semaphore.hpp"
#include "../utility/iterator.hpp"
#include "../utility/object_pool.hpp"
#include "../utility/traits.hpp"
#include "../utility/singleton.hpp"
#include "../utility/os.hpp"
#include "../utility/math.hpp"

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
  friend class Topology;
  friend class Taskflow;
  friend class Executor;
  friend class FlowBuilder;
  friend class Subflow;

  TF_ENABLE_POOLABLE_ON_THIS;

  // state bit flag
  constexpr static int BRANCHED = 0x1;
  constexpr static int DETACHED = 0x2;
  
  // static work handle
  struct StaticTask {

    template <typename C> 
    StaticTask(C&&);

    std::function<void()> work;
  };

  // dynamic work handle
  struct DynamicTask {

    template <typename C> 
    DynamicTask(C&&);

    std::function<void(Subflow&)> work;
    Graph subgraph;
  };
  
  // condition work handle
  struct ConditionTask {

    template <typename C> 
    ConditionTask(C&&);

    std::function<int()> work;
  };

  // module work handle
  struct ModuleTask {

    template <typename T>
    ModuleTask(T&&);

    Taskflow* module {nullptr};
  };

  // Async work
  struct AsyncTask {

    template <typename T>
    AsyncTask(T&&);

    std::function<void()> work;
  };
  
  // cudaFlow work handle
  struct cudaFlowTask {
    
    template <typename C, typename G> 
    cudaFlowTask(C&& c, G&& g);

    std::function<void(Executor&, Node*)> work;

    std::unique_ptr<CustomGraphBase> graph;
  };
    
  using handle_t = std::variant<
    std::monostate,  // placeholder
    StaticTask,      // static tasking
    DynamicTask,     // dynamic tasking
    ConditionTask,   // conditional tasking
    ModuleTask,      // composable tasking
    AsyncTask,       // async work
    cudaFlowTask     // cudaFlow
  >;
    
  struct Semaphores {  
    std::vector<Semaphore*> to_acquire;
    std::vector<Semaphore*> to_release;
  };

  public:
  
  // variant index
  constexpr static auto PLACEHOLDER_TASK = get_index_v<std::monostate, handle_t>;
  constexpr static auto STATIC_TASK      = get_index_v<StaticTask, handle_t>;
  constexpr static auto DYNAMIC_TASK     = get_index_v<DynamicTask, handle_t>;
  constexpr static auto CONDITION_TASK   = get_index_v<ConditionTask, handle_t>; 
  constexpr static auto MODULE_TASK      = get_index_v<ModuleTask, handle_t>; 
  constexpr static auto ASYNC_TASK       = get_index_v<AsyncTask, handle_t>; 
  constexpr static auto CUDAFLOW_TASK    = get_index_v<cudaFlowTask, handle_t>; 

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

    std::optional<Semaphores> _semaphores;

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

    bool _acquire_all(std::vector<Node*>&);
    std::vector<Node*> _release_all();
};

// ----------------------------------------------------------------------------
// Node Object Pool
// ----------------------------------------------------------------------------
inline ObjectPool<Node> node_pool;

// ----------------------------------------------------------------------------
// Definition for Node::StaticTask
// ----------------------------------------------------------------------------
    
// Constructor
template <typename C> 
Node::StaticTask::StaticTask(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::DynamicTask
// ----------------------------------------------------------------------------
    
// Constructor
template <typename C> 
Node::DynamicTask::DynamicTask(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::ConditionTask
// ----------------------------------------------------------------------------
    
// Constructor
template <typename C> 
Node::ConditionTask::ConditionTask(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::cudaFlowTask
// ----------------------------------------------------------------------------

template <typename C, typename G>
Node::cudaFlowTask::cudaFlowTask(C&& c, G&& g) :
  work  {std::forward<C>(c)},
  graph {std::forward<G>(g)} {
}
    
// ----------------------------------------------------------------------------
// Definition for Node::ModuleTask
// ----------------------------------------------------------------------------
    
// Constructor
template <typename T>
Node::ModuleTask::ModuleTask(T&& tf) : module {tf} {
}

// ----------------------------------------------------------------------------
// Definition for Node::AsyncTask
// ----------------------------------------------------------------------------
    
// Constructor
template <typename C>
Node::AsyncTask::AsyncTask(C&& c) : work {std::forward<C>(c)} {
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

  if(_handle.index() == DYNAMIC_TASK) {

    auto& subgraph = std::get<DynamicTask>(_handle).subgraph;

    std::vector<Node*> nodes;

    std::move(
      subgraph._nodes.begin(), subgraph._nodes.end(), std::back_inserter(nodes)
    );
    subgraph._nodes.clear();

    size_t i = 0;

    while(i < nodes.size()) {

      if(nodes[i]->_handle.index() == DYNAMIC_TASK) {

        auto& sbg = std::get<DynamicTask>(nodes[i]->_handle).subgraph;
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
  return std::count_if(
    _dependents.begin(), 
    _dependents.end(), 
    [](Node* node){ return node->_handle.index() == Node::CONDITION_TASK; } 
  );
}

// Function: num_strong_dependents
inline size_t Node::num_strong_dependents() const {
  return std::count_if(
    _dependents.begin(), 
    _dependents.end(), 
    [](Node* node){ return node->_handle.index() != Node::CONDITION_TASK; } 
  );
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

// Procedure: _set_up_join_counter
inline void Node::_set_up_join_counter() {

  int c = 0;

  for(auto p : _dependents) {
    if(p->_handle.index() == Node::CONDITION_TASK) {
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
  for(auto const & sem : to_release) {
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





