#pragma once

#include "error.hpp"
#include "../declarations.hpp"
#include "../utility/object_pool.hpp"
#include "../utility/traits.hpp"
#include "../utility/passive_vector.hpp"
#include "../utility/singleton.hpp"
#include "../utility/uuid.hpp"
#include "../utility/os.hpp"
#include "../nstd/variant.hpp"

#if defined(__CUDA__) || defined(__CUDACC__)
#define TF_ENABLE_CUDA
#include "../cuda/cuda_flow.hpp"
#endif

namespace tf {

// ----------------------------------------------------------------------------
// domain
// ----------------------------------------------------------------------------

enum Domain : int {
  HOST = 0,
#ifdef TF_ENABLE_CUDA
  CUDA,
#endif
  NUM_DOMAINS
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

    bool empty() const;

    size_t size() const;
    
    template <typename ...Args>
    Node* emplace_back(Args&& ...); 

    Node* emplace_back();

  private:

    static ObjectPool<Node>& _node_pool();
    
    std::vector<Node*> _nodes;
};

// ----------------------------------------------------------------------------

// Class: Node
class Node {

  friend class Task;
  friend class TaskView;
  friend class Topology;
  friend class Taskflow;
  friend class Executor;
  friend class FlowBuilder;
  friend class Subflow;

  TF_ENABLE_POOLABLE_ON_THIS;

  // state bit flag
  constexpr static int SPAWNED = 0x1;
  constexpr static int BRANCH  = 0x2;
  
  // static work handle
  struct StaticWork {

    template <typename C> 
    StaticWork(C&&);

    std::function<void()> work;
  };

  // dynamic work handle
  struct DynamicWork {

    template <typename C> 
    DynamicWork(C&&);

    std::function<void(Subflow&)> work;
    Graph subgraph;
  };
  
  // condition work handle
  struct ConditionWork {

    template <typename C> 
    ConditionWork(C&&);

    std::function<int()> work;
  };

  // module work handle
  struct ModuleWork {

    template <typename T>
    ModuleWork(T&&);

    Taskflow* module {nullptr};
  };
  
  // cudaFlow work handle
#ifdef TF_ENABLE_CUDA
  struct cudaFlowWork {
    
    template <typename C> 
    cudaFlowWork(C&& c) : work {std::forward<C>(c)} {}

    std::function<void(cudaFlow&)> work;

    cudaGraph graph;
  };
#endif
    
  using handle_t = nstd::variant<
    nstd::monostate,  // placeholder
#ifdef TF_ENABLE_CUDA
    cudaFlowWork,     // cudaFlow
#endif
    StaticWork,       // static tasking
    DynamicWork,      // dynamic tasking
    ConditionWork,    // conditional tasking
    ModuleWork        // composable tasking
  >;
  
  public:
  
  // variant index
  constexpr static auto PLACEHOLDER_WORK = get_index_v<nstd::monostate, handle_t>;
  constexpr static auto STATIC_WORK      = get_index_v<StaticWork, handle_t>;
  constexpr static auto DYNAMIC_WORK     = get_index_v<DynamicWork, handle_t>;
  constexpr static auto CONDITION_WORK   = get_index_v<ConditionWork, handle_t>; 
  constexpr static auto MODULE_WORK      = get_index_v<ModuleWork, handle_t>; 

#ifdef TF_ENABLE_CUDA
  constexpr static auto CUDAFLOW_WORK  = get_index_v<cudaFlowWork, handle_t>; 
#endif

    template <typename ...Args>
    Node(Args&&... args);

    ~Node();

    size_t num_successors() const;
    size_t num_dependents() const;
    size_t num_strong_dependents() const;
    size_t num_weak_dependents() const;
    
    const std::string& name() const;

    Domain domain() const;

  private:

    std::string _name;

    handle_t _handle;

    PassiveVector<Node*> _successors;
    PassiveVector<Node*> _dependents;

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

};

// ----------------------------------------------------------------------------
// Definition for Node::StaticWork
// ----------------------------------------------------------------------------
    
// Constructor
template <typename C> 
Node::StaticWork::StaticWork(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::DynamicWork
// ----------------------------------------------------------------------------
    
// Constructor
template <typename C> 
Node::DynamicWork::DynamicWork(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::ConditionWork
// ----------------------------------------------------------------------------
    
// Constructor
template <typename C> 
Node::ConditionWork::ConditionWork(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::ModuleWork
// ----------------------------------------------------------------------------
    
// Constructor
template <typename T>
Node::ModuleWork::ModuleWork(T&& tf) : module {tf} {
}

// ----------------------------------------------------------------------------
// Definition for Node
// ----------------------------------------------------------------------------

// Constructor
template <typename ...Args>
Node::Node(Args&&... args): _handle{std::forward<Args>(args)...} {
} 

// Destructor
inline Node::~Node() {
  // this is to avoid stack overflow

  if(_handle.index() == DYNAMIC_WORK) {

    auto& subgraph = nstd::get<DynamicWork>(_handle).subgraph;

    std::vector<Node*> nodes;

    std::move(
     subgraph._nodes.begin(), subgraph._nodes.end(), std::back_inserter(nodes)
    );
    subgraph._nodes.clear();

    size_t i = 0;

    while(i < nodes.size()) {

      if(nodes[i]->_handle.index() == DYNAMIC_WORK) {

        auto& sbg = nstd::get<DynamicWork>(nodes[i]->_handle).subgraph;
        std::move(
          sbg._nodes.begin(), sbg._nodes.end(), std::back_inserter(nodes)
        );
        sbg._nodes.clear();
      }

      ++i;
    }
      
    auto& np = Graph::_node_pool();
    for(i=0; i<nodes.size(); ++i) {
      //nodes[i]->~Node();
      //np.deallocate(nodes[i]);
      np.recycle(nodes[i]);
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
    [](Node* node){ return node->_handle.index() == Node::CONDITION_WORK; } 
  );
}

// Function: num_strong_dependents
inline size_t Node::num_strong_dependents() const {
  return std::count_if(
    _dependents.begin(), 
    _dependents.end(), 
    [](Node* node){ return node->_handle.index() != Node::CONDITION_WORK; } 
  );
}

// Function: name
inline const std::string& Node::name() const {
  return _name;
}

// Function: domain
inline Domain Node::domain() const {

  Domain domain;

  switch(_handle.index()) {

    case STATIC_WORK:
    case DYNAMIC_WORK:
    case CONDITION_WORK:
    case MODULE_WORK:
      domain = Domain::HOST;
    break;

#ifdef TF_ENABLE_CUDA
    case CUDAFLOW_WORK:
      domain = Domain::CUDA;
    break;
#endif

    default:
      domain = Domain::HOST;
    break;
  }

  return domain;
}

//
//// Function: dump
//inline std::string Node::dump() const {
//  std::ostringstream os;  
//  dump(os);
//  return os.str();
//}
//
//// Function: dump
//inline void Node::dump(std::ostream& os) const {
//
//  os << 'p' << this << "[label=\"";
//  if(_name.empty()) os << 'p' << this;
//  else os << _name;
//  os << "\" ";
//
//  // condition node is colored green
//  if(_handle.index() == CONDITION_WORK) {
//    os << " shape=diamond color=black fillcolor=aquamarine style=filled";
//  }
//
//  os << "];\n";
//  
//  for(size_t s=0; s<_successors.size(); ++s) {
//    if(_handle.index() == CONDITION_WORK) {
//      // case edge is dashed
//      os << 'p' << this << " -> p" << _successors[s] 
//         << " [style=dashed label=\"" << s << "\"];\n";
//    }
//    else {
//      os << 'p' << this << " -> p" << _successors[s] << ";\n";
//    }
//  }
//  
//  // subflow join node
//  if(_parent && _successors.size() == 0) {
//    os << 'p' << this << " -> p" << _parent << ";\n";
//  }
//  
//  if(_subgraph && !_subgraph->empty()) {
//
//    os << "subgraph cluster_p" << this << " {\nlabel=\"Subflow: ";
//    if(_name.empty()) os << 'p' << this;
//    else os << _name;
//
//    os << "\";\n" << "color=blue\n";
//
//    for(const auto& n : _subgraph->nodes()) {
//      n->dump(os);
//    }
//    os << "}\n";
//  }
//}
    
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

// Procedure: _set_up_join_counter
inline void Node::_set_up_join_counter() {

  int c = 0;

  for(auto p : _dependents) {
    if(p->_handle.index() == Node::CONDITION_WORK) {
      _set_state(Node::BRANCH);
    }
    else {
      c++;
    }
  }

  _join_counter.store(c, std::memory_order_relaxed);
}

// Function: _has_state
inline bool Node::_has_state(int flag) const {
  return _state & flag;
}

// ----------------------------------------------------------------------------
// Graph definition
// ----------------------------------------------------------------------------
    
// Function: _node_pool
inline ObjectPool<Node>& Graph::_node_pool() {
  static ObjectPool<Node> pool;
  return pool;
}

// Destructor
inline Graph::~Graph() {
  auto& np = _node_pool();
  for(auto node : _nodes) {
    //node->~Node();
    //np.deallocate(node);
    np.recycle(node);
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
  auto& np = _node_pool();
  for(auto node : _nodes) {
    //node->~Node();
    //np.deallocate(node);
    np.recycle(node);
  }
  _nodes.clear();
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
  _nodes.push_back(_node_pool().animate(std::forward<ArgsT>(args)...));
  return _nodes.back();
}

// Function: emplace_back
// create a node from a give argument; constructor is called if necessary
inline Node* Graph::emplace_back() {
  //auto node = _node_pool().allocate();
  //new (node) Node();
  //_nodes.push_back(node);
  _nodes.push_back(_node_pool().animate());
  return _nodes.back();
}


}  // end of namespace tf. ---------------------------------------------------





