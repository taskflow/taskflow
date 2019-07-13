#pragma once

#include "../error/error.hpp"
#include "../utility/traits.hpp"
#include "../utility/passive_vector.hpp"

namespace tf {

// Forward declaration
class Node;
class Topology;
class Task;
class FlowBuilder;
class Subflow;
class Taskflow;

// ----------------------------------------------------------------------------

// Class: Graph
class Graph {

  friend class Node;
  
  public:

    Graph() = default;
    Graph(const Graph&) = delete;
    Graph(Graph&&);

    Graph& operator = (const Graph&) = delete;
    Graph& operator = (Graph&&);
    
    void clear();

    bool empty() const;

    size_t size() const;
    
    template <typename C>
    Node& emplace_back(C&&); 

    Node& emplace_back();

    std::vector<std::unique_ptr<Node>>& nodes();

    const std::vector<std::unique_ptr<Node>>& nodes() const;

  private:
    
    std::vector<std::unique_ptr<Node>> _nodes;
};

// ----------------------------------------------------------------------------

// Class: Node
class Node {

  friend class Task;
  friend class TaskView;
  friend class Topology;
  friend class Taskflow;
  friend class Executor;

  using StaticWork  = std::function<void()>;
  using DynamicWork = std::function<void(Subflow&)>;

  constexpr static int SPAWNED = 0x1;
  constexpr static int SUBTASK = 0x2;

  public:

    Node() = default;

    template <typename C>
    Node(C&&);
    
    ~Node();
    
    void precede(Node&);
    void dump(std::ostream&) const;

    size_t num_successors() const;
    size_t num_dependents() const;
    
    const std::string& name() const;

    std::string dump() const;

    // Status-related functions
    bool is_spawned() const { return _status & SPAWNED; }
    bool is_subtask() const { return _status & SUBTASK; }

    void set_spawned()   { _status |= SPAWNED;  }
    void set_subtask()   { _status |= SUBTASK;  }
    void unset_spawned() { _status &= ~SPAWNED; }
    void unset_subtask() { _status &= ~SUBTASK; }
    void clear_status()  { _status = 0;         }

  private:
    
    std::string _name;
    std::variant<std::monostate, StaticWork, DynamicWork> _work;

    tf::PassiveVector<Node*> _successors;
    tf::PassiveVector<Node*> _dependents;
    
    std::optional<Graph> _subgraph;

    Topology* _topology {nullptr};
    Taskflow* _module {nullptr};

    int _status {0};
    
    std::atomic<int> _num_dependents {0};
};

// Constructor
template <typename C>
Node::Node(C&& c) : _work {std::forward<C>(c)} {
}

// Destructor
inline Node::~Node() {
  // this is to avoid stack overflow
  if(_subgraph.has_value()) {
    std::vector<std::unique_ptr<Node>> nodes;
    std::move(
     _subgraph->_nodes.begin(), _subgraph->_nodes.end(), std::back_inserter(nodes)
    );
    _subgraph->_nodes.clear();
    _subgraph.reset();
    size_t i = 0;
    while(i < nodes.size()) {
      if(auto& sbg = nodes[i]->_subgraph; sbg) {
        std::move(
          sbg->_nodes.begin(), sbg->_nodes.end(), std::back_inserter(nodes)
        );
        sbg->_nodes.clear();
        sbg.reset();
      }
      ++i;
    }
  }
}

// Procedure: precede
inline void Node::precede(Node& v) {
  _successors.push_back(&v);
  v._dependents.push_back(this);
  v._num_dependents.fetch_add(1, std::memory_order_relaxed);
}

// Function: num_successors
inline size_t Node::num_successors() const {
  return _successors.size();
}

// Function: dependents
inline size_t Node::num_dependents() const {
  return _dependents.size();
}

// Function: name
inline const std::string& Node::name() const {
  return _name;
}

// Function: dump
inline std::string Node::dump() const {
  std::ostringstream os;  
  dump(os);
  return os.str();
}

// Function: dump
inline void Node::dump(std::ostream& os) const {

  os << 'p' << this << "[label=\"";
  if(_name.empty()) os << 'p' << this;
  else os << _name;
  os << "\"];\n";
  
  for(const auto s : _successors) {
    os << 'p' << this << " -> " << 'p' << s << ";\n";
  }
  
  if(_subgraph && !_subgraph->empty()) {

    os << "subgraph cluster_";
    if(_name.empty()) os << 'p' << this;
    else os << _name;
    os << " {\n";

    os << "label=\"Subflow_";
    if(_name.empty()) os << 'p' << this;
    else os << _name;

    os << "\";\n" << "color=blue\n";

    for(const auto& n : _subgraph->nodes()) {
      n->dump(os);
    }
    os << "}\n";
  }
}

// ----------------------------------------------------------------------------

/*// Class: NodePool
class NodePool {

  public:

    template <typename C>
    std::unique_ptr<Node> acquire(C&&);

    std::unique_ptr<Node> acquire();

    void release(std::unique_ptr<Node>);
  
  private:
    
    //std::mutex _mutex;

    std::vector<std::unique_ptr<Node>> _nodes;

    void _recycle(Node&);
};

// Function: acquire
template <typename C>
inline std::unique_ptr<Node> NodePool::acquire(C&& c) {
  if(_nodes.empty()) {
    return std::make_unique<Node>(std::forward<C>(c));
  }
  else {
    auto node = std::move(_nodes.back());
    node->_work = std::forward<C>(c);
    _nodes.pop_back();
    return node;
  }
}

// Function: acquire
inline std::unique_ptr<Node> NodePool::acquire() {
  if(_nodes.empty()) {
    return std::make_unique<Node>();
  }
  else {
    auto node = std::move(_nodes.back());
    _nodes.pop_back();
    return node;
  }
}

// Procedure: release
inline void NodePool::release(std::unique_ptr<Node> node) {

  return;

  //assert(node);
  if(_nodes.size() >= 65536) {
    return;
  }
  
  auto children = node->_extract_children();

  for(auto& child : children) {
    _recycle(*child);
  }
  _recycle(*node);

  std::move(children.begin(), children.end(), std::back_inserter(_nodes));  
  _nodes.push_back(std::move(node));
}

// Procedure: _recycle
inline void NodePool::_recycle(Node& node) {
  node._name.clear();
  node._work = {};
  node._successors.clear();
  node._dependents.clear();
  node._topology = nullptr;
  node._module = nullptr;
  node._status = 0;
  node._num_dependents.store(0, std::memory_order_relaxed);
  //assert(!node._subgraph);
}

// ----------------------------------------------------------------------------

namespace this_thread {
  inline thread_local NodePool nodepool;
}
*/

// ----------------------------------------------------------------------------

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
// clear and recycle the nodes
inline void Graph::clear() {
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
    
// Function: nodes
// return a mutable reference to the node data structure
//inline std::vector<std::unique_ptr<Node>>& Graph::nodes() {
inline std::vector<std::unique_ptr<Node>>& Graph::nodes() {
  return _nodes;
}

// Function: nodes
// returns a constant reference to the node data structure
//inline const std::vector<std::unique_ptr<Node>>& Graph::nodes() const {
inline const std::vector<std::unique_ptr<Node>>& Graph::nodes() const {
  return _nodes;
}

// Function: emplace_back
// create a node from a give argument; constructor is called if necessary
template <typename C>
Node& Graph::emplace_back(C&& c) {
  _nodes.push_back(std::make_unique<Node>(std::forward<C>(c)));
  return *(_nodes.back());
}

// Function: emplace_back
// create a node from a give argument; constructor is called if necessary
inline Node& Graph::emplace_back() {
  _nodes.push_back(std::make_unique<Node>());
  return *(_nodes.back());
}


}  // end of namespace tf. ---------------------------------------------------





