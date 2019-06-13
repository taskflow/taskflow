#pragma once

#include "../error/error.hpp"
#include "../utility/object_pool.hpp"
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
  
  public:

    Graph() = default;
    ~Graph();

    Graph(const Graph&) = delete;
    Graph(Graph&&);

    Graph& operator = (const Graph&) = delete;
    Graph& operator = (Graph&&);
    
    void clear();

    bool empty() const;

    size_t size() const;
    
    template <typename... ArgsT>
    Node& emplace_back(ArgsT&&...); 

    std::vector<Node*>& nodes();

    const std::vector<Node*>& nodes() const;

  private:

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

  using StaticWork  = std::function<void()>;
  using DynamicWork = std::function<void(Subflow&)>;

  constexpr static int SPAWNED = 0x1;
  constexpr static int SUBTASK = 0x2;

  public:

    Node();
    ~Node();

    template <typename C>
    Node(C&&);

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
    std::variant<StaticWork, DynamicWork> _work;

    tf::PassiveVector<Node*> _successors;
    tf::PassiveVector<Node*> _dependents;

    std::optional<Graph> _subgraph;

    Topology* _topology;
    Taskflow* _module;

    int _status;
    
    std::atomic<int> _num_dependents;
};

// Constructor
inline Node::Node() {
  _num_dependents.store(0, std::memory_order_relaxed);
  _topology = nullptr;
  _module = nullptr;
  _status = 0;
}

// Constructor
template <typename C>
Node::Node(C&& c) : _work {std::forward<C>(c)} {
  _num_dependents.store(0, std::memory_order_relaxed);
  _topology = nullptr;
  _module = nullptr;
  _status = 0;
}

// Destructor
inline Node::~Node() {
  // this is to avoid stack overflow
  if(_subgraph.has_value()) {
    std::list<Graph> gs; 
    gs.push_back(std::move(*_subgraph));
    _subgraph.reset();
    auto i=gs.begin();
    while(i!=gs.end()) {
      auto n = i->nodes().begin();
      while(n != i->nodes().end()) {
        if((*n)->_subgraph.has_value()) {
          gs.push_back(std::move(*((*n)->_subgraph)));
          (*n)->_subgraph.reset();
        }
        ++n; 
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

    for(const auto n : _subgraph->nodes()) {
      n->dump(os);
    }
    os << "}\n";
  }
}

// ----------------------------------------------------------------------------

// Move constructor
inline Graph::Graph(Graph&& other) : 
  _nodes {std::move(other._nodes)} {
}

// Destructor
inline Graph::~Graph() {
  clear();
}

// Move assignment
inline Graph& Graph::operator = (Graph&& other) {
  _nodes = std::move(other._nodes);
  return *this;
}

// Procedure: clear
// clear and recycle the nodes
inline void Graph::clear() {
  auto& pool = per_thread_object_pool<Node>(); 
  for(auto& node : _nodes) {
    pool.recycle(node);
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
    
// Function: nodes
// return a mutable reference to the node data structure
//inline std::vector<std::unique_ptr<Node>>& Graph::nodes() {
inline std::vector<Node*>& Graph::nodes() {
  return _nodes;
}

// Function: nodes
// returns a constant reference to the node data structure
//inline const std::vector<std::unique_ptr<Node>>& Graph::nodes() const {
inline const std::vector<Node*>& Graph::nodes() const {
  return _nodes;
}

// Function: emplace_back
// create a node from a give argument; constructor is called if necessary
template <typename... ArgsT>
Node& Graph::emplace_back(ArgsT&&... args) {
  _nodes.push_back(
    per_thread_object_pool<Node>().get(std::forward<ArgsT>(args)...)
  );
  return *(_nodes.back());
}


}  // end of namespace tf. ---------------------------------------------------





