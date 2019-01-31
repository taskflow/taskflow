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
class SubflowBuilder;
class Framework;

using Graph = std::list<Node>;

// ----------------------------------------------------------------------------

// Class: Node
class Node {

  friend class Task;
  friend class Topology;

  template <template<typename...> typename E> 
  friend class BasicTaskflow;

  using StaticWork   = std::function<void()>;
  using DynamicWork  = std::function<void(SubflowBuilder&)>;

  public:

    Node();
    ~Node();

    template <typename C>
    Node(C&&);

    const std::string& name() const;
    
    void precede(Node&);
    void dump(std::ostream&) const;

    size_t num_successors() const;
    size_t num_dependents() const;

    std::string dump() const;

  private:
    
    std::string _name;
    std::variant<StaticWork, DynamicWork> _work;

    tf::PassiveVector<Node*> _successors;
    tf::PassiveVector<Node*> _predecessors;

    std::atomic<int> _dependents;

    std::optional<Graph> _subgraph;

    Topology* _topology;

    bool _spawned {false};
    bool _subtask {false};
};

// Constructor
inline Node::Node() {
  _dependents.store(0, std::memory_order_relaxed);
  _topology = nullptr;
}

// Constructor
template <typename C>
inline Node::Node(C&& c) : _work {std::forward<C>(c)} {
  _dependents.store(0, std::memory_order_relaxed);
  _topology = nullptr;
}


// Destructor
inline Node::~Node() {
  if(_subgraph.has_value()) {
    std::list<Graph> gs; 
    gs.push_back(std::move(*_subgraph));
    _subgraph.reset();
    auto i=gs.begin();
    while(i!=gs.end()) {
      auto n = i->begin();
      while(n != i->end()) {
        if(n->_subgraph.has_value()) {
          gs.push_back(std::move(*(n->_subgraph)));
          n->_subgraph.reset();
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
  v._predecessors.push_back(this);
  v._dependents.fetch_add(1, std::memory_order_relaxed);
}

// Function: num_successors
inline size_t Node::num_successors() const {
  return _successors.size();
}

// Function: dependents
inline size_t Node::num_dependents() const {
  return _predecessors.size();
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
  
  if(_name.empty()) os << '\"' << this << '\"';
  else os << std::quoted(_name);
  os << ";\n";

  for(const auto s : _successors) {

    if(_name.empty()) os << '\"' << this << '\"';
    else os << std::quoted(_name);

    os << " -> ";
    
    if(s->name().empty()) os << '\"' << s << '\"';
    else os << std::quoted(s->name());

    os << ";\n";
  }
  
  if(_subgraph && !_subgraph->empty()) {

    os << "subgraph cluster_";
    if(_name.empty()) os << this;
    else os << _name;
    os << " {\n";

    os << "label = \"Subflow_";
    if(_name.empty()) os << this;
    else os << _name;

    os << "\";\n" << "color=blue\n";

    for(const auto& n : *_subgraph) {
      n.dump(os);
    }
    os << "}\n";
  }
}

}  // end of namespace tf. ---------------------------------------------------


