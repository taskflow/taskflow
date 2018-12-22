#pragma once

#include "../error/error.hpp"
#include "../utility/utility.hpp"

namespace tf {

// Forward declaration
class Node;
class Topology;
class Task;
class FlowBuilder;
class SubflowBuilder;

using Graph = std::forward_list<Node>;

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
    std::vector<Node*> _successors;
    std::atomic<int> _dependents;

    std::optional<Graph> _subgraph;

    Topology* _topology;
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

// Procedure: precede
inline void Node::precede(Node& v) {
  _successors.push_back(&v);
  v._dependents.fetch_add(1, std::memory_order_relaxed);
}

// Function: num_successors
inline size_t Node::num_successors() const {
  return _successors.size();
}

// Function: dependents
inline size_t Node::num_dependents() const {
  return _dependents.load(std::memory_order_relaxed);
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

// ----------------------------------------------------------------------------
  
// class: Topology
class Topology {
  
  template <template<typename...> typename E> 
  friend class BasicTaskflow;

  public:

    Topology(Graph&&);

    template <typename C>
    Topology(Graph&&, C&&);

    std::string dump() const;
    void dump(std::ostream&) const;

  private:

    Graph _graph;

    std::shared_future<void> _future;

    std::vector<Node*> _sources;

    Node _target;
};

// TODO: remove duplicate code in the two constructors

// Constructor
inline Topology::Topology(Graph&& t) : 
  _graph(std::move(t)) {

  _target._topology = this;
  
  std::promise<void> promise;

  _future = promise.get_future().share();

  _target._work = [p=MoC{std::move(promise)}] () mutable { 
    p.get().set_value(); 
  };

  // Build the super source and super target.
  for(auto& node : _graph) {

    node._topology = this;

    if(node.num_dependents() == 0) {
      _sources.push_back(&node);
    }

    if(node.num_successors() == 0) {
      node.precede(_target);
    }
  }
}


// Constructor
template <typename C>
inline Topology::Topology(Graph&& t, C&& c) : 
  _graph(std::move(t)) {

  //_source._topology = this;
  _target._topology = this;
  
  std::promise<void> promise;

  _future = promise.get_future().share();

  _target._work = [p=MoC{std::move(promise)}, c{std::forward<C>(c)}] () mutable { 
    p.get().set_value();
    c();
  };

  // Build the super source and super target.
  for(auto& node : _graph) {

    node._topology = this;

    if(node.num_dependents() == 0) {
      _sources.push_back(&node);
    }

    if(node.num_successors() == 0) {
      node.precede(_target);
    }
  }
}


// Procedure: dump
inline void Topology::dump(std::ostream& os) const {

  assert(!(_target._subgraph));
  
  os << "digraph Topology {\n"
     << _target.dump();

  for(const auto& node : _graph) {
    os << node.dump();
  }

  os << "}\n";
}
  
// Function: dump
inline std::string Topology::dump() const { 
  std::ostringstream os;
  dump(os);
  return os.str();
}

// ----------------------------------------------------------------------------



};  // end of namespace tf. ---------------------------------------------------

