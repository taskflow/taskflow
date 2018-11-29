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

  // ensure the topology is connected
  //_source.precede(_target);

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

// Class: Task
class Task {

  friend class FlowBuilder;

  template <template<typename...> typename E> 
  friend class BasicTaskflow;

  public:
    
    Task() = default;
    Task(Node&);
    Task(const Task&);
    Task(Task&&);

    Task& operator = (const Task&);

    const std::string& name() const;

    size_t num_successors() const;
    size_t num_dependents() const;

    Task& name(const std::string&);

    template <typename C>
    Task& work(C&&);

    template <typename... Ts>
    Task& precede(Ts&&...);
    
    template <typename... Bs>
    Task& broadcast(Bs&&...);
    
    Task& broadcast(std::vector<Task>&);
    Task& broadcast(std::initializer_list<Task>);
  
    template <typename... Bs>
    Task& gather(Bs&&...);

    Task& gather(std::vector<Task>&);
    Task& gather(std::initializer_list<Task>);

  private:

    Node* _node {nullptr};

    template<typename S>
    void _broadcast(S&);

    template<typename S>
    void _gather(S&);
};

// Constructor
inline Task::Task(Node& t) : _node {&t} {
}

// Constructor
inline Task::Task(const Task& rhs) : _node {rhs._node} {
}

// Function: broadcast
template <typename... Bs>
Task& Task::broadcast(Bs&&... tgts) {
  (_node->precede(*(tgts._node)), ...);
  return *this;
}

// Procedure: _broadcast
template <typename S>
inline void Task::_broadcast(S& tgts) {
  for(auto& to : tgts) {
    _node->precede(*(to._node));
  }
}
      
// Function: broadcast
inline Task& Task::broadcast(std::vector<Task>& tgts) {
  _broadcast(tgts);
  return *this;
}

// Function: broadcast
inline Task& Task::broadcast(std::initializer_list<Task> tgts) {
  _broadcast(tgts);
  return *this;
}

// Function: precede
template <typename... Ts>
Task& Task::precede(Ts&&... tgts) {
  (_node->precede(*(tgts._node)), ...);
  return *this;
}

// Function: gather
template <typename... Bs>
Task& Task::gather(Bs&&... tgts) {
  (tgts._node->precede(*_node), ...);
  return *this;
}

// Procedure: _gather
template <typename S>
void Task::_gather(S& tgts) {
  for(auto& from : tgts) {
    from._node->precede(*_node);
  }
}

// Function: gather
inline Task& Task::gather(std::vector<Task>& tgts) {
  _gather(tgts);
  return *this;
}

// Function: gather
inline Task& Task::gather(std::initializer_list<Task> tgts) {
  _gather(tgts);
  return *this;
}

// Operator =
inline Task& Task::operator = (const Task& rhs) {
  _node = rhs._node;
  return *this;
}

// Constructor
inline Task::Task(Task&& rhs) : _node{rhs._node} { 
  rhs._node = nullptr; 
}

// Function: work
template <typename C>
inline Task& Task::work(C&& c) {
  _node->_work = std::forward<C>(c);
  return *this;
}

// Function: name
inline Task& Task::name(const std::string& name) {
  _node->_name = name;
  return *this;
}

// Function: name
inline const std::string& Task::name() const {
  return _node->_name;
}

// Function: num_dependents
inline size_t Task::num_dependents() const {
  return _node->_dependents.load(std::memory_order_relaxed);
}

// Function: num_successors
inline size_t Task::num_successors() const {
  return _node->_successors.size();
}

};  // end of namespace tf. ---------------------------------------------------

