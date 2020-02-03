#pragma once

#include <stack>

#include "flow_builder.hpp"
#include "topology.hpp"

namespace tf {

// ----------------------------------------------------------------------------

/**
@class Taskflow 

@brief the class to create a task dependency graph

*/
class Taskflow : public FlowBuilder {

  friend class Topology;
  friend class Executor;
  friend class FlowBuilder;

  public:

    /**
    @brief constructs a taskflow with a given name
    */
    Taskflow(const std::string& name);

    /**
    @brief constructs a taskflow
    */
    Taskflow();

    /**
    @brief destroy the taskflow (virtual call)
    */
    virtual ~Taskflow();
    
    /**
    @brief dumps the taskflow to a std::ostream in DOT format

    @param ostream a std::ostream target
    */
    void dump(std::ostream& ostream) const;
    
    /**
    @brief dumps the taskflow in DOT format to a std::string
    */
    std::string dump() const;
    
    /**
    @brief queries the number of tasks in the taskflow
    */
    size_t num_tasks() const;
    
    /**
    @brief queries the emptiness of the taskflow
    */
    bool empty() const;

    /**
    @brief sets the name of the taskflow
    
    @return @c *this
    */
    tf::Taskflow& name(const std::string&); 

    /**
    @brief queries the name of the taskflow
    */
    const std::string& name() const ;
    
    /**
    @brief clears the associated task dependency graph
    */
    void clear();

    /**
    @brief applies an visitor callable to each task in the taskflow
    */
    template <typename V>
    void for_each_task(V&& visitor) const;

  private:
 
    std::string _name;
   
    Graph _graph;

    std::mutex _mtx;

    std::list<Topology> _topologies;

    void _dump(std::ostream&, const Taskflow*) const;

    void _dump(
      std::ostream&, 
      const Node*,
      std::stack<const Taskflow*>&,
      std::unordered_set<const Taskflow*>&
    ) const;

    void _dump(
      std::ostream&,
      const Graph&,
      std::stack<const Taskflow*>&,
      std::unordered_set<const Taskflow*>&
    ) const;
};

// Constructor
inline Taskflow::Taskflow(const std::string& name) : 
  FlowBuilder {_graph},
  _name       {name} {
}

// Constructor
inline Taskflow::Taskflow() : FlowBuilder{_graph} {
}

// Destructor
inline Taskflow::~Taskflow() {
  assert(_topologies.empty());
}

// Procedure:
inline void Taskflow::clear() {
  _graph.clear();
}

// Function: num_noces
inline size_t Taskflow::num_tasks() const {
  return _graph.size();
}

// Function: empty
inline bool Taskflow::empty() const {
  return _graph.empty();
}

// Function: name
inline Taskflow& Taskflow::name(const std::string &name) {
  _name = name;
  return *this;
}

// Function: name
inline const std::string& Taskflow::name() const {
  return _name;
}

// Function: for_each_task
template <typename V>
void Taskflow::for_each_task(V&& visitor) const {
  for(size_t i=0; i<_graph._nodes.size(); ++i) {
    visitor(Task(_graph._nodes[i]));
  }
}

// Procedure: dump
inline std::string Taskflow::dump() const {
  std::ostringstream oss;
  dump(oss);
  return oss.str();
}

// Function: dump
inline void Taskflow::dump(std::ostream& os) const {
  os << "digraph Taskflow {\n";
  _dump(os, this);
  os << "}\n";
}

// Procedure: _dump
inline void Taskflow::_dump(std::ostream& os, const Taskflow* top) const {
  
  std::stack<const Taskflow*> stack;
  std::unordered_set<const Taskflow*> visited; 
  
  stack.push(top);
  visited.insert(top);

  while(!stack.empty()) {
    
    auto f = stack.top();
    stack.pop();
    
    os << "subgraph cluster_p" << f << " {\nlabel=\"Taskflow: ";
    if(f->_name.empty()) os << 'p' << f;
    else os << f->_name;
    os << "\";\n";
    _dump(os, f->_graph, stack, visited);
    os << "}\n";
  }
}

// Procedure: _dump
inline void Taskflow::_dump(
  std::ostream& os, 
  const Node* node,
  std::stack<const Taskflow*>& stack,
  std::unordered_set<const Taskflow*>& visited
) const {

  os << 'p' << node << "[label=\"";
  if(node->_name.empty()) os << 'p' << node;
  else os << node->_name;
  os << "\" ";

  // condition node is colored green
  if(node->_work.index() == Node::CONDITION_WORK) {
    os << " shape=diamond color=black fillcolor=aquamarine style=filled";
  }

  os << "];\n";
  
  for(size_t s=0; s<node->_successors.size(); ++s) {
    if(node->_work.index() == Node::CONDITION_WORK) {
      // case edge is dashed
      os << 'p' << node << " -> p" << node->_successors[s] 
         << " [style=dashed label=\"" << s << "\"];\n";
    }
    else {
      os << 'p' << node << " -> p" << node->_successors[s] << ";\n";
    }
  }
  
  // subflow join node
  if(node->_parent && node->_successors.size() == 0) {
    os << 'p' << node << " -> p" << node->_parent << ";\n";
  }
  
  if(node->_subgraph && !node->_subgraph->empty()) {

    os << "subgraph cluster_p" << node << " {\nlabel=\"Subflow: ";
    if(node->_name.empty()) os << 'p' << node;
    else os << node->_name;

    os << "\";\n" << "color=blue\n";
    _dump(os, *(node->_subgraph), stack, visited);
    os << "}\n";
  }
}

// Procedure: _dump
inline void Taskflow::_dump(
  std::ostream& os, 
  const Graph& graph,
  std::stack<const Taskflow*>& stack,
  std::unordered_set<const Taskflow*>& visited
) const {
    
  for(const auto& n : graph._nodes) {
    // regular task
    if(auto module = n->_module; !module) {
      _dump(os, n, stack, visited);
    }
    // module task
    else {
      os << 'p' << n << "[shape=box, color=blue, label=\"";
      if(n->_name.empty()) os << n;
      else os << n->_name;
      os << " [Taskflow: ";
      if(module->_name.empty()) os << 'p' << module;
      else os << module->_name;
      os << "]\"];\n";

      if(visited.find(module) == visited.end()) {
        visited.insert(module);
        stack.push(module);
      }

      for(const auto s : n->_successors) {
        os << 'p' << n << "->" << 'p' << s << ";\n";
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Backward compatibility
// ----------------------------------------------------------------------------

using Framework = Taskflow;

}  // end of namespace tf. ---------------------------------------------------

