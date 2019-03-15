#pragma once

#include <stack>
#include "flow_builder.hpp"

namespace tf {

/**
@class Framework 

@brief A reusable task dependency graph.

A framework is a task dependency graph that is independent
of a taskflow object. You can run a framework multiple times 
from a taskflow object to enable a reusable control flow.

*/
class Framework : public FlowBuilder {

  template <template<typename...> typename E> 
  friend class BasicTaskflow;
  
  friend class Topology;

  public:

    /**
    @brief constructs the framework with an empty task dependency graph
    */
    Framework();

    /**
    @brief destroy the framework (virtual call)
    */
    virtual ~Framework();
    
    /**
    @brief dumps the framework to a std::ostream in DOT format

    @param ostream a std::ostream target
    */
    void dump(std::ostream& ostream) const;
    
    /**
    @brief dumps the framework in DOT format to a std::string
    */
    std::string dump() const;
    
    /**
    @brief queries the number of nodes in the framework
    */
    size_t num_nodes() const;

    /**
    @brief creates a module task from a framework
    */
    tf::Task composed_of(Framework& framework);

    /**
    @brief sets the name of the framework
    */
    auto& name(const std::string&) ; 

    /**
    @brief queries the name of the framework
    */
    const std::string& name() const ;

  private:
 
    std::string _name;
   
    Graph _graph;

    std::mutex _mtx;
    std::list<Topology*> _topologies;
};

// Constructor
inline Framework::Framework() : FlowBuilder{_graph} {
}

// Destructor
inline Framework::~Framework() {
  assert(_topologies.empty());
}

// Function: num_noces
inline size_t Framework::num_nodes() const {
  return _graph.size();
}

// Function: name
inline auto& Framework::name(const std::string &name) {
  _name = name;
  return *this;
}

// Function: name
inline const std::string& Framework::name() const {
  return _name;
}

// Function: composed_of
inline tf::Task Framework::composed_of(Framework& framework) {
  auto &node = _graph.emplace_back();
  node._module = &framework;
  return Task(node);
}

// Procedure: dump
inline std::string Framework::dump() const {
  std::ostringstream oss;
  dump(oss);
  return oss.str();
}

// Function: dump
inline void Framework::dump(std::ostream& os) const {

  std::stack<const Framework*> stack;
  std::unordered_set<const Framework*> visited; 
  
  os << "digraph Framework_";
  if(_name.empty()) os << 'p' << this;
  else os << _name;
  os << " {\nrankdir=\"LR\";\n";
  
  stack.push(this);
  visited.insert(this);
  
  while(!stack.empty()) {
    
    auto f = stack.top();
    stack.pop();
    
    // create a subgraph field for this framework
    os << "subgraph cluster_";
    if(f->_name.empty()) os << 'p' << f;
    else os << f->_name;
    os << " {\n";

    os << "label=\"Framework_";
    if(f->_name.empty()) os << 'p' << f;
    else os << f->_name;
    os << "\";\n";

    // dump the details of this framework
    for(const auto& n: f->_graph) {
      // regular task
      if(auto module = n._module; !module) {
        n.dump(os);
      }
      // module task
      else {
        os << 'p' << &n << "[shape=box3d, color=blue, label=\"";
        if(n._name.empty()) os << &n;
        else os << n._name;
        os << " (Framework_";
        if(module->_name.empty()) os << module;
        else os << module->_name;
        os << ")\"];\n";

        if(visited.find(module) == visited.end()) {
          visited.insert(module);
          stack.push(module);
        }

        for(const auto s : n._successors) {
          os << 'p' << &n << "->" << 'p' << s << ";\n";
        }
      }
    }
    os << "}\n";
  }

  os << "}\n";
}


}  // end of namespace tf. ---------------------------------------------------

