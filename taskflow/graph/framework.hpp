#pragma once

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
    @brief gets the name of the framework
    */
    const std::string& name() const ;

  private:
 
    std::string _name;
   
    Graph _graph;

    std::mutex _mtx;
    std::list<Topology*> _topologies;

    std::string _addr_to_string() const;

    void _dump(std::ostream& ostream, const Framework&,  std::vector<Framework*>&, std::unordered_set<Framework*>&) const;
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


// Function: _addr_to_string
inline std::string Framework::_addr_to_string() const {
  std::stringstream ss;
  ss << 'p' << this;
  return ss.str();
}

// Procedure: dump
inline void Framework::dump(std::ostream& os) const {
  os << dump();
}

// TODO: 
// 1. check the bug from your composition_1 example?
// 2. use the format: F2 (pointer value)

// Procedure: dump
inline void Framework::_dump(
  std::ostream& os, const Framework& framework,
  std::vector<Framework*>& unseen, std::unordered_set<Framework*>& seen) const {

  for(const auto& n: framework._graph) {
    if(n._module == nullptr) {
      n.dump(os);
    }
    else {
      os << 'p' << &n << "[shape=oval, penwidth=5, color=blue, label = \"Module_";
      if(n._name.empty()) os << &n;
      else os << n._name;
      os << " (Framework_";
      if(n._module->_name.empty()) os << n._module;
      else os << n._module->_name;
      os << ")\"];\n";

      if(seen.find(n._module) == seen.end()) {
        seen.insert(n._module);
        unseen.emplace_back(n._module);
      }

      for(const auto s : n._successors) {
        os << 'p' << &n << "->" << 'p' << s << ";\n";
      }
    }
  }
}


// Function: dump
inline std::string Framework::dump() const {
  std::ostringstream os;

  std::unordered_set<Framework*> seen;
  std::vector<Framework*> unseen;
  size_t cursor {0};

  os << "digraph Framework_" << (_name.empty() ? _addr_to_string() : _name) << " {\n";

  {
    os << "subgraph cluster_";
    os << (_name.empty() ? _addr_to_string() : _name) << " {\n";

    os << "label = \"Framework_" << (_name.empty() ? _addr_to_string() : _name) << "\";\n";
    //os << "fontcolor = royalblue3;\n";
    //os << "fontsize = 35;\n";
    //os << "subgraph cluster_Top{\n";
    //os << "label = \"Top\";\n";
    os << "style = \"bold, rounded\";\n";
    _dump(os, *this, unseen, seen);
    os << "}\n";
  }

  cursor = unseen.size();
  for(auto i=0u; i<cursor; i++) {
    os << "subgraph cluster_";
    os << (unseen[i]->_name.empty() ? unseen[i]->_addr_to_string() : unseen[i]->_name) << " {\n";
    os << "label = \"Framework_" << (unseen[i]->_name.empty() ? unseen[i]->_addr_to_string() : unseen[i]->_name) << "\";\n";
    //os << "fontcolor = royalblue3;\n";
    //os << "fontsize = 35;\n";
    //os << "style = filled;\n";
    //os << "color = beige;\n";
    _dump(os, *unseen[i], unseen, seen);
    os << "}\n";
    cursor = unseen.size();
  }

  os << "}\n";
  return os.str();
}


}  // end of namespace tf. ---------------------------------------------------

