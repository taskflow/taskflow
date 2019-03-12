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

    // TODO: add doc 
    /**
    @brief create a module task from a framework
    */
    tf::Task composed_of(Framework& framework);

    /**
    @brief set the name of the framework
    */
    auto& name(const std::string&) ; 

    /**
    @brief get the name of the framework
    */
    const std::string& name() const ;

  private:
 
    std::string _name;
   
    Graph _graph;

    std::mutex _mtx;
    std::list<Topology*> _topologies;

    std::string _addr_to_string() const;
    void _dump(std::ostream& ostream, std::unordered_set<Framework*>&, std::unordered_set<Framework*>&) const;
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
  node.set_module();
  node._module = &framework;
  return Task(node);
}


// Function: _addr_to_string
inline std::string Framework::_addr_to_string() const {
  std::stringstream ss;
  ss << this;
  return ss.str();
}

// Procedure: dump
inline void Framework::dump(std::ostream& os) const {
  os << "digraph Framework {\n";
  for(const auto& n: _graph) {
    n.dump(os);
  }
  os << "}\n";
}


// Procedure: dump
inline void Framework::_dump(
  std::ostream& os, std::unordered_set<Framework*>& next_level, std::unordered_set<Framework*>& seen) const {

  for(const auto& n: _graph) {
    if(!n.is_module()) {
      n.dump(os);
    }
    else {
      if(seen.find(n._module) == seen.end()) {
        seen.insert(n._module);
        next_level.insert(n._module);
      }

      const auto name =  n._module->name().empty() ? n._module->_addr_to_string() : n._module->name();
      // prefix the pointer with '_' to avoid a dummy node in graph
      os << "_" << &n
         << " [shape=oval, penwidth=5, color=forestgreen, label=\"" << &n << " (" << name << ')'
         << "\"]"
         << ";\n";

      for(const auto s : n._successors) {

        //if(_name.empty()) os << '\"' << this << '\"';
        //else os << std::quoted(_name);
       os << "_" << &n;

        os << " -> ";

        if(!s->is_module()) {
          if(s->name().empty()) os << '\"' << s << '\"';
          else os << std::quoted(s->name());
        }
        else {
          os << "_" << s;
        }

        os << ";\n";
      }

      if(n._subgraph && !n._subgraph->empty()) {

        os << "subgraph cluster_";
        if(_name.empty()) os << this;
        else os << _name;
        os << " {\n";

        os << "label = \"Subflow_";
        if(_name.empty()) os << this;
        else os << _name;

        os << "\";\n" << "color=blue\n";
        for(const auto& n : *(n._subgraph)) {
          n.dump(os);
        }
        os << "}\n";
      }
    }
  }
}


// Function: dump
inline std::string Framework::dump() const {
  std::ostringstream os;

  std::unordered_set<Framework*> seen;
  std::unordered_set<Framework*> cur_level;
  std::unordered_set<Framework*> next_level;
  size_t level {1};

  os << "digraph Framework_" << (_name.empty() ? _addr_to_string() : _name) << " {\n";
  {
    os << "subgraph cluster_Top{\n";
    os << "label = \"Top\";\n";
    os << "style = \"bold\";\n";
    _dump(os, cur_level, seen);
    os << "}\n";
  }
  while(!cur_level.empty()) {
    os << "subgraph cluster_" << level << "{\n";
    os << "label = " << "\"Level " << level << "\"" << ";\n";
    os << "style = \"rounded\";\n";
    //os << "fontsize = 100\n";

    next_level.clear();
    for(auto& f: cur_level) {
      const auto name = (f->_name.empty() ? f->_addr_to_string() : f->_name);
      os << "subgraph cluster_" << name << "_" << level << "{\n";
      os << "label = " << "\"Framework " << name << "\"" << ";\n";
      f->_dump(os, next_level, seen);
      os << "}\n";
    }
    level ++;
    std::swap(cur_level, next_level);

    os << "}\n";
  }
  os << "}\n";
  return os.str();
}


}  // end of namespace tf. ---------------------------------------------------

