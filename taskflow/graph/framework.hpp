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

  private:
    
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

// Procedure: dump
inline void Framework::dump(std::ostream& os) const {
  os << "digraph Framework {\n";
  for(const auto& n: _graph) {
    n.dump(os);
  }
  os << "}\n";
}

// Function: dump
inline std::string Framework::dump() const { 
  std::ostringstream os;
  dump(os);
  return os.str();
}

}  // end of namespace tf. ---------------------------------------------------

