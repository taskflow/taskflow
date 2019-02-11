#pragma once

#include "flow_builder.hpp"

namespace tf {

// TODO: document the class

/**
@class Framework 

@brief A reusable task dependency graph.

A framework can be executed by a taskflow object repetitively and thus 
avoids the graph reconstruction overhead. 

*/
class Framework : public FlowBuilder {

  friend class Topology;
  
  template <template<typename...> typename E> 
  friend class BasicTaskflow;

  public:

    Framework();
    
    void dump(std::ostream& ostream) const;
    
    std::string dump() const;

  protected:

    Graph _graph;

  private:

    std::mutex _mtx;
    std::list<Topology*> _topologies;
};

// Constructor
inline Framework::Framework() : FlowBuilder{_graph} {
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

