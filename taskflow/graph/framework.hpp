#pragma once

#include "flow_builder.hpp"

// TODO:
// clear the graph only at the beginning of each run.

namespace tf {

// TODO: document the class

// Class: Framework
class Framework : public FlowBuilder {

  friend class Topology;
  
  template <template<typename...> typename E> 
  friend class BasicTaskflow;

  public:

    Framework();
    
    // TODO
    void dump(std::ostream& ostream) const;
    
    // TODO
    std::string dump() const;

  protected:

    Graph _graph;

  private:

    std::mutex _mtx;
    std::list<Topology*> _topologies;
    std::vector<size_t> _dependents;
};

// Constructor
inline Framework::Framework() : FlowBuilder{_graph} {
}

// Procedure: dump



};  // end of namespace tf. ---------------------------------------------------

