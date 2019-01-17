#pragma once

#include "flow_builder.hpp"

namespace tf {

// Class: Framework
class Framework : public FlowBuilder {
  
  template <template<typename...> typename E> 
  friend class BasicTaskflow;

  public:

    inline Framework();

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


};  // end of namespace tf. ---------------------------------------------------

