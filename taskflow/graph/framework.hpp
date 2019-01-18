#pragma once

#include "flow_builder.hpp"

namespace tf {

// TODO: document the class

// Class: Framework
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
    std::vector<size_t> _dependents;
    Node* _last_target {nullptr};   
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


};  // end of namespace tf. ---------------------------------------------------

