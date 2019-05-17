#pragma once

#include "taskflow.hpp"

namespace tf {

// ----------------------------------------------------------------------------
  
// class: Topology
class Topology {
  
  friend class Taskflow;
  friend class Executor;
  
  public:

    template <typename P>
    Topology(P&&);

  private:

    std::promise<void> _promise;
    std::shared_future<void> _future {_promise.get_future().share()};

    PassiveVector<Node*> _sources;
    std::atomic<int> _num_sinks {0};
    int _cached_num_sinks {0};
    
    std::function<bool()> _pred {nullptr};
    std::function<void()> _work {nullptr};

    void _bind(Graph& g);
    void _recover_num_sinks();
};

// Constructor
template <typename P>
inline Topology::Topology(P&& p): 
  _pred {std::forward<P>(p)} {
}

// Procedure: _bind
// Re-builds the source links and the sink number for this topology.
inline void Topology::_bind(Graph& g) {
  
  _num_sinks = 0;
  _sources.clear();
  
  // scan each node in the graph and build up the links
  for(auto& node : g) {

    node._topology = this;

    if(node.num_dependents() == 0) {
      _sources.push_back(&node);
    }

    if(node.num_successors() == 0) {
      _num_sinks++;
    }
  }
  _cached_num_sinks = _num_sinks;

}

// Procedure: _recover_num_sinks
inline void Topology::_recover_num_sinks() {
  _num_sinks = _cached_num_sinks;
}

}  // end of namespace tf. ----------------------------------------------------
