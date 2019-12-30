#pragma once

//#include "taskflow.hpp"

namespace tf {

// ----------------------------------------------------------------------------
  
// class: Topology
class Topology {
  
  friend class Taskflow;
  friend class Executor;
  
  public:

    template <typename P, typename C>
    Topology(Taskflow&, P&&, C&&);
    
  private:

    Taskflow& _taskflow;

    std::promise<void> _promise;

    PassiveVector<Node*> _sources;
    
    std::function<bool()> _pred;
    std::function<void()> _call;

    void _bind(Graph& g);

    std::atomic<int> _num_dependents {0};
};

// Constructor
template <typename P, typename C>
inline Topology::Topology(Taskflow& tf, P&& p, C&& c): 
  _taskflow(tf),
  _pred {std::forward<P>(p)},
  _call {std::forward<C>(c)} {
}

// Procedure: _bind
// Re-builds the source links and the sink number for this topology.
inline void Topology::_bind(Graph& g) {
  
  _sources.clear();

  PassiveVector<Node*> condition_nodes;
  
  // scan each node in the graph and build up the links
  for(auto& node : g.nodes()) {

    node->clear_status();
    node->_topology = this;

    if(node->num_dependents() == 0) {
      _sources.push_back(node.get());
    }

    // TODO: Merge with the loop below?
    if(node->_work.index() == Node::CONDITION) {
      condition_nodes.push_back(node.get());
    }

    // Reset each node's num_dependents
    node->_num_dependents.store(node->_dependents.size(), std::memory_order_relaxed);
  }

  // We need to deduct the condition predecessors in impure case nodes
  for(auto& n: condition_nodes) {
    for(auto& s: n->_successors) {
      s->_num_dependents.fetch_sub(1, std::memory_order_relaxed);
      s->set_branch();
    }
  }
}



}  // end of namespace tf. ----------------------------------------------------
