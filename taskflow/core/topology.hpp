#pragma once

namespace tf {

// ----------------------------------------------------------------------------

// class: TopologyBase
class TopologyBase {
  
  friend class Executor;
  friend class Node;
  
  template <typename T>
  friend class Future;

  protected:

  bool _is_cancelled { false };
};

// ----------------------------------------------------------------------------

// class: AsyncTopology
class AsyncTopology : public TopologyBase {
};

// ----------------------------------------------------------------------------
  
// class: Topology
class Topology : public TopologyBase {
  
  friend class Executor;

  public:

    template <typename P, typename C>
    Topology(Taskflow&, P&&, C&&);

  private:

    Taskflow& _taskflow;

    std::promise<void> _promise;

    std::vector<Node*> _sources;

    std::function<bool()> _pred;
    std::function<void()> _call;

    std::atomic<size_t> _join_counter {0};
};

// Constructor
template <typename P, typename C>
Topology::Topology(Taskflow& tf, P&& p, C&& c): 
  _taskflow(tf),
  _pred {std::forward<P>(p)},
  _call {std::forward<C>(c)} {
}

}  // end of namespace tf. ----------------------------------------------------
