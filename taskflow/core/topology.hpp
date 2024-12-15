#pragma once

namespace tf {

// ----------------------------------------------------------------------------

class TopologyBase {

};

// class: Topology
class Topology {

  friend class Executor;
  friend class Subflow;
  friend class Runtime;
  friend class Node;

  template <typename T>
  friend class Future;
  
  public:

    template <typename P, typename C>
    Topology(Taskflow&, P&&, C&&);

    bool cancelled() const;

  private:

    Taskflow& _taskflow;

    std::promise<void> _promise;
    
    size_t _num_sources {0};

    std::function<bool()> _pred;
    std::function<void()> _call;

    std::atomic<size_t> _join_counter {0};
    std::atomic<ESTATE::underlying_type> _estate {ESTATE::NONE};

    std::exception_ptr _exception_ptr {nullptr};

    void _carry_out_promise();
};

// Constructor
template <typename P, typename C>
Topology::Topology(Taskflow& tf, P&& p, C&& c):
  _taskflow(tf),
  _pred {std::forward<P>(p)},
  _call {std::forward<C>(c)} {
}

// Procedure
inline void Topology::_carry_out_promise() {
  if(_exception_ptr) {
    auto e = _exception_ptr;
    _exception_ptr = nullptr;
    _promise.set_exception(e);
  }
  else {
    _promise.set_value();
  }
}

// Function: cancelled
inline bool Topology::cancelled() const {
  return _estate.load(std::memory_order_relaxed) & ESTATE::CANCELLED;
}

}  // end of namespace tf. ----------------------------------------------------
