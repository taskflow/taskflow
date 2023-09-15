#pragma once

namespace tf {

// ----------------------------------------------------------------------------

// class: Topology
class Topology {

  friend class Executor;
  friend class Runtime;
  friend class Node;

  template <typename T>
  friend class Future;

  public:

    template <typename P, typename C>
    Topology(Taskflow&, P&&, C&&);

  private:

    Taskflow& _taskflow;

    std::promise<void> _promise;

    SmallVector<Node*> _sources;

    std::function<bool()> _pred;
    std::function<void()> _call;

    std::atomic<size_t> _join_counter {0};
    std::atomic<bool> _is_cancelled { false };

    std::exception_ptr _exception;

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
  _exception ? _promise.set_exception(_exception) : _promise.set_value();
}

}  // end of namespace tf. ----------------------------------------------------
