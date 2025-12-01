#pragma once

namespace tf {

// ----------------------------------------------------------------------------

/**
@private
*/
class Topology {

  friend class Executor;
  friend class Subflow;
  friend class Runtime;
  friend class NonpreemptiveRuntime;
  friend class Node;

  template <typename T>
  friend class Future;
  
  public:

  Topology(Taskflow&);

  virtual ~Topology() = default;

  bool cancelled() const;

  virtual bool predicate() = 0;
  virtual void on_finish() = 0;

  private:

  Taskflow& _taskflow;

  std::promise<void> _promise;
  
  std::atomic<size_t> _join_counter {0};
  std::atomic<ESTATE::underlying_type> _estate {ESTATE::NONE};

  std::exception_ptr _exception_ptr {nullptr};

  void _carry_out_promise();
};

// Constructor
inline Topology::Topology(Taskflow& tf):
  _taskflow(tf) {
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

// ----------------------------------------------------------------------------


/**
@private
*/
template <typename P, typename C>
class DerivedTopology : public Topology {

  using PredicateType = std::decay_t<P>;
  using CallbackType  = std::decay_t<C>;
  
  public:

  DerivedTopology(Taskflow& tf, P&& pred, C&& clbk) :
    Topology(tf), _pred(std::forward<P>(pred)), _clbk(std::forward<C>(clbk)) {
  }
    
  bool predicate() override final { return _pred(); }
  void on_finish() override final { _clbk(); }   

  private:

  PredicateType _pred;       // predicate, of type bool()
  CallbackType  _clbk;       // callback, of type void()
};

}  // end of namespace tf. ----------------------------------------------------








