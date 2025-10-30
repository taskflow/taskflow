#pragma once

#include "executor.hpp"
#include "runtime.hpp"

// https://hackmd.io/@sysprog/concurrency-atomics

namespace tf {

// ----------------------------------------------------------------------------
// Async Helper Methods
// ----------------------------------------------------------------------------

// Procedure: _schedule_async_task
TF_FORCE_INLINE void Executor::_schedule_async_task(Node* node) {  
  if(auto w = this_worker(); w) {
    _schedule(*w, node);
  }
  else{
    _schedule(node);
  }
}

// Procedure: _tear_down_async
inline void Executor::_tear_down_async(Worker& worker, Node* node, Node*& cache) {
  
  // node->_topology  |  node->_parent  |  secenario
  // nullptr          |  nullptr        |  exe.async();
  // nullptr          |  0x---          |  exe.async([](Runtime rt){ rt.async(); });
  // 0x---            |  nullptr        |  ?
  // 0x---            |  0x---          |  tf.emplace([](Runtime& rt){ rt.async(); });

  // from executor
  if(auto parent = node->_parent; parent == nullptr) {
    _decrement_topology();
  }
  // from runtime
  else {
    auto state = parent->_nstate;
    if(parent->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      if(state & NSTATE::PREEMPTED) {
        _update_cache(worker, cache, parent);
      }
    }
  }
  recycle(node);
}

// ----------------------------------------------------------------------------
// Async
// ----------------------------------------------------------------------------

// Function: async
template <typename F>
auto Executor::async(F&& f) {
  return async(DefaultTaskParams{}, std::forward<F>(f));
}

// Function: async
template <typename P, typename F>
auto Executor::async(P&& params, F&& f) {
  _increment_topology();
  return _async(std::forward<P>(params), std::forward<F>(f), nullptr, nullptr);
}

// Function: _async
template <typename P, typename F>
auto Executor::_async(P&& params, F&& f, Topology* tpg, Node* parent) {
  
  // async task with runtime: [] (tf::Runtime&) -> void {}
  if constexpr (is_runtime_task_v<F>) {

    std::promise<void> p;
    auto fu{p.get_future()};
    
    _schedule_async_task(animate(
      NSTATE::NONE, ESTATE::ANCHORED, std::forward<P>(params), tpg, parent, 0, 
      std::in_place_type_t<Node::Async>{}, 
      [p=MoC{std::move(p)}, f=std::forward<F>(f)](Runtime& rt, bool reentered) mutable { 
        if(!reentered) {
          f(rt);
        }
        else {
          auto& eptr = rt._parent->_exception_ptr;
          eptr ? p.object.set_exception(eptr) : p.object.set_value();
        }
      }
    ));
    return fu;
  }
  // async task with closure: [] () -> auto { return ... }
  else if constexpr (std::is_invocable_v<F>){
    using R = std::invoke_result_t<F>;
    std::packaged_task<R()> p(std::forward<F>(f));
    auto fu{p.get_future()};
    _schedule_async_task(animate(
      NSTATE::NONE, ESTATE::NONE, std::forward<P>(params), tpg, parent, 0, 
      std::in_place_type_t<Node::Async>{}, 
      [p=make_moc(std::move(p))]() mutable { p.object(); }
    ));
    return fu;
  }
  else {
    static_assert(dependent_false_v<F>, 
      "invalid async target - must be one of the following types:\n\
      (1) [] (tf::Runtime&) -> void {}\n\
      (2) [] () -> auto { ... return ... }\n"
    );
  }
}


// ----------------------------------------------------------------------------
// Silent Async
// ----------------------------------------------------------------------------

// Function: silent_async
template <typename F>
void Executor::silent_async(F&& f) {
  silent_async(DefaultTaskParams{}, std::forward<F>(f));
}

// Function: silent_async
template <typename P, typename F>
void Executor::silent_async(P&& params, F&& f) {
  _increment_topology();
  _silent_async(std::forward<P>(params), std::forward<F>(f), nullptr, nullptr);
}

// Function: _silent_async
template <typename P, typename F>
void Executor::_silent_async(P&& params, F&& f, Topology* tpg, Node* parent) {
  // silent task 
  if constexpr (is_runtime_task_v<F> || is_static_task_v<F>) {
    _schedule_async_task(animate(
      NSTATE::NONE, ESTATE::NONE, std::forward<P>(params), tpg, parent, 0,
      std::in_place_type_t<Node::Async>{}, std::forward<F>(f)
    ));
  }
  // invalid silent async target
  else {
    static_assert(dependent_false_v<F>, 
      "invalid silent_async target - must be one of the following types:\n\
      (1) [] (tf::Runtime&) -> void {}\n\
      (2) [] () -> void { ... }\n"
    );
  }
}

// ----------------------------------------------------------------------------
// Silent Dependent Async
// ----------------------------------------------------------------------------

// Function: silent_dependent_async
#if __cplusplus >= TF_CPP20
template <typename F, typename... Tasks>
requires all_same_v<AsyncTask, std::decay_t<Tasks>...>
#else
template <typename F, typename... Tasks,
  std::enable_if_t<all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>*
>
#endif
tf::AsyncTask Executor::silent_dependent_async(F&& func, Tasks&&... tasks) {
  return silent_dependent_async(
    DefaultTaskParams{}, std::forward<F>(func), std::forward<Tasks>(tasks)...
  );
}

// Function: silent_dependent_async
#if __cplusplus >= TF_CPP20
template <typename P, typename F, typename... Tasks>
requires is_task_params_v<P> && all_same_v<AsyncTask, std::decay_t<Tasks>...>
#else
template <typename P, typename F, typename... Tasks,
  std::enable_if_t<is_task_params_v<P> && all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>*
>
#endif
tf::AsyncTask Executor::silent_dependent_async(
  P&& params, F&& func, Tasks&&... tasks 
){
  std::array<AsyncTask, sizeof...(Tasks)> array = { std::forward<Tasks>(tasks)... };
  return silent_dependent_async(
    std::forward<P>(params), std::forward<F>(func), array.begin(), array.end()
  );
}

// Function: silent_dependent_async
#if __cplusplus >= TF_CPP20
template <typename F, typename I>
requires (!std::is_same_v<std::decay_t<I>, AsyncTask>)
#else
template <typename F, typename I,
  std::enable_if_t<!std::is_same_v<std::decay_t<I>, AsyncTask>, void>*
>
#endif
tf::AsyncTask Executor::silent_dependent_async(F&& func, I first, I last) {
  return silent_dependent_async(DefaultTaskParams{}, std::forward<F>(func), first, last);
}

// Function: silent_dependent_async
#if __cplusplus >= TF_CPP20
template <typename P, typename F, typename I>
requires (is_task_params_v<P> && !std::is_same_v<std::decay_t<I>, AsyncTask>)
#else
template <typename P, typename F, typename I,
  std::enable_if_t<is_task_params_v<P> && !std::is_same_v<std::decay_t<I>, AsyncTask>, void>*
>
#endif
tf::AsyncTask Executor::silent_dependent_async(
  P&& params, F&& func, I first, I last
) {
  _increment_topology();
  return _silent_dependent_async(
    std::forward<P>(params), std::forward<F>(func), first, last, nullptr, nullptr
  );
}

// Function: silent_dependent_async
template <typename P, typename F, typename I,
  std::enable_if_t<is_task_params_v<P> && !std::is_same_v<std::decay_t<I>, AsyncTask>, void>*
>
auto Executor::_silent_dependent_async(
  P&& params, F&& func, I first, I last, Topology* tpg, Node* parent
) {

  size_t num_predecessors = std::distance(first, last);
  
  AsyncTask task(animate(
    NSTATE::NONE, ESTATE::NONE, std::forward<P>(params), tpg, parent, num_predecessors,
    std::in_place_type_t<Node::DependentAsync>{}, std::forward<F>(func)
  ));
  
  for(; first != last; first++) {
    _process_dependent_async(task._node, *first, num_predecessors);
  }

  if(num_predecessors == 0) {
    _schedule_async_task(task._node);
  }

  return task;
}

// ----------------------------------------------------------------------------
// Dependent Async
// ----------------------------------------------------------------------------

// Function: dependent_async
#if __cplusplus >= TF_CPP20
template <typename F, typename... Tasks>
requires all_same_v<AsyncTask, std::decay_t<Tasks>...>
#else
template <typename F, typename... Tasks,
  std::enable_if_t<all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>*
>
#endif
auto Executor::dependent_async(F&& func, Tasks&&... tasks) {
  return dependent_async(DefaultTaskParams{}, std::forward<F>(func), std::forward<Tasks>(tasks)...);
}

// Function: dependent_async
#if __cplusplus >= TF_CPP20
template <typename P, typename F, typename... Tasks>
requires is_task_params_v<P> && all_same_v<AsyncTask, std::decay_t<Tasks>...>
#else
template <typename P, typename F, typename... Tasks,
  std::enable_if_t<is_task_params_v<P> && all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>*
>
#endif
auto Executor::dependent_async(P&& params, F&& func, Tasks&&... tasks) {
  std::array<AsyncTask, sizeof...(Tasks)> array = { std::forward<Tasks>(tasks)... };
  return dependent_async(
    std::forward<P>(params), std::forward<F>(func), array.begin(), array.end()
  );
}

// Function: dependent_async
#if __cplusplus >= TF_CPP20
template <typename F, typename I>
requires (!std::is_same_v<std::decay_t<I>, AsyncTask>)
#else
template <typename F, typename I,
  std::enable_if_t<!std::is_same_v<std::decay_t<I>, AsyncTask>, void>*
>
#endif
auto Executor::dependent_async(F&& func, I first, I last) {
  return dependent_async(DefaultTaskParams{}, std::forward<F>(func), first, last);
}

// Function: dependent_async
#if __cplusplus >= TF_CPP20
template <typename P, typename F, typename I>
requires (is_task_params_v<P> && !std::is_same_v<std::decay_t<I>, AsyncTask>)
#else
template <typename P, typename F, typename I,
  std::enable_if_t<is_task_params_v<P> && !std::is_same_v<std::decay_t<I>, AsyncTask>, void>*
>
#endif
auto Executor::dependent_async(P&& params, F&& func, I first, I last) {
  _increment_topology();
  return _dependent_async(std::forward<P>(params), std::forward<F>(func), first, last, nullptr, nullptr);
}

// Function: dependent_async
template <typename P, typename F, typename I,
  std::enable_if_t<is_task_params_v<P> && !std::is_same_v<std::decay_t<I>, AsyncTask>, void>*
>
auto Executor::_dependent_async(P&& params, F&& func, I first, I last, Topology* tpg, Node* parent) {
    
  size_t num_predecessors = std::distance(first, last);
  
  // async with runtime: [] (tf::Runtime&) -> void {}
  if constexpr (is_runtime_task_v<F>) {

    std::promise<void> p;
    auto fu{p.get_future()};

    AsyncTask task(animate(
      NSTATE::NONE, ESTATE::ANCHORED, std::forward<P>(params), tpg, parent, num_predecessors,
      std::in_place_type_t<Node::DependentAsync>{},
      [p=MoC{std::move(p)}, f=std::forward<F>(func)] (tf::Runtime& rt, bool reentered) mutable { 
        if(!reentered) {
          f(rt); 
        }
        else {
          auto& eptr = rt._parent->_exception_ptr;
          eptr ? p.object.set_exception(eptr) : p.object.set_value();
        }
      }
    ));

    for(; first != last; first++) {
      _process_dependent_async(task._node, *first, num_predecessors);
    }

    if(num_predecessors == 0) {
      _schedule_async_task(task._node);
    }

    return std::make_pair(std::move(task), std::move(fu));
  }
  // async without runtime: [] () -> auto { return ... }
  else if constexpr(std::is_invocable_v<F>) {

    using R = std::invoke_result_t<F>;
    std::packaged_task<R()> p(std::forward<F>(func));
    auto fu{p.get_future()};

    AsyncTask task(animate(
      NSTATE::NONE, ESTATE::NONE, std::forward<P>(params), tpg, parent, num_predecessors,
      std::in_place_type_t<Node::DependentAsync>{},
      [p=make_moc(std::move(p))] () mutable { p.object(); }
    ));

    for(; first != last; first++) {
      _process_dependent_async(task._node, *first, num_predecessors);
    }

    if(num_predecessors == 0) {
      _schedule_async_task(task._node);
    }

    return std::make_pair(std::move(task), std::move(fu));
  }
  else {
    static_assert(dependent_false_v<F>, "invalid async callable");
  }
}

// ----------------------------------------------------------------------------
// Dependent Async Helper Functions
// ----------------------------------------------------------------------------

// Procedure: _process_dependent_async
inline void Executor::_process_dependent_async(
  Node* node, tf::AsyncTask& task, size_t& num_predecessors
) {

  // special case: the task is not associated with any dependent-async task
  if(task.empty()) {
    num_predecessors = node->_join_counter.fetch_sub(1, std::memory_order_acq_rel) - 1;
    return;
  }

  auto& state = std::get_if<Node::DependentAsync>(&(task._node->_handle))->state;

  while (true) {

    auto target = ASTATE::UNFINISHED;

    // Try to acquire the lock
    if (state.compare_exchange_strong(target, ASTATE::LOCKED, 
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
      task._node->_edges.push_back(node);
      state.store(ASTATE::UNFINISHED, std::memory_order_release);
      break;
    }

    // If already finished, decrement the join counter
    if (target == ASTATE::FINISHED) {
      num_predecessors = node->_join_counter.fetch_sub(1, std::memory_order_acq_rel) - 1;
      break;
    }

    // If locked by another worker, retry
  }
}

// Procedure: _tear_down_dependent_async
inline void Executor::_tear_down_dependent_async(Worker& worker, Node* node, Node*& cache) {

  auto handle = std::get_if<Node::DependentAsync>(&(node->_handle));

  // this async task comes from Executor
  auto target = ASTATE::UNFINISHED;

  while(!handle->state.compare_exchange_weak(target, ASTATE::FINISHED,
                                             std::memory_order_acq_rel,
                                             std::memory_order_relaxed)) {
    target = ASTATE::UNFINISHED;
  }
  
  // spawn successors whenever their dependencies are resolved
  for(size_t i=0; i<node->_edges.size(); ++i) {
    if(auto s = node->_edges[i]; 
      s->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1
    ) {
      _update_cache(worker, cache, s);
    }
  }
  
  // node->_topology  |  node->_parent  |  secenario
  // nullptr          |  nullptr        |  exe.async();
  // nullptr          |  0x---          |  exe.async([](Runtime rt){ rt.async(); });
  // 0x---            |  nullptr        |  ?
  // 0x---            |  0x---          |  tf.emplace([](Runtime& rt){ rt.async(); });

  // from executor
  if(auto parent = node->_parent; parent == nullptr) {
    _decrement_topology();
  }
  // from runtime
  else {
    auto state = parent->_nstate;
    if(parent->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      if(state & NSTATE::PREEMPTED) {
        _update_cache(worker, cache, parent);
      }
    }
  }
  
  // now the executor no longer needs to retain ownership
  if(handle->use_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    recycle(node);
  }
}





}  // end of namespace tf -----------------------------------------------------

