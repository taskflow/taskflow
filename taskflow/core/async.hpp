#pragma once

#include "executor.hpp"

// https://hackmd.io/@sysprog/concurrency-atomics

namespace tf {

// ----------------------------------------------------------------------------
// Async
// ----------------------------------------------------------------------------

// Function: async
template <typename P, typename F>
auto Executor::async(P&& params, F&& f) {
  _increment_topology();
  return _async(std::forward<P>(params), std::forward<F>(f), nullptr, nullptr);
}

// Function: async
template <typename F>
auto Executor::async(F&& f) {
  return async(DefaultTaskParams{}, std::forward<F>(f));
}

// ----------------------------------------------------------------------------
// Silent Async
// ----------------------------------------------------------------------------

// Function: silent_async
template <typename P, typename F>
void Executor::silent_async(P&& params, F&& f) {
  _increment_topology();
  _silent_async(
    std::forward<P>(params), std::forward<F>(f), nullptr, nullptr
  );
}

// Function: silent_async
template <typename F>
void Executor::silent_async(F&& f) {
  silent_async(DefaultTaskParams{}, std::forward<F>(f));
}

// ----------------------------------------------------------------------------
// Async Helper Methods
// ----------------------------------------------------------------------------

// Procedure: _schedule_async_task
inline void Executor::_schedule_async_task(Node* node) {  
  // Here we don't use _this_worker since _schedule will check if the
  // given worker belongs to this executor.
  (pt::this_worker && pt::this_worker->_executor == this) ? _schedule(*pt::this_worker, node) : 
                                                            _schedule(node);
}

// Procedure: _tear_down_async
inline void Executor::_tear_down_async(Worker& worker, Node* node, Node*& cache) {
  
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

  size_t num_dependents = std::distance(first, last);
  
  AsyncTask task(animate(
    std::forward<P>(params), nullptr, nullptr, num_dependents,
    std::in_place_type_t<Node::DependentAsync>{}, std::forward<F>(func)
  ));
  
  for(; first != last; first++) {
    _process_async_dependent(task._node, *first, num_dependents);
  }

  if(num_dependents == 0) {
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
    
  size_t num_dependents = std::distance(first, last);
  
  // async with runtime: [] (tf::Runtime&) {}
  if constexpr (is_runtime_task_v<F>) {

    std::promise<void> p;
    auto fu{p.get_future()};

    AsyncTask task(animate(
      NSTATE::NONE, ESTATE::ANCHORED, std::forward<P>(params), nullptr, nullptr, num_dependents,
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
      _process_async_dependent(task._node, *first, num_dependents);
    }

    if(num_dependents == 0) {
      _schedule_async_task(task._node);
    }

    return std::make_pair(std::move(task), std::move(fu));
  }
  // async without runtime: [] () {}
  else if constexpr(std::is_invocable_v<F>) {

    using R = std::invoke_result_t<F>;
    std::packaged_task<R()> p(std::forward<F>(func));
    auto fu{p.get_future()};

    AsyncTask task(animate(
      std::forward<P>(params), nullptr, nullptr, num_dependents,
      std::in_place_type_t<Node::DependentAsync>{},
      [p=make_moc(std::move(p))] () mutable { p.object(); }
    ));

    for(; first != last; first++) {
      _process_async_dependent(task._node, *first, num_dependents);
    }

    if(num_dependents == 0) {
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

// Procedure: _process_async_dependent
inline void Executor::_process_async_dependent(
  Node* node, tf::AsyncTask& task, size_t& num_dependents
) {

  auto& state = std::get_if<Node::DependentAsync>(&(task._node->_handle))->state;

  add_successor:

  auto target = ASTATE::UNFINISHED;
  
  // acquires the lock
  if(state.compare_exchange_weak(target, ASTATE::LOCKED,
                                 std::memory_order_acq_rel,
                                 std::memory_order_acquire)) {
    task._node->_successors.push_back(node);
    state.store(ASTATE::UNFINISHED, std::memory_order_release);
  }
  // dep's state is FINISHED, which means dep finished its callable already
  // thus decrement the node's join counter by 1
  else if (target == ASTATE::FINISHED) {
    num_dependents = node->_join_counter.fetch_sub(1, std::memory_order_acq_rel) - 1;
  }
  // another worker adding its async task to the same successors of this node
  else {
    goto add_successor;
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
  for(size_t i=0; i<node->_successors.size(); ++i) {
    if(auto s = node->_successors[i]; 
      s->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1
    ) {
      _update_cache(worker, cache, s);
    }
  }
  
  // now the executor no longer needs to retain ownership
  if(handle->use_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    recycle(node);
  }

  _decrement_topology();
}





}  // end of namespace tf -----------------------------------------------------

