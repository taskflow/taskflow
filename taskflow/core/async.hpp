#pragma once

#include "executor.hpp"

// https://hackmd.io/@sysprog/concurrency-atomics

namespace tf {

// ----------------------------------------------------------------------------
// Async
// ----------------------------------------------------------------------------

// Function: async
template <typename F>
auto Executor::async(const std::string& name, F&& f) {

  _increment_topology();

  using R = std::invoke_result_t<std::decay_t<F>>;

  std::promise<R> p;
  auto fu{p.get_future()};

  auto node = node_pool.animate(
    name, 0, nullptr, nullptr, 0,
    std::in_place_type_t<Node::Async>{}, 
    _make_promised_async(std::move(p), std::forward<F>(f))
  );

  _schedule_async_task(node);

  return fu;
}

// Function: async
template <typename F>
auto Executor::async(F&& f) {
  return async("", std::forward<F>(f));
}

// ----------------------------------------------------------------------------
// Silent Async
// ----------------------------------------------------------------------------

// Function: silent_async
template <typename F>
void Executor::silent_async(const std::string& name, F&& f) {

  _increment_topology();

  auto node = node_pool.animate(
    name, 0, nullptr, nullptr, 0,
    std::in_place_type_t<Node::Async>{}, std::forward<F>(f)
  );

  _schedule_async_task(node);
}

// Function: silent_async
template <typename F>
void Executor::silent_async(F&& f) {
  silent_async("", std::forward<F>(f));
}

// ----------------------------------------------------------------------------
// Async Helper Methods
// ----------------------------------------------------------------------------

// Function: _make_promised_async
template <typename R, typename F>
auto Executor::_make_promised_async(std::promise<R>&& p, F&& func) {
  return [p=make_moc(std::move(p)), func=std::forward<F>(func)]() mutable {
    if constexpr(std::is_same_v<R, void>) {
      func();
      p.object.set_value();
    }
    else {
      p.object.set_value(func());
    }
  };
}
  
// Procedure: _schedule_async_task
inline void Executor::_schedule_async_task(Node* node) {  
  if(auto w = _this_worker(); w) {
    _schedule(*w, node);
  }
  else{
    _schedule(node);
  }
}

// Procedure: _tear_down_async
inline void Executor::_tear_down_async(Node* node) {
  // from runtime
  if(node->_parent) {
    node->_parent->_join_counter.fetch_sub(1, std::memory_order_release);
  }
  // from executor
  else {
    _decrement_topology_and_notify();
  }
  node_pool.recycle(node);
}

// ----------------------------------------------------------------------------
// Silent Dependent Async
// ----------------------------------------------------------------------------

// Function: silent_dependent_async
template <typename F, typename... Tasks,
  std::enable_if_t<all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>*
>
tf::AsyncTask Executor::silent_dependent_async(F&& func, Tasks&&... tasks) {
  return silent_dependent_async("", std::forward<F>(func), std::forward<Tasks>(tasks)...);
}

// Function: silent_dependent_async
template <typename F, typename... Tasks,
  std::enable_if_t<all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>*
>
tf::AsyncTask Executor::silent_dependent_async(
  const std::string& name, F&& func, Tasks&&... tasks 
){

  _increment_topology();

  size_t num_dependents = sizeof...(Tasks);
  
  std::shared_ptr<Node> node(
    node_pool.animate(
      name, 0, nullptr, nullptr, num_dependents,
      std::in_place_type_t<Node::DependentAsync>{}, std::forward<F>(func)
    ),
    [&](Node* ptr){ node_pool.recycle(ptr); }
  );
  
  {
    std::scoped_lock lock(_asyncs_mutex);
    _asyncs.insert(node);
  }
  
  if constexpr(sizeof...(Tasks) > 0) {
    (_process_async_dependent(node.get(), tasks, num_dependents), ...);
  }

  if(num_dependents == 0) {
    _schedule_async_task(node.get());
  }

  return AsyncTask(std::move(node));
}

// Function: silent_dependent_async
template <typename F, typename I,
  std::enable_if_t<!std::is_same_v<std::decay_t<I>, AsyncTask>, void>*
>
tf::AsyncTask Executor::silent_dependent_async(F&& func, I first, I last) {
  return silent_dependent_async("", std::forward<F>(func), first, last);
}

// Function: silent_dependent_async
template <typename F, typename I,
  std::enable_if_t<!std::is_same_v<std::decay_t<I>, AsyncTask>, void>*
>
tf::AsyncTask Executor::silent_dependent_async(
  const std::string& name, F&& func, I first, I last
) {

  _increment_topology();

  size_t num_dependents = std::distance(first, last);
  
  std::shared_ptr<Node> node(
    node_pool.animate(
      name, 0, nullptr, nullptr, num_dependents,
      std::in_place_type_t<Node::DependentAsync>{}, std::forward<F>(func)
    ),
    [&](Node* ptr){ node_pool.recycle(ptr); }
  );
  
  {
    std::scoped_lock lock(_asyncs_mutex);
    _asyncs.insert(node);
  }
  
  for(; first != last; first++){
    _process_async_dependent(node.get(), *first, num_dependents);
  }

  if(num_dependents == 0) {
    _schedule_async_task(node.get());
  }

  return AsyncTask(std::move(node));
}

// ----------------------------------------------------------------------------
// Dependent Async
// ----------------------------------------------------------------------------

// Function: dependent_async
template <typename F, typename... Tasks,
  std::enable_if_t<all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>*
>
auto Executor::dependent_async(F&& func, Tasks&&... tasks) {
  return dependent_async("", std::forward<F>(func), std::forward<Tasks>(tasks)...);
}

// Function: dependent_async
template <typename F, typename... Tasks,
  std::enable_if_t<all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>*
>
auto Executor::dependent_async(
  const std::string& name, F&& func, Tasks&&... tasks 
) {
  
  _increment_topology();
  
  using R = std::invoke_result_t<std::decay_t<F>>;

  std::promise<R> p;
  auto fu{p.get_future()};

  size_t num_dependents = sizeof...(tasks);

  std::shared_ptr<Node> node(
    node_pool.animate(
      name, 0, nullptr, nullptr, num_dependents,
      std::in_place_type_t<Node::DependentAsync>{},
      _make_promised_async(std::move(p), std::forward<F>(func))
    ),
    [&](Node* ptr){ node_pool.recycle(ptr); }
  );
  
  {
    std::scoped_lock lock(_asyncs_mutex);
    _asyncs.insert(node);
  }
  
  if constexpr(sizeof...(Tasks) > 0) {
    (_process_async_dependent(node.get(), tasks, num_dependents), ...);
  }

  if(num_dependents == 0) {
    _schedule_async_task(node.get());
  }

  return std::make_pair(AsyncTask(std::move(node)), std::move(fu));
}

// Function: dependent_async
template <typename F, typename I,
  std::enable_if_t<!std::is_same_v<std::decay_t<I>, AsyncTask>, void>*
>
auto Executor::dependent_async(F&& func, I first, I last) {
  return dependent_async("", std::forward<F>(func), first, last);
}

// Function: dependent_async
template <typename F, typename I,
  std::enable_if_t<!std::is_same_v<std::decay_t<I>, AsyncTask>, void>*
>
auto Executor::dependent_async(
  const std::string& name, F&& func, I first, I last
) {
  
  _increment_topology();
  
  using R = std::invoke_result_t<std::decay_t<F>>;

  std::promise<R> p;
  auto fu{p.get_future()};

  size_t num_dependents = std::distance(first, last);

  std::shared_ptr<Node> node(
    node_pool.animate(
      name, 0, nullptr, nullptr, num_dependents,
      std::in_place_type_t<Node::DependentAsync>{},
      _make_promised_async(std::move(p), std::forward<F>(func))
    ),
    [&](Node* ptr){ node_pool.recycle(ptr); }
  );
  
  {
    std::scoped_lock lock(_asyncs_mutex);
    _asyncs.insert(node);
  }
  
  for(; first != last; first++) {
    _process_async_dependent(node.get(), *first, num_dependents);
  }

  if(num_dependents == 0) {
    _schedule_async_task(node.get());
  }

  return std::make_pair(AsyncTask(std::move(node)), std::move(fu));
}

// ----------------------------------------------------------------------------
// Dependent Async Helper Functions
// ----------------------------------------------------------------------------

// Procedure: _process_async_dependent
inline void Executor::_process_async_dependent(
  Node* node, tf::AsyncTask& task, size_t& num_dependents
) {

  std::shared_ptr<Node> dep;
  {
    std::scoped_lock lock(_asyncs_mutex);
    if(auto itr = _asyncs.find(task._node); itr != _asyncs.end()){
      dep = *itr;
    }
  }
  
  // if the dependent task exists
  if(dep) {
    auto& state = std::get_if<Node::DependentAsync>(&(dep->_handle))->state;

    add_dependent:

    auto target = Node::AsyncState::UNFINISHED;
    
    // acquires the lock
    if(state.compare_exchange_weak(target, Node::AsyncState::LOCKED,
                                   std::memory_order_acq_rel,
                                   std::memory_order_acquire)) {
      dep->_successors.push_back(node);
      state.store(Node::AsyncState::UNFINISHED, std::memory_order_release);
    }
    // dep's state is FINISHED, which means dep finished its callable already
    // thus decrement the node's join counter by 1
    else if (target == Node::AsyncState::FINISHED) {
      // decrement the counter needs to be the order of acquire and release
      // to synchronize with the worker
      num_dependents = node->_join_counter.fetch_sub(1, std::memory_order_acq_rel) - 1;
    }
    // another worker adding an async task that shares the same dependent
    else {
      goto add_dependent;
    }
  }
  else {
    num_dependents = node->_join_counter.fetch_sub(1, std::memory_order_acq_rel) - 1;
  }
}

// Procedure: _tear_down_dependent_async
inline void Executor::_tear_down_dependent_async(Worker& worker, Node* node) {
  
  // this async task comes from Executor
  auto& state = std::get_if<Node::DependentAsync>(&(node->_handle))->state;
  auto target = Node::AsyncState::UNFINISHED;

  while(!state.compare_exchange_weak(target, Node::AsyncState::FINISHED,
                                     std::memory_order_acq_rel,
                                     std::memory_order_relaxed)) {
    target = Node::AsyncState::UNFINISHED;
  }
  
  // spaw successors whenever their dependencies are resolved
  worker._cache = nullptr;
  for(size_t i=0; i<node->_successors.size(); ++i) {
    //if(auto s = node->_successors[i]; --(s->_join_counter) == 0) {
    if(auto s = node->_successors[i]; 
      s->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1
    ) {
      if(worker._cache) {
        _schedule(worker, worker._cache);
      }
      worker._cache = s;
    }
  }
    
  // remove myself from the asyncs using extraction to avoid calling
  // ~Node inside the lock
  typename std::unordered_set<std::shared_ptr<Node>>::node_type extracted;
  {
    std::shared_ptr<Node> ptr(node, [](Node*){});
    std::scoped_lock lock(_asyncs_mutex); 
    extracted = _asyncs.extract(ptr);
    // assert(extracted.empty() == false);
  }
  
  _decrement_topology_and_notify();
}





}  // end of namespace tf -----------------------------------------------------

