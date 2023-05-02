#pragma once

#include "executor.hpp"

// https://hackmd.io/@sysprog/concurrency-atomics

namespace tf {

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
      std::in_place_type_t<Node::SilentDependentAsync>{}, std::forward<F>(func)
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
    if(auto w = _this_worker(); w) {
      _schedule(*w, node.get());
    }
    else {
      _schedule(node.get());
    }
  }

  return AsyncTask(std::move(node));
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
      std::in_place_type_t<Node::SilentDependentAsync>{}, std::forward<F>(func)
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
    if(auto w = _this_worker(); w) {
      _schedule(*w, node.get());
    }
    else {
      _schedule(node.get());
    }
  }

  return AsyncTask(std::move(node));
}

// Function: dependent_async
template <typename F, typename I,
  std::enable_if_t<!std::is_same_v<std::decay_t<I>, AsyncTask>, void>*
>
auto Executor::dependent_async(
  const std::string& name, F&& f, I first, I last
) {
  
  _increment_topology();
  
  using T = std::invoke_result_t<std::decay_t<F>>;
  using R = std::conditional_t<std::is_same_v<T, void>, void, std::optional<T>>;

  std::promise<R> p;

  auto tpg = std::make_shared<AsyncTopology>();

  Future<R> fu(p.get_future(), tpg);

  size_t num_dependents = std::distance(first, last);

  std::shared_ptr<Node> node(
    node_pool.animate(
      name, 0, nullptr, nullptr, num_dependents,
      std::in_place_type_t<Node::SilentDependentAsync>{},
      [p=make_moc(std::move(p)), f=std::forward<F>(f)]
      (bool cancel) mutable {
        if constexpr(std::is_same_v<R, void>) {
          if(!cancel) {
            f();
          }
          p.object.set_value();
        }
        else {
          p.object.set_value(cancel ? std::nullopt : std::make_optional(f()));
        }
      },
      std::move(tpg)
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
    if(auto w = _this_worker(); w) {
      _schedule(*w, node.get());
    }
    else {
      _schedule(node.get());
    }
  }

  return std::make_pair(AsyncTask(std::move(node)), std::move(fu));
}

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
    auto& state = std::get_if<Node::SilentDependentAsync>(&(dep->_handle))->state;
    auto target = Node::AsyncState::UNFINISHED;
    if(state.compare_exchange_strong(target, Node::AsyncState::LOCKED,
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {
      dep->_successors.push_back(node);
      state.store(Node::AsyncState::UNFINISHED, std::memory_order_release);
    }
    // dep's state is FINISHED, which means dep finished its callable already
    // thus decrement the node's join counter by 1
    else {
      // Here, we use release to ensure this node's creation is seen by other worker
      // that will invoke this node
      num_dependents = node->_join_counter.fetch_sub(1, std::memory_order_release) - 1;
    }
  }
  // dep is removed from the queue - since there is a lock, we do not care
  // the oerder and can be memory-relaxed
  else {
    num_dependents = node->_join_counter.fetch_sub(1, std::memory_order_relaxed) - 1;
  }
}

// Procedure: _tear_down_dependent_async
inline void Executor::_tear_down_dependent_async(Worker& worker, Node* node) {
  
  // this async task comes from Runtime
  //if(node->_parent) {
  //  node->_parent->_join_counter.fetch_sub(1);
  //  node_pool.recycle(node);
  //  return;
  //}
  
  // this async task comes from Executor
  auto& state = std::get_if<Node::SilentDependentAsync>(&(node->_handle))->state;
  auto target = Node::AsyncState::UNFINISHED;

  while(!state.compare_exchange_weak(target, Node::AsyncState::FINISHED,
                                     std::memory_order_acq_rel,
                                     std::memory_order_relaxed)) {
    target = Node::AsyncState::UNFINISHED;
  }
  
  // TODO: optimize this out using cache
  for(size_t i=0; i<node->_successors.size(); ++i) {
    //if(auto s = node->_successors[i]; --(s->_join_counter) == 0) {
    if(auto s = node->_successors[i]; 
      s->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1
    ) {
      _schedule(worker, s);
    }
  }
    
  // remove myself from the asyncs using extraction to avoid calling
  // ~Node inside the lock
  std::unordered_set<std::shared_ptr<Node>>::node_type extracted;
  {
    std::shared_ptr<Node> ptr(node, [](Node*){});
    std::scoped_lock lock(_asyncs_mutex); 
    extracted = _asyncs.extract(ptr);
    // assert(extracted.empty() == false);
  }
  
  _decrement_topology_and_notify();
}





}  // end of namespace tf -----------------------------------------------------
