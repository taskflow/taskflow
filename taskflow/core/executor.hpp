#pragma once

#include "observer.hpp"
#include "taskflow.hpp"

/** 
@file executor.hpp
@brief executor include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// Executor Definition
// ----------------------------------------------------------------------------


/** @class Executor

@brief execution interface for running a taskflow graph

An executor object manages a set of worker threads to run taskflow(s)
using an efficient work-stealing scheduling algorithm.

*/
class Executor {

  friend class FlowBuilder;
  friend class Subflow;
  friend class cudaFlow;

  struct PerThread {
    Worker* worker;
    PerThread() : worker {nullptr} { }
  };

  public:

    /**
    @brief constructs the executor with N worker threads
    */
    explicit Executor(size_t N = std::thread::hardware_concurrency());
    
    /**
    @brief destructs the executor 
    */
    ~Executor();

    /**
    @brief runs the taskflow once
    
    @param taskflow a tf::Taskflow object

    @return a std::future to access the execution status of the taskflow
    */
    std::future<void> run(Taskflow& taskflow);

    /**
    @brief runs the taskflow once and invoke a callback upon completion

    @param taskflow a tf::Taskflow object 
    @param callable a callable object to be invoked after this run

    @return a std::future to access the execution status of the taskflow
    */
    template<typename C>
    std::future<void> run(Taskflow& taskflow, C&& callable);

    /**
    @brief runs the taskflow for N times
    
    @param taskflow a tf::Taskflow object
    @param N number of runs

    @return a std::future to access the execution status of the taskflow
    */
    std::future<void> run_n(Taskflow& taskflow, size_t N);

    /**
    @brief runs the taskflow for N times and then invokes a callback

    @param taskflow a tf::Taskflow 
    @param N number of runs
    @param callable a callable object to be invoked after this run

    @return a std::future to access the execution status of the taskflow
    */
    template<typename C>
    std::future<void> run_n(Taskflow& taskflow, size_t N, C&& callable);

    /**
    @brief runs the taskflow multiple times until the predicate becomes true and 
           then invokes a callback

    @param taskflow a tf::Taskflow 
    @param pred a boolean predicate to return true for stop

    @return a std::future to access the execution status of the taskflow
    */
    template<typename P>
    std::future<void> run_until(Taskflow& taskflow, P&& pred);

    /**
    @brief runs the taskflow multiple times until the predicate becomes true and 
           then invokes the callback

    @param taskflow a tf::Taskflow 
    @param pred a boolean predicate to return true for stop
    @param callable a callable object to be invoked after this run

    @return a std::future to access the execution status of the taskflow
    */
    template<typename P, typename C>
    std::future<void> run_until(Taskflow& taskflow, P&& pred, C&& callable);
    
    /**
    @brief wait for all pending graphs to complete
    */
    void wait_for_all();

    /**
    @brief queries the number of worker threads (can be zero)
    */
    size_t num_workers() const;
    
    /**
    @brief queries the number of running topologies at the time of this call

    When a taskflow is submitted to an executor, a topology is created to store
    runtime metadata of the running taskflow.
    */
    size_t num_topologies() const;

    /**
    @brief queries the id of the caller thread in this executor

    Each worker has an unique id from 0 to N-1 exclusive to the associated executor.
    If the caller thread does not belong to the executor, -1 is returned.
    */
    int this_worker_id() const;

    /** 
    @brief runs a given function asynchronously

    @tparam F callable type
    @tparam ArgsT parameter types

    @param f callable object to call
    @param args parameters to pass to the callable
    
    @return a std::future that will eventually hold the result of the function call

    This method is thread-safe. Multiple threads can launch asynchronous tasks 
    at the same time.
    */
    template <typename F, typename... ArgsT>
    auto async(F&& f, ArgsT&&... args);
    
    /**
    @brief similar to tf::Executor::async but does not return a future object
    */
    template <typename F, typename... ArgsT>
    void silent_async(F&& f, ArgsT&&... args);
    
    /**
    @brief constructs an observer to inspect the activities of worker threads

    Each executor manage a list of observers in shared ownership with callers.
    
    @tparam Observer observer type derived from tf::ObserverInterface
    @tparam ArgsT argument parameter pack

    @param args arguments to forward to the constructor of the observer
    
    @return a shared pointer to the created observer
    */
    template <typename Observer, typename... ArgsT>
    std::shared_ptr<Observer> make_observer(ArgsT&&... args);
    
    /**
    @brief removes the associated observer
    */
    template <typename Observer>
    void remove_observer(std::shared_ptr<Observer> observer);

    /**
    @brief queries the number of observers
    */
    size_t num_observers() const;

  private:

    inline static thread_local PerThread _per_thread;

    const size_t _VICTIM_BEG;
    const size_t _VICTIM_END;
    const size_t _MAX_STEALS;
    const size_t _MAX_YIELDS;
   
    std::condition_variable _topology_cv;
    std::mutex _topology_mutex;
    std::mutex _wsq_mutex;

    size_t _num_topologies {0};
    
    std::vector<Worker> _workers;
    std::vector<std::thread> _threads;

    Notifier _notifier;

    TaskQueue<Node*> _wsq;

    std::atomic<size_t> _num_actives {0};
    std::atomic<size_t> _num_thieves {0};
    std::atomic<bool>   _done {0};
    
    std::unordered_set<std::shared_ptr<ObserverInterface>> _observers;

    TFProfObserver* _tfprof;
    
    bool _wait_for_task(Worker&, Node*&);
    
    void _instantiate_tfprof();
    void _flush_tfprof();
    void _observer_prologue(Worker&, Node*);
    void _observer_epilogue(Worker&, Node*);
    void _spawn(size_t);
    void _worker_loop(Worker&);
    void _exploit_task(Worker&, Node*&);
    void _explore_task(Worker&, Node*&);
    void _schedule(Node*);
    void _schedule(const std::vector<Node*>&);
    void _invoke(Worker&, Node*);
    void _invoke_static_task(Worker&, Node*);
    void _invoke_dynamic_task(Worker&, Node*);
    void _invoke_dynamic_task_internal(Worker&, Node*, Graph&, bool);
    void _invoke_dynamic_task_external(Node*, Graph&, bool);
    void _invoke_condition_task(Worker&, Node*, int&);
    void _invoke_module_task(Worker&, Node*);
    void _invoke_async_task(Worker&, Node*);
    void _set_up_topology(Topology*);
    void _tear_down_topology(Topology*); 
    void _increment_topology();
    void _decrement_topology();
    void _decrement_topology_and_notify();

    void _invoke_cudaflow_task(Worker&, Node*);
    
    template <typename C, std::enable_if_t<
      std::is_invocable_r_v<void, C, cudaFlow&>, void>* = nullptr
    >
    void _invoke_cudaflow_task_entry(C&&, Node*);
    
    template <typename C, std::enable_if_t<
      std::is_invocable_r_v<void, C, cudaFlowCapturer&>, void>* = nullptr
    >
    void _invoke_cudaflow_task_entry(C&&, Node*);
    
    //template <typename P>
    //void _invoke_cudaflow_task_internal(cudaFlow&, P&&, bool);
    
    //template <typename P>
    //void _invoke_cudaflow_task_external(cudaFlow&, P&&, bool);
};


// Constructor
inline Executor::Executor(size_t N) : 
  _VICTIM_BEG {0},
  _VICTIM_END {N - 1},
  _MAX_STEALS {(N + 1) << 1},
  _MAX_YIELDS {100},
  _workers    {N},
  _notifier   {N} {
  
  if(N == 0) {
    TF_THROW("no cpu workers to execute taskflows");
  }
  
  _spawn(N);

  // instantite the default observer if requested
  _instantiate_tfprof();
}

// Destructor
inline Executor::~Executor() {
  
  // wait for all topologies to complete
  wait_for_all();
  
  // shut down the scheduler
  _done = true;

  _notifier.notify(true);
  
  for(auto& t : _threads){
    t.join();
  } 
  
  // flush the default observer
  _flush_tfprof();
}

// Procedure: _instantiate_tfprof
inline void Executor::_instantiate_tfprof() {
  // TF_OBSERVER_TYPE
  _tfprof = get_env("TF_ENABLE_PROFILER").empty() ? 
    nullptr : make_observer<TFProfObserver>().get();
}

// Procedure: _flush_tfprof
inline void Executor::_flush_tfprof() {
  if(_tfprof) {
    std::ostringstream fpath;
    fpath << get_env("TF_ENABLE_PROFILER") << _tfprof->_UID << ".tfp";
    std::ofstream ofs(fpath.str());
    _tfprof->dump(ofs);
  }
}

// Function: num_workers
inline size_t Executor::num_workers() const {
  return _workers.size();
}

// Function: num_topologies
inline size_t Executor::num_topologies() const {
  return _num_topologies;
}
    
// Function: async
template <typename F, typename... ArgsT>
auto Executor::async(F&& f, ArgsT&&... args) {

  _increment_topology();

  using R = typename function_traits<F>::return_type;

  std::promise<R> p;

  auto fu = p.get_future();

  auto node = node_pool.animate(
    std::in_place_type_t<Node::AsyncTask>{},
    [p=make_moc(std::move(p)), f=std::forward<F>(f), args...] () mutable {
      if constexpr(std::is_same_v<R, void>) {
        f(args...);
        p.object.set_value();
      }
      else {
        p.object.set_value(f(args...));
      }
    }
  );

  _schedule(node);

  return fu;
}

// Function: silent_async
template <typename F, typename... ArgsT>
void Executor::silent_async(F&& f, ArgsT&&... args) {

  _increment_topology();

  Node* node = node_pool.animate(
    std::in_place_type_t<Node::AsyncTask>{},
    [f=std::forward<F>(f), args...] () mutable { f(args...); }
  );

  _schedule(node);
}

// Function: this_worker_id
inline int Executor::this_worker_id() const {
  auto worker = _per_thread.worker;
  return worker ? static_cast<int>(worker->id) : -1;
}

// Procedure: _spawn
inline void Executor::_spawn(size_t N) {
  for(size_t id=0; id<N; ++id) {

    _workers[id].id = id;
    _workers[id].vtm = id;
    _workers[id].executor = this;
    _workers[id].waiter = &_notifier._waiters[id];
    
    _threads.emplace_back([this] (Worker& w) -> void {

      _per_thread.worker = &w;

      Node* t = nullptr;

      // must use 1 as condition instead of !done
      while(1) {
        
        // execute the tasks.
        _exploit_task(w, t);

        // wait for tasks
        if(_wait_for_task(w, t) == false) {
          break;
        }
      }
      
    }, std::ref(_workers[id]));     
  }
}

// Function: _explore_task
inline void Executor::_explore_task(Worker& w, Node*& t) {
  
  //assert(_workers[w].wsq.empty());
  assert(!t);

  size_t num_steals = 0;
  size_t num_yields = 0;

  std::uniform_int_distribution<size_t> rdvtm(_VICTIM_BEG, _VICTIM_END);

  //while(!_done) {
  //
  //  size_t vtm = rdvtm(w.rdgen);
  //    
  //  t = (vtm == w.id) ? _wsq[d].steal() : _workers[vtm].wsq[d].steal();

  //  if(t) {
  //    break;
  //  }

  //  if(num_steal++ > _MAX_STEALS) {
  //    std::this_thread::yield();
  //    if(num_yields++ > _MAX_YIELDS) {
  //      break;
  //    }
  //  }
  //}

  do {
    t = (w.id == w.vtm) ? _wsq.steal() : _workers[w.vtm].wsq.steal();

    if(t) {
      break;
    }
    
    if(num_steals++ > _MAX_STEALS) {
      std::this_thread::yield();
      if(num_yields++ > _MAX_YIELDS) {
        break;
      }
    }
    
    w.vtm = rdvtm(w.rdgen);
  } while(!_done);

}

// Procedure: _exploit_task
inline void Executor::_exploit_task(Worker& w, Node*& t) {
  
  if(t) {

    if(_num_actives.fetch_add(1) == 0 && _num_thieves == 0) {
      _notifier.notify(false);
    }

    while(t) {
      _invoke(w, t);
      t = w.wsq.pop();
    }

    --_num_actives;
  }
}

// Function: _wait_for_task
inline bool Executor::_wait_for_task(Worker& worker, Node*& t) {

  wait_for_task:

  assert(!t);

  ++_num_thieves;

  explore_task:

  _explore_task(worker, t);

  if(t) {
    if(_num_thieves.fetch_sub(1) == 1) {
      _notifier.notify(false);
    }
    return true;
  }

  _notifier.prepare_wait(worker.waiter);
  
  //if(auto vtm = _find_vtm(me); vtm != _workers.size()) {
  if(!_wsq.empty()) {

    _notifier.cancel_wait(worker.waiter);
    //t = (vtm == me) ? _wsq.steal() : _workers[vtm].wsq.steal();
    
    t = _wsq.steal();  // must steal here
    if(t) {
      if(_num_thieves.fetch_sub(1) == 1) {
        _notifier.notify(false);
      }
      return true;
    }
    else {
      worker.vtm = worker.id;
      goto explore_task;
    }
  }

  if(_done) {
    _notifier.cancel_wait(worker.waiter);
    _notifier.notify(true);
    --_num_thieves;
    return false;
  }

  if(_num_thieves.fetch_sub(1) == 1) {
    if(_num_actives) {
      _notifier.cancel_wait(worker.waiter);
      goto wait_for_task;
    }
    // check all queues again
    for(auto& w : _workers) {
      if(!w.wsq.empty()) {
        worker.vtm = w.id;
        _notifier.cancel_wait(worker.waiter);
        goto wait_for_task;
      }
    }
  }
    
  // Now I really need to relinguish my self to others
  _notifier.commit_wait(worker.waiter);

  return true;
}

// Function: make_observer    
template<typename Observer, typename... ArgsT>
std::shared_ptr<Observer> Executor::make_observer(ArgsT&&... args) {

  static_assert(
    std::is_base_of_v<ObserverInterface, Observer>,
    "Observer must be derived from ObserverInterface"
  );
  
  // use a local variable to mimic the constructor 
  auto ptr = std::make_shared<Observer>(std::forward<ArgsT>(args)...);
  
  ptr->set_up(_workers.size());

  _observers.emplace(std::static_pointer_cast<ObserverInterface>(ptr));

  return ptr;
}

// Procedure: remove_observer
template <typename Observer>
void Executor::remove_observer(std::shared_ptr<Observer> ptr) {
  
  static_assert(
    std::is_base_of_v<ObserverInterface, Observer>,
    "Observer must be derived from ObserverInterface"
  );

  _observers.erase(std::static_pointer_cast<ObserverInterface>(ptr));
}

// Function: num_observers
inline size_t Executor::num_observers() const {
  return _observers.size();
}

// Procedure: _schedule
// The main procedure to schedule a give task node.
// Each task node has two types of tasks - regular and subflow.
inline void Executor::_schedule(Node* node) {
  
  //assert(_workers.size() != 0);

  // caller is a worker to this pool
  auto worker = _per_thread.worker;

  if(worker != nullptr && worker->executor == this) {
    worker->wsq.push(node);
    return;
  }

  // other threads
  {
    std::lock_guard<std::mutex> lock(_wsq_mutex);
    _wsq.push(node);
  }

  _notifier.notify(false);
}

// Procedure: _schedule
// The main procedure to schedule a set of task nodes.
// Each task node has two types of tasks - regular and subflow.
inline void Executor::_schedule(const std::vector<Node*>& nodes) {

  //assert(_workers.size() != 0);
  
  // We need to cacth the node count to avoid accessing the nodes
  // vector while the parent topology is removed!
  const auto num_nodes = nodes.size();
  
  if(num_nodes == 0) {
    return;
  }

  // worker thread
  auto worker = _per_thread.worker;

  if(worker != nullptr && worker->executor == this) {
    for(size_t i=0; i<num_nodes; ++i) {
      worker->wsq.push(nodes[i]);
    }
    return;
  }
  
  // other threads
  {
    std::lock_guard<std::mutex> lock(_wsq_mutex);
    for(size_t k=0; k<num_nodes; ++k) {
      _wsq.push(nodes[k]);
    }
  }
  
  _notifier.notify_n(num_nodes);
}


// Procedure: _invoke
inline void Executor::_invoke(Worker& worker, Node* node) {
  
  // if acquiring semaphore(s) exists, acquire them first
  if(node->_semaphores && !node->_semaphores->to_acquire.empty()) {
    std::vector<Node*> nodes;
    if(!node->_acquire_all(nodes)) {
      _schedule(nodes);
      return;
    }
  }

  //assert(_workers.size() != 0);

  // Here we need to fetch the num_successors first to avoid the invalid memory
  // access caused by topology clear.
  const auto num_successors = node->num_successors();
  
  // task type
  auto type = node->_handle.index();
  
  // condition task
  int cond = -1;
  
  // switch is faster than nested if-else due to jump table
  switch(type) {
    // static task
    case Node::STATIC_TASK:{
      _invoke_static_task(worker, node);
    } 
    break;
    
    // dynamic task
    case Node::DYNAMIC_TASK: {
      _invoke_dynamic_task(worker, node);
    }
    break;
    
    // condition task
    case Node::CONDITION_TASK: {
      _invoke_condition_task(worker, node, cond);
    }
    break;
    //}  // no need to add a break here due to the immediate return

    // module task
    case Node::MODULE_TASK: {
      _invoke_module_task(worker, node);
    }
    break;

    // async task
    case Node::ASYNC_TASK: {
      _invoke_async_task(worker, node);
      if(node->_parent) {
        node->_parent->_join_counter.fetch_sub(1);
      }
      else {
        _decrement_topology_and_notify();
      }
      // recycle the node
      node_pool.recycle(node);
      return ;
    }
    break;

    // cudaflow task
    case Node::CUDAFLOW_TASK: {
      _invoke_cudaflow_task(worker, node);
    }
    break; 

    // monostate
    default:
    break;
  }

  // if releasing semaphores exist, release them
  if(node->_semaphores && !node->_semaphores->to_release.empty()) {
    _schedule(node->_release_all());
  }

  // We MUST recover the dependency since the graph may have cycles.
  // This must be done before scheduling the successors, otherwise this might cause 
  // race condition on the _dependents
  if(node->_has_state(Node::BRANCHED)) {
    node->_join_counter = node->num_strong_dependents();
  }
  else {
    node->_join_counter = node->num_dependents();
  }
  
  // acquire the parent flow counter
  auto& c = (node->_parent) ? node->_parent->_join_counter : 
                              node->_topology->_join_counter;
  
  // At this point, the node storage might be destructed (to be verified)
  // case 1: non-condition task
  if(type != Node::CONDITION_TASK) {
    for(size_t i=0; i<num_successors; ++i) {
      if(--(node->_successors[i]->_join_counter) == 0) {
        c.fetch_add(1);
        _schedule(node->_successors[i]);
      }
    }
  }
  // case 2: condition task
  else {
    if(cond >= 0 && static_cast<size_t>(cond) < num_successors) {
      auto s = node->_successors[cond];
      s->_join_counter.store(0);  // seems redundant but just for invariant
      c.fetch_add(1);
      _schedule(s);
    }
  }

  // tear down topology if the node is the last one
  if(node->_parent == nullptr) {
    if(node->_topology->_join_counter.fetch_sub(1) == 1) {
      _tear_down_topology(node->_topology);
    }
  }
  else {  // joined subflow
    node->_parent->_join_counter.fetch_sub(1);
  }
}

// Procedure: _observer_prologue
inline void Executor::_observer_prologue(Worker& worker, Node* node) {
  for(auto& observer : _observers) {
    observer->on_entry(WorkerView(worker), TaskView(*node));
  }
}

// Procedure: _observer_epilogue
inline void Executor::_observer_epilogue(Worker& worker, Node* node) {
  for(auto& observer : _observers) {
    observer->on_exit(WorkerView(worker), TaskView(*node));
  }
}

// Procedure: _invoke_static_task
inline void Executor::_invoke_static_task(Worker& worker, Node* node) {
  _observer_prologue(worker, node);
  std::get<Node::StaticTask>(node->_handle).work();
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_dynamic_task
inline void Executor::_invoke_dynamic_task(Worker& w, Node* node) {

  _observer_prologue(w, node);

  auto& handle = std::get<Node::DynamicTask>(node->_handle);

  handle.subgraph.clear();

  Subflow sf(*this, node, handle.subgraph); 

  handle.work(sf);

  if(sf._joinable) {
    _invoke_dynamic_task_internal(w, node, handle.subgraph, false);
  }
  
  _observer_epilogue(w, node);
}

// Procedure: _invoke_dynamic_task_external
inline void Executor::_invoke_dynamic_task_external(Node*p, Graph& g, bool detach) {

  auto worker = _per_thread.worker;

  assert(worker && worker->executor == this);
  
  _invoke_dynamic_task_internal(*worker, p, g, detach);
}

// Procedure: _invoke_dynamic_task_internal
inline void Executor::_invoke_dynamic_task_internal(
  Worker& w, Node* p, Graph& g, bool detach
) {

  // graph is empty and has no async tasks
  if(g.empty() && p->_join_counter == 0) {
    return;
  }

  std::vector<Node*> src; 

  for(auto n : g._nodes) {

    n->_topology = p->_topology;
    n->_set_up_join_counter();

    if(detach) {
      n->_parent = nullptr;
      n->_set_state(Node::DETACHED);
    }
    else {
      n->_parent = p;
    }
    
    if(n->num_dependents() == 0) {
      src.push_back(n);
    }
  }
  
  // detach here
  if(detach) {    
    
    {
      std::lock_guard<std::mutex> lock(p->_topology->_taskflow._mtx);
      p->_topology->_taskflow._graph.merge(std::move(g));
    }

    p->_topology->_join_counter.fetch_add(src.size());
    _schedule(src);
  }
  // join here
  else {  
    p->_join_counter.fetch_add(src.size());
    _schedule(src);
    Node* t = nullptr;
  
    std::uniform_int_distribution<size_t> rdvtm(_VICTIM_BEG, _VICTIM_END);

    while(p->_join_counter != 0) {

      t = w.wsq.pop();

      exploit:

      if(t) {
        _invoke(w, t);
      }
      else {

        explore:
        t = (w.id == w.vtm) ? _wsq.steal() : _workers[w.vtm].wsq.steal();
        if(t) {
          goto exploit;
        }
        else if(p->_join_counter != 0){
          std::this_thread::yield();
          w.vtm = rdvtm(w.rdgen);
          goto explore;
        }
        else {
          break;
        }
      }
    }
  }
}

// Procedure: _invoke_condition_task
inline void Executor::_invoke_condition_task(
  Worker& worker, Node* node, int& cond
) {
  _observer_prologue(worker, node);
  cond = std::get<Node::ConditionTask>(node->_handle).work();
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_cudaflow_task
inline void Executor::_invoke_cudaflow_task(Worker& worker, Node* node) {
  _observer_prologue(worker, node);  
  std::get<Node::cudaFlowTask>(node->_handle).work(*this, node);
  _observer_epilogue(worker, node);
}

//// Procedure: _invoke_cudaflow_task_internal
//template <typename P>
//void Executor::_invoke_cudaflow_task_internal(
//  cudaFlow& cf, P&& predicate, bool join
//) {
//  
//  if(cf.empty()) {
//    return;
//  }
//  
//  // transforms cudaFlow to a native cudaGraph under the specified device
//  // and launches the graph through a given or an internal device stream
//  if(cf._executable == nullptr) {
//    cf._create_executable();
//    //cuda_dump_graph(std::cout, cf._graph._native_handle);
//  }
//
//  cudaScopedPerThreadStream s;
//
//  while(!predicate()) {
//
//    TF_CHECK_CUDA(
//      cudaGraphLaunch(cf._executable, s), "failed to execute cudaFlow"
//    );
//
//    TF_CHECK_CUDA(
//      cudaStreamSynchronize(s), "failed to synchronize cudaFlow execution"
//    );
//  }
//
//  if(join) {
//    //cuda_dump_graph(std::cout, cf._graph._native_handle);
//    cf._destroy_executable();
//  }
//}

//// Procedure: _invoke_cudaflow_task_external
//template <typename P>
//void Executor::_invoke_cudaflow_task_external(
//  cudaFlow& cf, P&& predicate, bool join
//) {
//
//  auto w = _per_thread().worker;
//  
//  assert(w && w->executor == this);
//
//  _invoke_cudaflow_task_internal(*w, cf, std::forward<P>(predicate), join);
//}

// Procedure: _invoke_module_task
inline void Executor::_invoke_module_task(Worker& w, Node* node) {
  _observer_prologue(w, node);
  auto module = std::get<Node::ModuleTask>(node->_handle).module;
  _invoke_dynamic_task_internal(w, node, module->_graph, false);
  _observer_epilogue(w, node);  
}

// Procedure: _invoke_async_task
inline void Executor::_invoke_async_task(Worker& w, Node* node) {
  _observer_prologue(w, node);
  std::get<Node::AsyncTask>(node->_handle).work();
  _observer_epilogue(w, node);  
}

// Function: run
inline std::future<void> Executor::run(Taskflow& f) {
  return run_n(f, 1, [](){});
}

// Function: run
template <typename C>
std::future<void> Executor::run(Taskflow& f, C&& c) {
  return run_n(f, 1, std::forward<C>(c));
}

// Function: run_n
inline std::future<void> Executor::run_n(Taskflow& f, size_t repeat) {
  return run_n(f, repeat, [](){});
}

// Function: run_n
template <typename C>
std::future<void> Executor::run_n(Taskflow& f, size_t repeat, C&& c) {
  return run_until(
    f, [repeat]() mutable { return repeat-- == 0; }, std::forward<C>(c)
  );
}

// Function: run_until    
template<typename P>
std::future<void> Executor::run_until(Taskflow& f, P&& pred) {
  return run_until(f, std::forward<P>(pred), [](){});
}

// Function: _set_up_topology
inline void Executor::_set_up_topology(Topology* tpg) {

  tpg->_sources.clear();
  tpg->_taskflow._graph.clear_detached();
  
  // scan each node in the graph and build up the links
  for(auto node : tpg->_taskflow._graph._nodes) {
    
    node->_topology = tpg;
    node->_clear_state();

    if(node->num_dependents() == 0) {
      tpg->_sources.push_back(node);
    }

    node->_set_up_join_counter();
  }

  tpg->_join_counter.store(tpg->_sources.size(), std::memory_order_relaxed);
}

// Function: _tear_down_topology
inline void Executor::_tear_down_topology(Topology* tpg) {

  auto &f = tpg->_taskflow;

  //assert(&tpg == &(f._topologies.front()));

  // case 1: we still need to run the topology again
  if(! tpg->_pred() ) {
    //tpg->_recover_num_sinks();

    assert(tpg->_join_counter == 0);
    tpg->_join_counter = tpg->_sources.size();

    _schedule(tpg->_sources); 
  }
  // case 2: the final run of this topology
  else {
    
    if(tpg->_call != nullptr) {
      tpg->_call();
    }

    f._mtx.lock();

    // If there is another run (interleave between lock)
    if(f._topologies.size() > 1) {

      assert(tpg->_join_counter == 0);

      // Set the promise
      tpg->_promise.set_value();
      f._topologies.pop_front();
      f._mtx.unlock();
      
      // decrement the topology but since this is not the last we don't notify
      _decrement_topology();

      tpg = &(f._topologies.front());

      _set_up_topology(tpg);
      _schedule(tpg->_sources);

      //f._topologies.front()._bind(f._graph);
      //*tpg = &(f._topologies.front());

      //assert(f._topologies.front()._join_counter == 0);

      //f._topologies.front()._join_counter = f._topologies.front()._sources.size();

      //_schedule(f._topologies.front()._sources);
    }
    else {
      assert(f._topologies.size() == 1);

      // Need to back up the promise first here becuz taskflow might be 
      // destroy before taskflow leaves
      auto p {std::move(tpg->_promise)};

      // Back up lambda capture in case it has the topology pointer, to avoid it releasing on 
      // pop_front ahead of _mtx.unlock & _promise.set_value. Released safely when leaving scope.
      auto bc{ std::move( tpg->_call ) };

      f._topologies.pop_front();

      f._mtx.unlock();

      // We set the promise in the end in case taskflow leaves before taskflow
      p.set_value();

      _decrement_topology_and_notify();
    }
  }
}

// Function: run_until
template <typename P, typename C>
std::future<void> Executor::run_until(Taskflow& f, P&& pred, C&& c) {

  _increment_topology();

  // Special case of predicate
  if(f.empty() || pred()) {
    std::promise<void> promise;
    promise.set_value();
    _decrement_topology_and_notify();
    return promise.get_future();
  }
  
  // Multi-threaded execution.
  bool run_now {false};
  Topology* tpg;
  std::future<void> future;
  
  {
    std::lock_guard<std::mutex> lock(f._mtx);

    // create a topology for this run
    //tpg = &(f._topologies.emplace_back(f, std::forward<P>(pred), std::forward<C>(c)));
    f._topologies.emplace_back(f, std::forward<P>(pred), std::forward<C>(c));
    tpg = &(f._topologies.back());
    future = tpg->_promise.get_future();
   
    if(f._topologies.size() == 1) {
      run_now = true;
      //tpg->_bind(f._graph);
      //_schedule(tpg->_sources);
    }
  }
  
  // Notice here calling schedule may cause the topology to be removed sonner 
  // before the function leaves.
  if(run_now) {
    _set_up_topology(tpg);
    _schedule(tpg->_sources);
  }

  return future;
}

// Procedure: _increment_topology
inline void Executor::_increment_topology() {
  std::lock_guard<std::mutex> lock(_topology_mutex);
  ++_num_topologies;
}

// Procedure: _decrement_topology_and_notify
inline void Executor::_decrement_topology_and_notify() {
  std::lock_guard<std::mutex> lock(_topology_mutex);
  if(--_num_topologies == 0) {
    _topology_cv.notify_all();
  }
}

// Procedure: _decrement_topology
inline void Executor::_decrement_topology() {
  std::lock_guard<std::mutex> lock(_topology_mutex);
  --_num_topologies;
}

// Procedure: wait_for_all
inline void Executor::wait_for_all() {
  std::unique_lock<std::mutex> lock(_topology_mutex);
  _topology_cv.wait(lock, [&](){ return _num_topologies == 0; });
}

// ############################################################################
// Forward Declaration: Subflow
// ############################################################################

inline void Subflow::join() {

  if(!_joinable) {
    TF_THROW("subflow not joinable");
  }

  _executor._invoke_dynamic_task_external(_parent, _graph, false);
  _joinable = false;
}

inline void Subflow::detach() {

  if(!_joinable) {
    TF_THROW("subflow already joined or detached");
  }

  _executor._invoke_dynamic_task_external(_parent, _graph, true);
  _joinable = false;
}

// Function: async
template <typename F, typename... ArgsT>
auto Subflow::async(F&& f, ArgsT&&... args) {

  _parent->_join_counter.fetch_add(1);

  using R = typename function_traits<F>::return_type;

  std::promise<R> p;

  auto fu = p.get_future();

  auto node = node_pool.animate(
    std::in_place_type_t<Node::AsyncTask>{},
    [p=make_moc(std::move(p)), f=std::forward<F>(f), args...] () mutable {
      if constexpr(std::is_same_v<R, void>) {
        f(args...);
        p.object.set_value();
      }
      else {
        p.object.set_value(f(args...));
      }
    }
  );

  node->_topology = _parent->_topology;
  node->_parent = _parent;

  _executor._schedule(node);

  return fu;
}

// Function: silent_async
template <typename F, typename... ArgsT>
void Subflow::silent_async(F&& f, ArgsT&&... args) {

  _parent->_join_counter.fetch_add(1);

  auto node = node_pool.animate(
    std::in_place_type_t<Node::AsyncTask>{},
    [f=std::forward<F>(f), args...] () mutable { f(args...); }
  );

  node->_topology = _parent->_topology;
  node->_parent = _parent;

  _executor._schedule(node);
}


}  // end of namespace tf -----------------------------------------------------



