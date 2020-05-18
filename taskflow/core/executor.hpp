#pragma once

#include "tsq.hpp"
#include "notifier.hpp"
#include "observer.hpp"
#include "taskflow.hpp"

namespace tf {


/** @class WorkerView

@brief class to access worker information from the observer interface

*/
//class WorkerView {
//
//  friend class Executor;
//
//  public:
//
//
//  private:
//
//    Worker* _worker;
//
//};


// ----------------------------------------------------------------------------
// Executor Definition
// ----------------------------------------------------------------------------


/** @class Executor

@brief execution interface for running a taskflow graph

An executor object manages a set of worker threads and implements 
an efficient work-stealing scheduling algorithm to run a taskflow.

*/
class Executor {

  struct Worker {
    size_t id;
    size_t victim;
    Domain domain;
    Executor* executor;
    Notifier::Waiter* waiter;
    std::mt19937 rdgen { std::random_device{}() };
    TaskQueue<Node*> wsq[NUM_DOMAINS];
    Node* cache {nullptr};
  };
    
  struct PerThread {
    Worker* worker {nullptr};
  };

#ifdef TF_ENABLE_CUDA
  struct cudaDevice {
    std::vector<cudaStream_t> streams;
  };
#endif

  public:

#ifdef TF_ENABLE_CUDA    
    /**
    @brief constructs the executor with N/M cpu/gpu worker threads
    */
    explicit Executor(
      size_t N = std::thread::hardware_concurrency(),
      size_t M = cuda_num_devices()
    );
#else
    /**
    @brief constructs the executor with N worker threads
    */
    explicit Executor(size_t N = std::thread::hardware_concurrency());
#endif
    
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
    @brief queries the number of worker domains 

    Each domain manages a subset of worker threads to execute domain-specific tasks,
    for example, HOST tasks and CUDA tasks.
    */
    size_t num_domains() const;

    /**
    @brief queries the id of the caller thread in this executor

    Each worker has an unique id from 0 to N-1 exclusive to the associated executor.
    If the caller thread does not belong to the executor, -1 is returned.
    */
    int this_worker_id() const;
    
    /**
    @brief constructs an observer to inspect the activities of worker threads

    Each executor manage a list of observers in shared ownership with callers.
    
    @tparam Observer observer type derived from tf::ObserverInterface
    @tparam ArgsT... argument parameter pack

    @param args arguments to forward to the constructor of the observer
    
    @return a shared pointer to the created observer
    */
    template <typename Observer, typename... Args>
    std::shared_ptr<Observer> make_observer(Args&&... args);
    
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

#ifdef TF_ENABLE_CUDA
    std::vector<cudaDevice> _cuda_devices;
#endif
    
    Notifier _notifier[NUM_DOMAINS];

    TaskQueue<Node*> _wsq[NUM_DOMAINS];

    size_t _id_offset[NUM_DOMAINS] = {0};

    std::atomic<size_t> _num_actives[NUM_DOMAINS];
    std::atomic<size_t> _num_thieves[NUM_DOMAINS];
    std::atomic<bool>   _done {0};
    
    std::unordered_set<std::shared_ptr<ObserverInterface>> _observers;

    TFProfObserver* _tfprof;
    
    PerThread& _per_thread() const;

    bool _wait_for_task(Worker&, Node*&);
    
    void _instantiate_tfprof();
    void _flush_tfprof();
    void _observer_prologue(Worker&, Node*);
    void _observer_epilogue(Worker&, Node*);
    void _spawn(size_t, Domain);
    void _worker_loop(Worker&);
    void _exploit_task(Worker&, Node*&);
    void _explore_task(Worker&, Node*&);
    void _schedule(Node*, bool);
    void _schedule(PassiveVector<Node*>&);
    void _invoke(Worker&, Node*);
    void _invoke_static_work(Worker&, Node*);
    void _invoke_dynamic_work(Worker&, Node*, bool&);
    void _invoke_condition_work(Worker&, Node*);
    void _invoke_module_work(Worker&, Node*, bool&);

#ifdef TF_ENABLE_CUDA
    void _invoke_cudaflow_work(Worker&, Node*);
    void _invoke_cudaflow_work_impl(Worker&, Node*);
#endif

    void _set_up_topology(Topology*);
    void _tear_down_topology(Topology**); 
    void _increment_topology();
    void _decrement_topology();
    void _decrement_topology_and_notify();
};


#ifdef TF_ENABLE_CUDA
// Constructor
inline Executor::Executor(size_t N, size_t M) :
  _VICTIM_BEG   {0},
  _VICTIM_END   {N + M - 1},
  _MAX_STEALS   {(N + M + 1) << 1},
  _MAX_YIELDS   {100},
  _workers      {N + M},
  _cuda_devices {cuda_num_devices()},
  _notifier     {Notifier(N), Notifier(M)} {

  if(N == 0) {
    TF_THROW("no cpu workers to execute taskflows");
  }

  if(M == 0) {
    TF_THROW("no gpu workers to execute cudaflows");
  }

  for(int i=0; i<NUM_DOMAINS; ++i) {
    _num_actives[i].store(0, std::memory_order_relaxed);
    _num_thieves[i].store(0, std::memory_order_relaxed); 
  }
  
  // create a per-worker stream on each cuda device
  for(size_t i=0; i<_cuda_devices.size(); ++i) {
    _cuda_devices[i].streams.resize(M);
    cudaScopedDevice ctx(i);
    for(size_t m=0; m<M; ++m) {
      TF_CHECK_CUDA(
        cudaStreamCreate(&(_cuda_devices[i].streams[m])),
        "failed to create a cudaStream for worker ", m, " on device ", i
      );
    }
  }

  _spawn(N, HOST);
  _spawn(M, CUDA);

  // initiate the observer if requested
  _instantiate_tfprof();
}

#else
// Constructor
inline Executor::Executor(size_t N) : 
  _VICTIM_BEG {0},
  _VICTIM_END {N - 1},
  _MAX_STEALS {(N + 1) << 1},
  _MAX_YIELDS {100},
  _workers    {N},
  _notifier   {Notifier(N)} {
  
  if(N == 0) {
    TF_THROW("no cpu workers to execute taskflows");
  }
  
  for(int i=0; i<NUM_DOMAINS; ++i) {
    _num_actives[i].store(0, std::memory_order_relaxed);
    _num_thieves[i].store(0, std::memory_order_relaxed); 
  }

  _spawn(N, HOST);

  // instantite the default observer if requested
  _instantiate_tfprof();
}
#endif

// Destructor
inline Executor::~Executor() {
  
  // wait for all topologies to complete
  wait_for_all();
  
  // shut down the scheduler
  _done = true;

  for(int i=0; i<NUM_DOMAINS; ++i) {
    _notifier[i].notify(true);
  }
  
  for(auto& t : _threads){
    t.join();
  } 
  
#ifdef TF_ENABLE_CUDA  
  // clean up the cuda streams
  for(size_t i=0; i<_cuda_devices.size(); ++i) {
    cudaScopedDevice ctx(i);
    for(size_t m=0; m<_cuda_devices[i].streams.size(); ++m) {
      cudaStreamDestroy(_cuda_devices[i].streams[m]);
    }
  }
#endif
  
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
    fpath << get_env("TF_ENABLE_PROFILER") << _tfprof->_uuid << ".tfp";
    std::ofstream ofs(fpath.str());
    _tfprof->dump(ofs);
  }
}

// Function: num_workers
inline size_t Executor::num_workers() const {
  return _workers.size();
}

// Function: num_domains
inline size_t Executor::num_domains() const {
  return NUM_DOMAINS;
}

// Function: num_topologies
inline size_t Executor::num_topologies() const {
  return _num_topologies;
}

// Function: _per_thread
inline Executor::PerThread& Executor::_per_thread() const {
  thread_local PerThread pt;
  return pt;
}

// Function: this_worker_id
inline int Executor::this_worker_id() const {
  auto worker = _per_thread().worker;
  return worker ? static_cast<int>(worker->id) : -1;
}

// Procedure: _spawn
inline void Executor::_spawn(size_t N, Domain d) {
  
  auto id = _threads.size();

  _id_offset[d] = id;

  for(size_t i=0; i<N; ++i, ++id) {

    _workers[id].id = id;
    _workers[id].victim = id;
    _workers[id].domain = d;
    _workers[id].executor = this;
    _workers[id].waiter = &_notifier[d]._waiters[i];
    
    _threads.emplace_back([this] (Worker& w) -> void {

      PerThread& pt = _per_thread();  
      pt.worker = &w;

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

  const auto d = w.domain;

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
    t = (w.id == w.victim) ? _wsq[d].steal() : _workers[w.victim].wsq[d].steal();

    if(t) {
      break;
    }
    
    if(num_steals++ > _MAX_STEALS) {
      std::this_thread::yield();
      if(num_yields++ > _MAX_YIELDS) {
        break;
      }
    }
    
    w.victim = rdvtm(w.rdgen);
  } while(!_done);

}

// Procedure: _exploit_task
inline void Executor::_exploit_task(Worker& w, Node*& t) {
  
  assert(!w.cache);

  if(t) {

    const auto d = w.domain;

    if(_num_actives[d].fetch_add(1) == 0 && _num_thieves[d] == 0) {
      _notifier[d].notify(false);
    }
    
    auto tpg = t->_topology;
    auto par = t->_parent;
    auto exe = size_t{1};

    do {
      _invoke(w, t);

      if(w.cache) {
        t = w.cache;
        w.cache = nullptr;
      }
      else {
        t = w.wsq[d].pop();
        if(t) {
          // We only increment the counter when poping task from wsq 
          // (NOT including cache!)
          if(t->_parent == par) {
            exe++;
          }
          // joined subflow
          else {
            if(par == nullptr) {
              // still have tasks so the topology join counter can't be zero
              t->_topology->_join_counter.fetch_sub(exe);
            }
            else {
              auto ret = par->_join_counter.fetch_sub(exe);
              if(ret == exe) {
                if(par->domain() == d) {
                  w.wsq[d].push(par);
                }
                else {
                  _schedule(par, false);
                }
              }
            }
            exe = 1;
            par = t->_parent;
          }
        }
        else {
          // If no more local tasks!
          if(par == nullptr) {
            if(tpg->_join_counter.fetch_sub(exe) == exe) {
              // TODO: Store tpg in local variable not in w
              _tear_down_topology(&tpg);
              if(tpg != nullptr) {
                t = w.wsq[d].pop();
                if(t) {
                  exe = 1;
                }
              }
            }
          }
          else {
            if(par->_join_counter.fetch_sub(exe) == exe) {
              if(par->domain() == d) {
                t = par;
                par = par->_parent;
                exe = 1;
              }
              else {
                _schedule(par, false);
              }
            }
          }
        }
      }
    } while(t);

    --_num_actives[d];
  }
}

// Function: _wait_for_task
inline bool Executor::_wait_for_task(Worker& worker, Node*& t) {

  const auto d = worker.domain;

  wait_for_task:

  assert(!t);

  ++_num_thieves[d];

  explore_task:

  _explore_task(worker, t);

  if(t) {
    if(_num_thieves[d].fetch_sub(1) == 1) {
      _notifier[d].notify(false);
    }
    return true;
  }

  _notifier[d].prepare_wait(worker.waiter);
  
  //if(auto vtm = _find_victim(me); vtm != _workers.size()) {
  if(!_wsq[d].empty()) {

    _notifier[d].cancel_wait(worker.waiter);
    //t = (vtm == me) ? _wsq.steal() : _workers[vtm].wsq.steal();
    
    t = _wsq[d].steal();
    if(t) {
      if(_num_thieves[d].fetch_sub(1) == 1) {
        _notifier[d].notify(false);
      }
      return true;
    }
    else {
      worker.victim = worker.id;
      goto explore_task;
    }
  }

  if(_done) {
    _notifier[d].cancel_wait(worker.waiter);
    for(int i=0; i<NUM_DOMAINS; ++i) {
      _notifier[i].notify(true);
    }
    --_num_thieves[d];
    return false;
  }

  if(_num_thieves[d].fetch_sub(1) == 1) {
    if(_num_actives[d]) {
      _notifier[d].cancel_wait(worker.waiter);
      goto wait_for_task;
    }
    // check all domain queue again
    for(auto& w : _workers) {
      if(!w.wsq[d].empty()) {
        worker.victim = w.id;
        _notifier[d].cancel_wait(worker.waiter);
        goto wait_for_task;
      }
    }
  }
    
  // Now I really need to relinguish my self to others
  _notifier[d].commit_wait(worker.waiter);

  return true;
}

// Function: make_observer    
template<typename Observer, typename... Args>
std::shared_ptr<Observer> Executor::make_observer(Args&&... args) {

  static_assert(
    std::is_base_of<ObserverInterface, Observer>::value,
    "Observer must be derived from ObserverInterface"
  );
  
  // use a local variable to mimic the constructor 
  auto ptr = std::make_shared<Observer>(std::forward<Args>(args)...);
  
  ptr->set_up(_workers.size());

  _observers.emplace(std::static_pointer_cast<ObserverInterface>(ptr));

  return ptr;
}

// Procedure: remove_observer
template <typename Observer>
void Executor::remove_observer(std::shared_ptr<Observer> ptr) {
  
  static_assert(
    std::is_base_of<ObserverInterface, Observer>::value,
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
inline void Executor::_schedule(Node* node, bool bypass_hint) {
  
  //assert(_workers.size() != 0);

  const auto d = node->domain();
  
  // caller is a worker to this pool
  auto worker = _per_thread().worker;

  if(worker != nullptr && worker->executor == this) {
    if(bypass_hint) {
      assert(!worker->cache);
      worker->cache = node;
    }
    else {
      worker->wsq[d].push(node);
      if(worker->domain != d) {
        if(_num_actives[d] == 0 && _num_thieves[d] == 0) {
          _notifier[d].notify(false);
        }
      }
    }
    return;
  }

  // other threads
  {
    std::lock_guard<std::mutex> lock(_wsq_mutex);
    _wsq[d].push(node);
  }

  _notifier[d].notify(false);
}

// Procedure: _schedule
// The main procedure to schedule a set of task nodes.
// Each task node has two types of tasks - regular and subflow.
inline void Executor::_schedule(PassiveVector<Node*>& nodes) {

  //assert(_workers.size() != 0);
  
  // We need to cacth the node count to avoid accessing the nodes
  // vector while the parent topology is removed!
  const auto num_nodes = nodes.size();
  
  if(num_nodes == 0) {
    return;
  }

  // worker thread
  auto worker = _per_thread().worker;

  // task counts
  size_t tcount[NUM_DOMAINS] = {0};

  if(worker != nullptr && worker->executor == this) {
    for(size_t i=0; i<num_nodes; ++i) {
      const auto d = nodes[i]->domain();
      worker->wsq[d].push(nodes[i]);
      tcount[d]++;
    }
    
    for(int d=0; d<NUM_DOMAINS; ++d) {
      if(tcount[d] && d != worker->domain) {
        if(_num_actives[d] == 0 && _num_thieves[d] == 0) {
          _notifier[d].notify_n(tcount[d]);
        }
      }
    }

    return;
  }
  
  // other threads
  {
    std::lock_guard<std::mutex> lock(_wsq_mutex);
    for(size_t k=0; k<num_nodes; ++k) {
      const auto d = nodes[k]->domain();
      _wsq[d].push(nodes[k]);
      tcount[d]++;
    }
  }
  
  for(int d=0; d<NUM_DOMAINS; ++d) {
    _notifier[d].notify_n(tcount[d]);
  }
}


// Procedure: _invoke
inline void Executor::_invoke(Worker& worker, Node* node) {

  //assert(_workers.size() != 0);

  // Here we need to fetch the num_successors first to avoid the invalid memory
  // access caused by topology clear.
  const auto num_successors = node->num_successors();
  
  // acquire the parent flow counter
  auto& c = (node->_parent) ? node->_parent->_join_counter : 
                              node->_topology->_join_counter;
  
  // switch is faster than nested if-else due to jump table
  switch(node->_handle.index()) {
    // static task
    case Node::STATIC_WORK:{
      _invoke_static_work(worker, node);
    } 
    break;

    // module task
    case Node::MODULE_WORK: {
      bool first_time = !node->_has_state(Node::SPAWNED);
      bool emptiness  = false;
      _invoke_module_work(worker, node, emptiness);
      if(first_time && !emptiness) {
        return;
      }
    }
    break;

    // dynamic task
    case Node::DYNAMIC_WORK: {
      // Need to create a subflow if it is the first time entering here
      if(!node->_has_state(Node::SPAWNED)) {
        bool join = false;
        _invoke_dynamic_work(worker, node, join);
        if(join) {
          return;
        }
      }
    }
    break;

    // condition task
    case Node::CONDITION_WORK: {
      _invoke_condition_work(worker, node);
      return ;
    }  // no need to add a break here due to the immediate return

    // cudaflow task
#ifdef TF_ENABLE_CUDA
    case Node::CUDAFLOW_WORK: {
      _invoke_cudaflow_work(worker, node);
    }
    break; 
#endif

    // monostate
    default:
    break;
  }
  

  // We MUST recover the dependency since subflow may have  
  // a condition node to go back (cyclic).
  // This must be done before scheduling the successors, otherwise this might cause 
  // race condition on the _dependents
  if(node->_has_state(Node::BRANCH)) {
    // If this is a case node, we need to deduct condition predecessors
    node->_join_counter = node->num_strong_dependents();
  }
  else {
    node->_join_counter = node->num_dependents();
  }

  node->_unset_state(Node::SPAWNED);

  // At this point, the node storage might be destructed.
  Node* cache {nullptr};

  for(size_t i=0; i<num_successors; ++i) {
    if(--(node->_successors[i]->_join_counter) == 0) {
      if(node->_successors[i]->domain() != worker.domain) {
        c.fetch_add(1);
        _schedule(node->_successors[i], false);
      }
      else {
        if(cache) {
          c.fetch_add(1);
          _schedule(cache, false);
        }
        cache = node->_successors[i];
      }
    }
  }

  if(cache) {
    _schedule(cache, true);
  }
}

// Procedure: _observer_prologue
inline void Executor::_observer_prologue(Worker& worker, Node* node) {
  for(auto& observer : _observers) {
    observer->on_entry(worker.id, TaskView(node));
  }
}

// Procedure: _observer_epilogue
inline void Executor::_observer_epilogue(Worker& worker, Node* node) {
  for(auto& observer : _observers) {
    observer->on_exit(worker.id, TaskView(node));
  }
}

// Procedure: _invoke_static_work
inline void Executor::_invoke_static_work(Worker& worker, Node* node) {
  _observer_prologue(worker, node);
  nstd::get<Node::StaticWork>(node->_handle).work();
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_dynamic_work
inline void Executor::_invoke_dynamic_work(Worker& worker, Node* node, bool& join) {

  _observer_prologue(worker, node);
    
  auto& subgraph = nstd::get<Node::DynamicWork>(node->_handle).subgraph;

  subgraph.clear();
  Subflow fb(subgraph); 

  nstd::get<Node::DynamicWork>(node->_handle).work(fb);

  node->_set_state(Node::SPAWNED);

  if(!subgraph.empty()) {

    PassiveVector<Node*> src; 

    for(auto n : subgraph._nodes) {

      n->_topology = node->_topology;
      n->_set_up_join_counter();
      
      if(!fb.detached()) {
        n->_parent = node;
      }

      if(n->num_dependents() == 0) {
        src.push_back(n);
      }
    }

    join = fb.joined();

    if(!join) {  // Detach mode
      node->_topology->_join_counter.fetch_add(src.size());         
    }
    else {       // Join mode (spawned nodes need second-round execution
      node->_join_counter.fetch_add(src.size());

      node->_parent ? node->_parent->_join_counter.fetch_add(1) :
                      node->_topology->_join_counter.fetch_add(1);
    }

    _schedule(src);
  }
  
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_condition_work
inline void Executor::_invoke_condition_work(Worker& worker, Node* node) {

  _observer_prologue(worker, node);
  
  if(node->_has_state(Node::BRANCH)) {
    node->_join_counter = node->num_strong_dependents();
  }
  else {
    node->_join_counter = node->num_dependents();
  }
  
  auto id = nstd::get<Node::ConditionWork>(node->_handle).work();

  if(id >= 0 && static_cast<size_t>(id) < node->num_successors()) {
    auto s = node->_successors[id];
    s->_join_counter.store(0);

    if(s->domain() == worker.domain) {
      _schedule(s, true);
    }
    else {
      node->_parent ? node->_parent->_join_counter.fetch_add(1) :
                      node->_topology->_join_counter.fetch_add(1);
      _schedule(s, false);
    }
  }

  _observer_epilogue(worker, node);
}

#ifdef TF_ENABLE_CUDA
// Procedure: _invoke_cudaflow_work
inline void Executor::_invoke_cudaflow_work(Worker& worker, Node* node) {
  _observer_prologue(worker, node);  
  _invoke_cudaflow_work_impl(worker, node);
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_cudaflow_work_impl
inline void Executor::_invoke_cudaflow_work_impl(Worker& w, Node* node) {
  
  assert(w.domain == node->domain());

  auto& h = nstd::get<Node::cudaFlowWork>(node->_handle);

  h.graph.clear();

  cudaFlow cf(h.graph, [repeat=1] () mutable { return repeat-- == 0; });

  h.work(cf); 

  if(h.graph.empty()) {
    return;
  }
  
  // transforms cudaFlow to a native cudaGraph under the specified device
  // and launches the graph through a given or an internal device stream
  const int d = cf._device;

  cudaScopedDevice ctx(d);
  
  auto s = cf._stream ? *(cf._stream) : 
                        _cuda_devices[d].streams[w.id - _id_offset[w.domain]];

  h.graph._make_native_graph();

  cudaGraphExec_t exec;

  TF_CHECK_CUDA(
    cudaGraphInstantiate(&exec, h.graph._native_handle, nullptr, nullptr, 0),
    "failed to create an executable cudaGraph"
  );
  
  while(!cf._predicate()) {
    TF_CHECK_CUDA(
      cudaGraphLaunch(exec, s), "failed to launch cudaGraph on stream ", s
    );

    TF_CHECK_CUDA(
      cudaStreamSynchronize(s), "failed to synchronize stream ", s
    );
  }

  TF_CHECK_CUDA(
    cudaGraphExecDestroy(exec), "failed to destroy an executable cudaGraph"
  );
}
#endif

// Procedure: _invoke_module_work
inline void Executor::_invoke_module_work(Worker& worker, Node* node, bool& ept) {

  // second time to enter this context
  if(node->_has_state(Node::SPAWNED)) {
    return;
  }
  
  _observer_prologue(worker, node);
  
  // first time to enter this context
  node->_set_state(Node::SPAWNED);

  auto module = nstd::get<Node::ModuleWork>(node->_handle).module;

  if(module->empty()) {
    ept = true;
    return;
  }

  PassiveVector<Node*> src;

  for(auto n: module->_graph._nodes) {

    n->_topology = node->_topology;
    n->_parent = node;
    n->_set_up_join_counter();

    if(n->num_dependents() == 0) {
      src.push_back(n);
    }
  }

  node->_join_counter.fetch_add(src.size());
    
  if(node->_parent == nullptr) {
    node->_topology->_join_counter.fetch_add(1);
  }
  else {
    node->_parent->_join_counter.fetch_add(1);
  }
  
  // src can't be empty (banned outside)
  _schedule(src);
  
  _observer_epilogue(worker, node);  
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
  return run_until(f, [repeat]() mutable { return repeat-- == 0; }, std::forward<C>(c));
}

// Function: run_until    
template<typename P>
std::future<void> Executor::run_until(Taskflow& f, P&& pred) {
  return run_until(f, std::forward<P>(pred), [](){});
}

// Function: _set_up_topology
inline void Executor::_set_up_topology(Topology* tpg) {

  tpg->_sources.clear();
  
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
inline void Executor::_tear_down_topology(Topology** tpg) {

  auto &f = (*tpg)->_taskflow;

  //assert(&tpg == &(f._topologies.front()));

  // case 1: we still need to run the topology again
  if(! (*tpg)->_pred() ) {
    //tpg->_recover_num_sinks();

    assert((*tpg)->_join_counter == 0);
    (*tpg)->_join_counter = (*tpg)->_sources.size();

    _schedule((*tpg)->_sources); 
  }
  // case 2: the final run of this topology
  else {
    
    if((*tpg)->_call != nullptr) {
      (*tpg)->_call();
    }

    f._mtx.lock();

    // If there is another run (interleave between lock)
    if(f._topologies.size() > 1) {

      assert((*tpg)->_join_counter == 0);

      // Set the promise
      (*tpg)->_promise.set_value();
      f._topologies.pop_front();
      f._mtx.unlock();
      
      // decrement the topology but since this is not the last we don't notify
      _decrement_topology();

      *tpg = &(f._topologies.front());

      _set_up_topology(*tpg);
      _schedule((*tpg)->_sources);

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
      auto p {std::move((*tpg)->_promise)};

      f._topologies.pop_front();

      f._mtx.unlock();

      // We set the promise in the end in case taskflow leaves before taskflow
      p.set_value();

      _decrement_topology_and_notify();

      // Reset topology so caller can stop execution
      *tpg = nullptr;
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

}  // end of namespace tf -----------------------------------------------------


