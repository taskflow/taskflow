// 2019/08/03 - modified by Tsung-Wei Huang
//  - made executor thread-safe
//
// 2019/07/26 - modified by Chun-Xun Lin
//  - Combine explore_task & wait_for_task
//  - Remove CAS operations 
//  - Update _num_thieves after pre_wait
//  - TODO: Check will underutilization happen?
//  - TODO: Find out uppper bound (does cycle exist?)
//  - TODO: Does performance drop due to the busy looping after pre_wait?
//
// 2019/07/25 - modified by Tsung-Wei Huang & Chun-Xun Lin
//  - fixed the potential underutilization 
//  - use CAS in both last thief & active worker to make the notification less aggressive
//
// 2019/06/18 - modified by Tsung-Wei Huang
//  - fixed the cache to enable continuity
//  - TODO: do we need a special optimization for 0 workers?
//
// 2019/06/11 - modified by Tsung-Wei Huang
//  - fixed the bug in calling observer while the user
//    may clear the data
//  - added object pool for nodes
//
// 2019/05/17 - modified by Chun-Xun Lin
//  - moved topology to taskflow
// 
// 2019/05/14 - modified by Tsung-Wei Huang
//  - isolated the executor from the taskflow
//
// 2019/04/09 - modified by Tsung-Wei Huang
//  - removed silent_dispatch method
//
// 2019/03/12 - modified by Chun-Xun Lin
//  - added taskflow
//
// 2019/02/11 - modified by Tsung-Wei Huang
//  - refactored run_until
//  - added allocator to topologies
//  - changed to list for topologies
//
// 2019/02/10 - modified by Chun-Xun Lin
//  - added run_n to execute taskflow
//  - finished first peer-review with TW
//
// 2018/07 - 2019/02/09 - missing logs
//
// 2018/06/30 - created by Tsung-Wei Huang
//  - added BasicTaskflow template

// TODO items:
// 1. come up with a better way to remove the "joined" links 
//    during the execution of a static node (1st layer)
//

#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <atomic>
#include <memory>
#include <deque>
#include <optional>
#include <thread>
#include <algorithm>
#include <set>
#include <numeric>
#include <cassert>

#include "spmc_queue.hpp"
#include "notifier.hpp"
#include "observer.hpp"
#include "taskflow.hpp"

namespace tf {

/** @class Executor

@brief The executor class to run a taskflow graph.

An executor object manages a set of worker threads and implements 
an efficient work-stealing scheduling algorithm to run a task graph.

*/
class Executor {
  
  struct Worker {
    std::mt19937 rdgen { std::random_device{}() };
    WorkStealingQueue<Node*> queue;
    std::optional<Node*> cache;
  };
    
  struct PerThread {
    Executor* pool {nullptr}; 
    int worker_id  {-1};
  };

  public:
    
    /**
    @brief constructs the executor with N worker threads
    */
    explicit Executor(unsigned n = std::thread::hardware_concurrency());
    
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

    @return the number of worker threads
    */
    size_t num_workers() const;
    
    /**
    @brief constructs an observer to inspect the activities of worker threads

    Each executor manages at most one observer at a time through std::unique_ptr.
    Createing multiple observers will only keep the lastest one.
    
    @tparam Observer observer type derived from tf::ExecutorObserverInterface
    @tparam ArgsT... argument parameter pack

    @param args arguments to forward to the constructor of the observer
    
    @return a raw pointer to the observer associated with this executor
    */
    template<typename Observer, typename... Args>
    Observer* make_observer(Args&&... args);
    
    /**
    @brief removes the associated observer
    */
    void remove_observer();

  private:
   
    std::condition_variable _topology_cv;
    std::mutex _topology_mutex;
    std::mutex _queue_mutex;

    unsigned _num_topologies {0};
    
    // scheduler field
    std::vector<Worker> _workers;
    std::vector<Notifier::Waiter> _waiters;
    std::vector<std::thread> _threads;

    WorkStealingQueue<Node*> _queue;

    std::atomic<size_t> _num_actives {0};
    std::atomic<size_t> _num_thieves {0};
    std::atomic<bool>   _done        {0};

    Notifier _notifier;
    
    std::unique_ptr<ExecutorObserverInterface> _observer;
    
    unsigned _find_victim(unsigned);

    PerThread& _per_thread() const;

    bool _wait_for_task(unsigned, std::optional<Node*>&);
    
    void _spawn(unsigned);
    void _exploit_task(unsigned, std::optional<Node*>&);
    void _explore_task(unsigned, std::optional<Node*>&);
    void _schedule(Node*, bool);
    void _schedule(PassiveVector<Node*>&);
    void _schedule_unsync(Node*, std::stack<Node*>&) const;
    void _schedule_unsync(PassiveVector<Node*>&, std::stack<Node*>&) const;
    void _invoke(unsigned, Node*);
    void _invoke_unsync(Node*, std::stack<Node*>&) const;
    void _invoke_static_work(unsigned, Node*);
    void _invoke_dynamic_work(unsigned, Node*, Subflow&);
    void _init_module_node(Node*);
    void _init_module_node_unsync(Node*, std::stack<Node*>&) const;
    void _tear_down_topology(Topology*); 
    void _increment_topology();
    void _decrement_topology();
    void _decrement_topology_and_notify();
};

// Constructor
inline Executor::Executor(unsigned N) : 
  _workers  {N},
  _waiters  {N},
  _notifier {_waiters} {
  _spawn(N);
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
}

// Function: num_workers
inline size_t Executor::num_workers() const {
  return _workers.size();
}

// Function: _per_thread
inline Executor::PerThread& Executor::_per_thread() const {
  thread_local PerThread pt;
  return pt;
}

// Procedure: _spawn
inline void Executor::_spawn(unsigned N) {

  // Lock to synchronize all workers before creating _worker_maps
  for(unsigned i=0; i<N; ++i) {
    _threads.emplace_back([this, i] () -> void {

      PerThread& pt = _per_thread();  
      pt.pool = this;
      pt.worker_id = i;
    
      std::optional<Node*> t;
      
      // must use 1 as condition instead of !done
      while(1) {
        
        // execute the tasks.
        _exploit_task(i, t);

        // wait for tasks
        if(_wait_for_task(i, t) == false) {
          break;
        }
      }
      
    });     
  }
}

// Function: _find_victim
inline unsigned Executor::_find_victim(unsigned thief) {
  
  /*unsigned l = 0;
  unsigned r = _workers.size() - 1;
  unsigned vtm = std::uniform_int_distribution<unsigned>{l, r}(
    _workers[thief].rdgen
  );

  // try to look for a task from other workers
  for(unsigned i=0; i<_workers.size(); ++i){

    if((thief == vtm && !_queue.empty()) ||
       (thief != vtm && !_workers[vtm].queue.empty())) {
      return vtm;
    }

    if(++vtm; vtm == _workers.size()) {
      vtm = 0;
    }
  } */

  // try to look for a task from other workers
  for(unsigned vtm=0; vtm<_workers.size(); ++vtm){
    if((thief == vtm && !_queue.empty()) ||
       (thief != vtm && !_workers[vtm].queue.empty())) {
      return vtm;
    }
  }

  return _workers.size();
}

// Function: _explore_task
inline void Executor::_explore_task(unsigned thief, std::optional<Node*>& t) {
  
  //assert(_workers[thief].queue.empty());
  assert(!t);

  const unsigned l = 0;
  const unsigned r = _workers.size() - 1;

  const size_t F = (_workers.size() + 1) << 1;
  const size_t Y = 100;

  size_t f = 0;
  size_t y = 0;

  // explore
  while(!_done) {
  
    unsigned vtm = std::uniform_int_distribution<unsigned>{l, r}(
      _workers[thief].rdgen
    );
      
    t = (vtm == thief) ? _queue.steal() : _workers[vtm].queue.steal();

    if(t) {
      break;
    }

    if(f++ > F) {
      if(std::this_thread::yield(); y++ > Y) {
        break;
      }
    }

    /*if(auto vtm = _find_victim(thief); vtm != _workers.size()) {
      t = (vtm == thief) ? _queue.steal() : _workers[vtm].queue.steal();
      // successful thief
      if(t) {
        break;
      }
    }
    else {
      if(f++ > F) {
        if(std::this_thread::yield(); y++ > Y) {
          break;
        }
      }
    }*/
  }

}

// Procedure: _exploit_task
inline void Executor::_exploit_task(unsigned i, std::optional<Node*>& t) {
  
  assert(!_workers[i].cache);

  if(t) {
    auto& worker = _workers[i];
    if(_num_actives.fetch_add(1) == 0 && _num_thieves == 0) {
      _notifier.notify(false);
    }
    do {
      _invoke(i, *t);

      if(worker.cache) {
        t = *worker.cache;
        worker.cache = std::nullopt;
      }
      else {
        t = worker.queue.pop();
      }

    } while(t);

    --_num_actives;
  }
}

// Function: _wait_for_task
inline bool Executor::_wait_for_task(unsigned me, std::optional<Node*>& t) {

  wait_for_task:

  assert(!t);

  ++_num_thieves;

  explore_task:

  if(_explore_task(me, t); t) {
    if(auto N = _num_thieves.fetch_sub(1); N == 1) {
      _notifier.notify(false);
    }
    return true;
  }

  _notifier.prepare_wait(&_waiters[me]);
  
  //if(auto vtm = _find_victim(me); vtm != _workers.size()) {
  if(!_queue.empty()) {

    _notifier.cancel_wait(&_waiters[me]);
    //t = (vtm == me) ? _queue.steal() : _workers[vtm].queue.steal();

    if(t = _queue.steal(); t) {
      if(auto N = _num_thieves.fetch_sub(1); N == 1) {
        _notifier.notify(false);
      }
      return true;
    }
    else {
      goto explore_task;
    }
  }

  if(_done) {
    _notifier.cancel_wait(&_waiters[me]);
    _notifier.notify(true);
    --_num_thieves;
    return false;
  }

  if(_num_thieves.fetch_sub(1) == 1 && _num_actives) {
    _notifier.cancel_wait(&_waiters[me]);
    goto wait_for_task;
  }
    
  // Now I really need to relinguish my self to others
  _notifier.commit_wait(&_waiters[me]);

  return true;
}

// Function: make_observer    
template<typename Observer, typename... Args>
Observer* Executor::make_observer(Args&&... args) {
  // use a local variable to mimic the constructor 
  auto tmp = std::make_unique<Observer>(std::forward<Args>(args)...);
  tmp->set_up(_workers.size());
  _observer = std::move(tmp);
  return static_cast<Observer*>(_observer.get());
}

// Procedure: remove_observer
inline void Executor::remove_observer() {
  _observer.reset();
}

// Procedure: _schedule_unsync
inline void Executor::_schedule_unsync(
  Node* node, 
  std::stack<Node*>& stack
) const {
  
  // module node need another initialization
  if(node->_module != nullptr && !node->_module->empty() && !node->is_spawned()) {
    _init_module_node_unsync(node, stack);
  }

  stack.push(node);
}

// Procedure: _schedule_unsync
inline void Executor::_schedule_unsync(
  PassiveVector<Node*>& nodes, 
  std::stack<Node*>& stack
) const {
  
  // here we guarantee to run by a thread so no need to cache the
  // size from nodes
  for(auto node : nodes) {
    if(node->_module != nullptr && !node->_module->empty() && !node->is_spawned()) {
      _init_module_node_unsync(node, stack);
    }
    stack.push(node);
  }
}

// Procedure: _schedule
// The main procedure to schedule a give task node.
// Each task node has two types of tasks - regular and subflow.
inline void Executor::_schedule(Node* node, bool bypass) {
  
  assert(_workers.size() != 0);
  
  // module node need another initialization
  if(node->_module != nullptr && !node->_module->empty() && !node->is_spawned()) {
    _init_module_node(node);
  }
  
  // caller is a worker to this pool
  if(auto& pt = _per_thread(); pt.pool == this) {
    if(!bypass) {
      _workers[pt.worker_id].queue.push(node);
    }
    else {
      assert(!_workers[pt.worker_id].cache);
      _workers[pt.worker_id].cache = node;
    }
    return;
  }

  // other threads
  {
    std::scoped_lock lock(_queue_mutex);
    _queue.push(node);
  }

  _notifier.notify(false);
}

// Procedure: _schedule
// The main procedure to schedule a set of task nodes.
// Each task node has two types of tasks - regular and subflow.
inline void Executor::_schedule(PassiveVector<Node*>& nodes) {

  assert(_workers.size() != 0);
  
  // We need to cacth the node count to avoid accessing the nodes
  // vector while the parent topology is removed!
  const auto num_nodes = nodes.size();
  
  if(num_nodes == 0) {
    return;
  }

  for(auto node : nodes) {
    if(node->_module != nullptr && !node->_module->empty() && !node->is_spawned()) {
      _init_module_node(node);
    }
  }

  // worker thread
  if(auto& pt = _per_thread(); pt.pool == this) {
    for(size_t i=0; i<num_nodes; ++i) {
      _workers[pt.worker_id].queue.push(nodes[i]);
    }
    return;
  }
  
  // other threads
  {
    std::scoped_lock lock(_queue_mutex);
    for(size_t k=0; k<num_nodes; ++k) {
      _queue.push(nodes[k]);
    }
  }

  if(num_nodes >= _workers.size()) {
    _notifier.notify(true);
  }
  else {
    for(size_t k=0; k<num_nodes; ++k) {
      _notifier.notify(false);
    }
  }
}

// Procedure: _init_module_node
inline void Executor::_init_module_node(Node* node) {

  node->_work = [node=node, this, tgt{PassiveVector<Node*>()}] () mutable {

    // second time to enter this context
    if(node->is_spawned()) {
      node->_dependents.resize(node->_dependents.size()-tgt.size());
      for(auto& t: tgt) {
        t->_successors.clear();
      }
      return ;
    }

    // first time to enter this context
    node->set_spawned();

    PassiveVector<Node*> src;

    for(auto& n: node->_module->_graph.nodes()) {
      n->_topology = node->_topology;
      if(n->num_dependents() == 0) {
        src.push_back(n.get());
      }
      if(n->num_successors() == 0) {
        n->precede(*node);
        tgt.push_back(n.get());
      }
    }

    _schedule(src);
  };
}

// Procedure: _init_module_node_unsync
inline void Executor::_init_module_node_unsync(
  Node* node, 
  std::stack<Node*>& stack
) const {

  node->_work = [this, node=node, &stack, tgt{PassiveVector<Node*>()}] () mutable {

    // second time to enter this context
    if(node->is_spawned()) {
      node->_dependents.resize(node->_dependents.size()-tgt.size());
      for(auto& t: tgt) {
        t->_successors.clear();
      }
      return ;
    }

    // first time to enter this context
    node->set_spawned();

    PassiveVector<Node*> src;

    for(auto& n: node->_module->_graph.nodes()) {
      n->_topology = node->_topology;
      if(n->num_dependents() == 0) {
        src.push_back(n.get());
      }
      if(n->num_successors() == 0) {
        n->precede(*node);
        tgt.push_back(n.get());
      }
    }

    _schedule_unsync(src, stack);
  };
}

// Procedure: _invoke
inline void Executor::_invoke(unsigned me, Node* node) {

  assert(_workers.size() != 0);

  // Here we need to fetch the num_successors first to avoid the invalid memory
  // access caused by topology clear.
  const auto num_successors = node->num_successors();

  // static task
  // The default node work type. We only need to execute the callback if any.
  if(auto index=node->_work.index(); index == 1) {
    if(node->_module != nullptr) {
      bool first_time = !node->is_spawned();
      _invoke_static_work(me, node);
      if(first_time) {
        return ;
      }
    }
    else {
      _invoke_static_work(me, node);
    }
  }
  // dynamic task
  else if (index == 2){
    
    // Clear the subgraph before the task execution
    if(!node->is_spawned()) {
      if(node->_subgraph) {
        node->_subgraph->clear();
      }
      else {
        node->_subgraph.emplace();
      }
    }
   
    Subflow fb(*(node->_subgraph));

    _invoke_dynamic_work(me, node, fb);
    
    // Need to create a subflow if first time & subgraph is not empty 
    if(!node->is_spawned()) {
      node->set_spawned();
      if(!node->_subgraph->empty()) {
        // For storing the source nodes
        PassiveVector<Node*> src; 
        for(auto& n: node->_subgraph->nodes()) {
          n->_topology = node->_topology;
          n->set_subtask();
          if(n->num_successors() == 0) {
            if(fb.detached()) {
              node->_topology->_num_sinks++;
            }
            else {
              n->precede(*node);
            }
          }
          if(n->num_dependents() == 0) {
            src.push_back(n.get());
          }
        }

        _schedule(src);

        if(fb.joined()) {
          return;
        }
      }
    }
  } // End of DynamicWork -----------------------------------------------------
  
  // Recover the runtime change due to dynamic tasking except the target & spawn tasks 
  // This must be done before scheduling the successors, otherwise this might cause 
  // race condition on the _dependents
  //if(num_successors && !node->_subtask) {
  if(!node->is_subtask()) {
    // Only dynamic tasking needs to restore _dependents
    // TODO:
    if(node->_work.index() == 2 && !node->_subgraph->empty()) {
      while(!node->_dependents.empty() && node->_dependents.back()->is_subtask()) {
        node->_dependents.pop_back();
      }
    }
    node->_num_dependents = static_cast<int>(node->_dependents.size());
    node->unset_spawned();
  }

  // At this point, the node storage might be destructed.
  Node* cache {nullptr};

  for(size_t i=0; i<num_successors; ++i) {
    if(--(node->_successors[i]->_num_dependents) == 0) {
      if(cache) {
        _schedule(cache, false);
      }
      cache = node->_successors[i];
    }
  }

  if(cache) {
    _schedule(cache, true);
  }

  // A node without any successor should check the termination of topology
  if(num_successors == 0) {
    if(--(node->_topology->_num_sinks) == 0) {
      _tear_down_topology(node->_topology);
    }
  }
}

// Procedure: _invoke_static_work
inline void Executor::_invoke_static_work(unsigned me, Node* node) {
  if(_observer) {
    _observer->on_entry(me, TaskView(node));
    std::invoke(std::get<Node::StaticWork>(node->_work));
    _observer->on_exit(me, TaskView(node));
  }
  else {
    std::invoke(std::get<Node::StaticWork>(node->_work));
  }
}

// Procedure: _invoke_dynamic_work
inline void Executor::_invoke_dynamic_work(unsigned me, Node* node, Subflow& sf) {
  if(_observer) {
    _observer->on_entry(me, TaskView(node));
    std::invoke(std::get<Node::DynamicWork>(node->_work), sf);
    _observer->on_exit(me, TaskView(node));
  }
  else {
    std::invoke(std::get<Node::DynamicWork>(node->_work), sf);
  }
}

// Procedure: _invoke_unsync
inline void Executor::_invoke_unsync(Node* node, std::stack<Node*>& stack) const {

  const auto num_successors = node->num_successors();

  // static task
  // The default node work type. We only need to execute the callback if any.
  if(auto index=node->_work.index(); index == 1) {
    if(node->_module != nullptr) {
      bool first_time = !node->is_spawned();
      std::invoke(std::get<Node::StaticWork>(node->_work));
      if(first_time) {
        return ;
      }
    }
    else {
      std::invoke(std::get<Node::StaticWork>(node->_work));
    }
  }
  // dynamic task
  else if (index == 2){
    
    // Clear the subgraph before the task execution
    if(!node->is_spawned()) {
      if(node->_subgraph) {
        node->_subgraph->clear();
      }
      else {
        node->_subgraph.emplace();
      }
    }
   
    Subflow fb(*(node->_subgraph));

    std::invoke(std::get<Node::DynamicWork>(node->_work), fb);
    
    // Need to create a subflow if first time & subgraph is not empty 
    if(!node->is_spawned()) {
      node->set_spawned();
      if(!node->_subgraph->empty()) {
        // For storing the source nodes
        PassiveVector<Node*> src; 
        for(auto& n: node->_subgraph->nodes()) {
          n->_topology = node->_topology;
          n->set_subtask();
          if(n->num_successors() == 0) {
            if(fb.detached()) {
              node->_topology->_num_sinks++;
            }
            else {
              n->precede(*node);
            }
          }
          if(n->num_dependents() == 0) {
            src.push_back(n.get());
          }
        }

        _schedule_unsync(src, stack);

        if(fb.joined()) {
          return;
        }
      }
    }
  } // End of DynamicWork -----------------------------------------------------
  
  // Recover the runtime change due to dynamic tasking except the target & spawn tasks 
  // This must be done before scheduling the successors, otherwise this might cause 
  // race condition on the _dependents
  //if(num_successors && !node->_subtask) {
  if(!node->is_subtask()) {
    // Only dynamic tasking needs to restore _dependents
    // TODO:
    if(node->_work.index() == 2 && !node->_subgraph->empty()) {
      while(!node->_dependents.empty() && node->_dependents.back()->is_subtask()) {
        node->_dependents.pop_back();
      }
    }
    node->_num_dependents = static_cast<int>(node->_dependents.size());
    node->unset_spawned();
  }

  // At this point, the node storage might be destructed.
  for(size_t i=0; i<num_successors; ++i) {
    if(--(node->_successors[i]->_num_dependents) == 0) {
      _schedule_unsync(node->_successors[i], stack);
    }
  }

  // A node without any successor should check the termination of topology
  if(num_successors == 0) {
    --(node->_topology->_num_sinks);
  }
}

// Function: run
inline std::future<void> Executor::run(Taskflow& f) {
  return run_n(f, 1, [](){});
}

// Function: run
template <typename C>
std::future<void> Executor::run(Taskflow& f, C&& c) {
  static_assert(std::is_invocable<C>::value);
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

// Function: _tear_down_topology
inline void Executor::_tear_down_topology(Topology* tpg) {

  auto &f = tpg->_taskflow;

  //assert(&tpg == &(f._topologies.front()));

  // case 1: we still need to run the topology again
  if(!std::invoke(tpg->_pred)) {
    tpg->_recover_num_sinks();
    _schedule(tpg->_sources); 
  }
  // case 2: the final run of this topology
  else {
    
    if(tpg->_call != nullptr) {
      std::invoke(tpg->_call);
    }

    f._mtx.lock();

    // If there is another run (interleave between lock)
    if(f._topologies.size() > 1) {

      // Set the promise
      tpg->_promise.set_value();
      f._topologies.pop_front();
      f._mtx.unlock();
      
      // decrement the topology but since this is not the last we don't notify
      _decrement_topology();

      f._topologies.front()._bind(f._graph);
      _schedule(f._topologies.front()._sources);
    }
    else {
      assert(f._topologies.size() == 1);

      // Need to back up the promise first here becuz taskflow might be 
      // destroy before taskflow leaves
      auto p {std::move(tpg->_promise)};

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

  // Predicate must return a boolean value
  static_assert(std::is_invocable_v<C> && std::is_invocable_v<P>);
  
  _increment_topology();

  // Special case of predicate
  if(std::invoke(pred)) {
    std::promise<void> promise;
    promise.set_value();
    _decrement_topology_and_notify();
    return promise.get_future();
  }
  
  // Special case of zero workers requires:
  //  - iterative execution to avoid stack overflow
  //  - avoid execution of last_work
  if(_workers.size() == 0 || f.empty()) {
    
    Topology tpg(f, std::forward<P>(pred), std::forward<C>(c));

    // Clear last execution data & Build precedence between nodes and target
    tpg._bind(f._graph);

    std::stack<Node*> stack;

    do {
      _schedule_unsync(tpg._sources, stack);
      while(!stack.empty()) {
        auto node = stack.top();
        stack.pop();
        _invoke_unsync(node, stack);
      }
      tpg._recover_num_sinks();
    } while(!std::invoke(tpg._pred));

    if(tpg._call != nullptr) {
      std::invoke(tpg._call);
    }

    tpg._promise.set_value();
    
    _decrement_topology_and_notify();
    
    return tpg._promise.get_future();
  }
  
  // Multi-threaded execution.
  bool run_now {false};
  Topology* tpg;
  std::future<void> future;
  
  {
    std::scoped_lock lock(f._mtx);

    // create a topology for this run
    tpg = &(f._topologies.emplace_back(f, std::forward<P>(pred), std::forward<C>(c)));
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
    tpg->_bind(f._graph);
    _schedule(tpg->_sources);
  }

  return future;
}

// Procedure: _increment_topology
inline void Executor::_increment_topology() {
  std::scoped_lock lock(_topology_mutex);
  ++_num_topologies;
}

// Procedure: _decrement_topology_and_notify
inline void Executor::_decrement_topology_and_notify() {
  std::scoped_lock lock(_topology_mutex);
  if(--_num_topologies == 0) {
    _topology_cv.notify_all();
  }
}

// Procedure: _decrement_topology
inline void Executor::_decrement_topology() {
  std::scoped_lock lock(_topology_mutex);
  --_num_topologies;
}

// Procedure: wait_for_all
inline void Executor::wait_for_all() {
  std::unique_lock lock(_topology_mutex);
  _topology_cv.wait(lock, [&](){ return _num_topologies == 0; });
}

}  // end of namespace tf -----------------------------------------------------


