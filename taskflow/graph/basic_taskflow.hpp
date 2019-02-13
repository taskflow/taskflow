#pragma once

#include "topology.hpp"

namespace tf {

/** @class BasicTaskflow

@brief The base class to derive a taskflow class.

@tparam E: executor type to use in this taskflow

This class is the base class to derive a taskflow class. 
It inherits all public methods to create tasks from tf::FlowBuilder
and defines means to execute task dependency graphs.

*/
template <template <typename...> typename E>
class BasicTaskflow : public FlowBuilder {
  
  using StaticWork  = typename Node::StaticWork;
  using DynamicWork = typename Node::DynamicWork;
  
  // Closure
  struct Closure {
  
    Closure() = default;
    Closure(const Closure&) = default;
    Closure(BasicTaskflow&, Node&);

    Closure& operator = (const Closure&) = default;
    
    void operator ()() const;

    BasicTaskflow* taskflow {nullptr};
    Node*          node     {nullptr};
  };

  public:
  
  /**
  @typedef Executor

  @brief alias of executor type
  */
  using Executor = E<Closure>;
    
    /**
    @brief constructs the taskflow with std::thread::hardware_concurrency worker threads
    */
    explicit BasicTaskflow();
    
    /**
    @brief constructs the taskflow with N worker threads
    */
    explicit BasicTaskflow(unsigned N);
    
    /**
    @brief constructs the taskflow with a given executor
    */
    explicit BasicTaskflow(std::shared_ptr<Executor> executor);
    
    /**
    @brief destructs the taskflow

    Destructing a taskflow object will first wait for all running topologies to finish
    and then clean up all associated data storages.
    */
    ~BasicTaskflow();
    
    /**
    @brief shares ownership of the executor associated with this taskflow object

    @return a std::shared_ptr of the executor
    */
    std::shared_ptr<Executor> share_executor();
    
    /**
    @brief dispatches the present graph to threads and returns immediately

    @return a std::shared_future to access the execution status of the dispatched graph
    */
    std::shared_future<void> dispatch();
    
    /**
    @brief dispatches the present graph to threads and run a callback when the graph completes

    @return a std::shared_future to access the execution status of the dispatched graph
    */
    template <typename C>
    std::shared_future<void> dispatch(C&&);
  
    /**
    @brief dispatches the present graph to threads and returns immediately
    */
    void silent_dispatch();
    
    /**
    @brief dispatches the present graph to threads and run a callback when the graph completes

    @param callable a callable object to execute on completion
    */
    template <typename C>
    void silent_dispatch(C&& callable);
    
    /**
    @brief dispatches the present graph to threads and wait for all topologies to complete
    */
    void wait_for_all();

    /**
    @brief blocks until all running topologies complete and then
           cleans up all associated storages
    */
    void wait_for_topologies();
    
    /**
    @brief dumps the present task dependency graph to a std::ostream in DOT format

    @param ostream a std::ostream target
    */
    void dump(std::ostream& ostream) const;

    /**
    @brief dumps the present topologies to a std::ostream in DOT format

    @param ostream a std::ostream target
    */
    void dump_topologies(std::ostream& ostream) const;
    
    /**
    @brief queries the number of nodes in the present task dependency graph
    */
    size_t num_nodes() const;

    /**
    @brief queries the number of worker threads in the associated executor
    */
    size_t num_workers() const;

    /**
    @brief queries the number of existing topologies
    */
    size_t num_topologies() const;
    
    /**
    @brief dumps the present task dependency graph in DOT format to a std::string
    */
    std::string dump() const;
    
    /**
    @brief dumps the existing topologies in DOT format to a std::string
    */
    std::string dump_topologies() const;

    /**
    @brief runs the framework once
    
    @param framework a tf::Framework

    @return a std::shared_future to access the execution status of the framework
    */
    std::shared_future<void> run(Framework& framework);

    /**
    @brief runs the framework once and invoke a callback upon completion

    @param framework a tf::Framework 
    @param callable a callable object to be invoked after every run

    @return a std::shared_future to access the execution status of the framework
    */
    template<typename C>
    std::shared_future<void> run(Framework& framework, C&& callable);

    /**
    @brief runs the framework for N times
    
    @param framework a tf::Framework 
    @param N number of runs

    @return a std::shared_future to access the execution status of the framework
    */
    std::shared_future<void> run_n(Framework& framework, size_t N);

    /**
    @brief runs the framework for N times and invokes a callback upon completion

    @param framework a tf::Framework 
    @param N number of runs
    @param callable a callable object to be invoked after every run

    @return a std::shared_future to access the execution status of the framework
    */
    template<typename C>
    std::shared_future<void> run_n(Framework& framework, size_t N, C&& callable);

    /**
    @brief runs the framework multiple times until the predicate becomes true and invoke a callback

    @param framework a tf::Framework 
    @param P predicate (a callable object returns true or false)

    @return a std::shared_future to access the execution status of the framework
    */
    template<typename P>
    std::shared_future<void> run_until(Framework& framework, P&& predicate);

    /**
    @brief runs the framework multiple times until the predicate becomes true and invoke a callback

    @param framework a tf::Framework 
    @param P predicate (a callable object returns true or false)
    @param callable a callable object to be invoked after every run

    @return a std::shared_future to access the execution status of the framework
    */
    template<typename P, typename C>
    std::shared_future<void> run_until(Framework& framework, P&& predicate, C&& callable);

  private:
    
    Graph _graph;

    std::shared_ptr<Executor> _executor;

    std::forward_list<Topology> _topologies;

    void _schedule(Node&);
    void _schedule(std::vector<Node*>&);
};

// ============================================================================
// BasicTaskflow::Closure Method Definitions
// ============================================================================

// Function: run
template <template <typename...> typename E>
std::shared_future<void> BasicTaskflow<E>::run(Framework& f) {
  return run_n(f, 1, [](){});
}

// Function: run
template <template <typename...> typename E>
template <typename C>
std::shared_future<void> BasicTaskflow<E>::run(Framework& f, C&& c) {
  static_assert(std::is_invocable<C>::value);
  return run_n(f, 1, std::forward<C>(c));
}

// Function: run_n
template <template <typename...> typename E>
std::shared_future<void> BasicTaskflow<E>::run_n(Framework& f, size_t repeat) {
  return run_n(f, repeat, [](){});
}

// Function: run_n
template <template <typename...> typename E>
template <typename C>
std::shared_future<void> BasicTaskflow<E>::run_n(Framework& f, size_t repeat, C&& c) {
  return run_until(f, [repeat]() mutable { return repeat-- == 0; }, std::forward<C>(c));
}


// Function: run_until
template <template <typename...> typename E>
template <typename P>
std::shared_future<void> BasicTaskflow<E>::run_until(Framework& f, P&& predicate) {
  return run_until(f, std::forward<P>(predicate), [](){});
}

// Function: run_until
template <template <typename...> typename E>
template <typename P, typename C>
std::shared_future<void> BasicTaskflow<E>::run_until(Framework& f, P&& predicate, C&& c) {

  // Predicate must return a boolean value
  static_assert(std::is_invocable_v<C> && std::is_same_v<bool, std::invoke_result_t<P>>);

  if(std::invoke(predicate)) {
    return std::async(std::launch::deferred, [](){}).share();
  }
  
  auto &tpg = _topologies.emplace_front(f, std::forward<P>(predicate));

  // TODO (clin99): after PV (2/12)
  // 1. move setup_topology to the constructor
  // 2. 


  std::scoped_lock lock(f._mtx);

  f._topologies.push_back(&tpg);
  
  const auto setup_topology = [](auto& f, auto& tpg) {
    
    // PV (2/12) 
    // 1. assert num_sinks == 0 ?
    // 2. clear sources?
    // 3. move this guy to constructor ... ?
    assert(tpg._num_sinks == 0);
    assert(tpg._sources.empty());

    for(auto& n: f._graph) {
      // reset the target links
      n._topology = &tpg;
      if(n.num_dependents() == 0) {
        tpg._sources.push_back(&n);
      }
      if(n.num_successors() == 0) {
        tpg._num_sinks ++;       
      }
    }
  };

  // Iterative execution to avoid stack overflow
  if(num_workers() == 0) {
    // Clear last execution data & Build precedence between nodes and target
    setup_topology(f, tpg);

    const int tgt_predecessor = tpg._num_sinks; 
    do {
      _schedule(tpg._sources);
      f._topologies.front()->_num_sinks = tgt_predecessor;
    } while(!std::invoke(tpg._predicate));

    std::invoke(c);
    auto &p = f._topologies.front()->_promise;
    f._topologies.pop_front();
    p.set_value();
  }
  else { 
    // case 1: the previous execution is still running
    if(f._topologies.size() > 1) {
      // PV (2/10): skip the predicate?
      tpg._work = std::forward<C>(c);
    }
    // case 2: this epoch should run
    else {
      setup_topology(f, tpg);

      //Set up target node's work
      tpg._work = [
        &f, 
        c=std::function<void()>{std::forward<C>(c)}, 
        tgt_predecessor=tpg._num_sinks.load(std::memory_order_relaxed), 
        this
      ] () mutable {

        // f._topologies.front is myself

        // PV 1/31 (twhuang): thread safety? 
        // case 1: we still need to run the topology again
        if(!std::invoke(f._topologies.front()->_predicate)) {
        //if(--f._topologies.front()->_repeat != 0) {
          f._topologies.front()->_num_sinks = tgt_predecessor;
          _schedule(f._topologies.front()->_sources); 
        }
        // case 2: the final run of this topology
        // notice that there can be another new run request before we acquire the lock
        else {
          std::invoke(c);

          f._mtx.lock();

          // If there is another run
          if(f._topologies.size() > 1) {

            // Set the promise
            f._topologies.front()->_promise.set_value();
            
            // PV (2/10): why not just using the next_tpg other than moving all
            // things to the previous one
            auto next_tpg = std::next(f._topologies.begin());
            //c = std::move(std::get<StaticWork>((*next_tpg)->_target._work));
            c = std::move((*next_tpg)->_work);

            f._topologies.front()->_predicate = std::move((*next_tpg)->_predicate);
            f._topologies.front()->_promise = std::move((*next_tpg)->_promise);
            f._topologies.erase(next_tpg);

            f._mtx.unlock();
            // The graph should be exactly the same as previous dispatch
            f._topologies.front()->_num_sinks = tgt_predecessor;

            _schedule(f._topologies.front()->_sources);
          }
          else {
            assert(f._topologies.size() == 1);
            // Need to back up the promise first here becuz framework might be 
            // destroy before taskflow leaves
            auto &p = f._topologies.front()->_promise; 
            f._topologies.pop_front();
            f._mtx.unlock();
           
            // We set the promise in the end in case framework leaves before taskflow
            p.set_value();
          }
        }
      }; // End of target's work ------------------------------------------------

      _schedule(tpg._sources);
    }
  }
    
  return tpg._future;
}

// Constructor
template <template <typename...> typename E>
BasicTaskflow<E>::Closure::Closure(BasicTaskflow& t, Node& n) : 
  taskflow{&t}, node {&n} {
}

// Operator ()
template <template <typename...> typename E>
void BasicTaskflow<E>::Closure::operator () () const {

  // Here we need to fetch the num_successors first to avoid the invalid memory
  // access caused by topology clear.
  const auto num_successors = node->num_successors();
  
  // regular node type
  // The default node work type. We only need to execute the callback if any.
  if(auto index=node->_work.index(); index == 0) {
    if(auto &f = std::get<StaticWork>(node->_work); f != nullptr){
      std::invoke(f);
    }
  }
  // subflow node type 
  else {
    
    // Clear the subgraph before the task execution
    if(!node->is_spawned()) {
      node->_subgraph.emplace();
    }
   
    SubflowBuilder fb(*(node->_subgraph));

    std::invoke(std::get<DynamicWork>(node->_work), fb);
    
    // Need to create a subflow if first time & subgraph is not empty 
    if(!node->is_spawned()) {
      node->set_spawned();
      if(!node->_subgraph->empty()) {
        // For storing the source nodes
        std::vector<Node*> src; 
        for(auto n = node->_subgraph->begin(); n != node->_subgraph->end(); ++n) {
          n->_topology = node->_topology;
          n->set_subtask();
          if(n->num_successors() == 0) {
            if(fb.detached()) {
              node->_topology->_num_sinks ++;
            }
            else {
              n->precede(*node);
            }
          }
          if(n->num_dependents() == 0) {
            src.emplace_back(&(*n));
          }
        }

        taskflow->_schedule(src);

        if(!fb.detached()) {
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
    // Only dynamic tasking needs to restore _predecessors
    if(node->_work.index() == 1 &&  !node->_subgraph->empty()) {
      while(!node->_predecessors.empty() && node->_predecessors.back()->is_subtask()) {
        node->_predecessors.pop_back();
      }
    }
    node->_dependents = node->_predecessors.size();
    node->clear_status();
  }

  // At this point, the node storage might be destructed.
  for(size_t i=0; i<num_successors; ++i) {
    if(--(node->_successors[i]->_dependents) == 0) {
      taskflow->_schedule(*(node->_successors[i]));
    }
  }

  // A node without any successor should check the termination of topology
  if(num_successors == 0) {
    if(--(node->_topology->_num_sinks) == 0) {

      // This is the last executing node 
      bool is_framework = node->_topology->_handle.index() == 1;
      if(node->_topology->_work != nullptr) {
        std::invoke(node->_topology->_work);
      }
      if(!is_framework) {
        node->_topology->_promise.set_value();
      }
    }
  }
}

// ============================================================================
// BasicTaskflow Method Definitions
// ============================================================================

// Constructor
template <template <typename...> typename E>
BasicTaskflow<E>::BasicTaskflow() : 
  FlowBuilder {_graph},
  _executor {std::make_shared<Executor>(std::thread::hardware_concurrency())} {
}

// Constructor
template <template <typename...> typename E>
BasicTaskflow<E>::BasicTaskflow(unsigned N) : 
  FlowBuilder {_graph},
  _executor {std::make_shared<Executor>(N)} {
}

// Constructor
template <template <typename...> typename E>
BasicTaskflow<E>::BasicTaskflow(std::shared_ptr<Executor> e) :
  FlowBuilder {_graph},
  _executor {std::move(e)} {

  if(_executor == nullptr) {
    TF_THROW(Error::EXECUTOR, 
      "failed to construct taskflow (executor cannot be null)"
    );
  }
}

// Destructor
template <template <typename...> typename E>
BasicTaskflow<E>::~BasicTaskflow() {
  wait_for_topologies();
}

// Function: num_nodes
template <template <typename...> typename E>
size_t BasicTaskflow<E>::num_nodes() const {
  return _graph.size();
}

// Function: num_workers
template <template <typename...> typename E>
size_t BasicTaskflow<E>::num_workers() const {
  return _executor->num_workers();
}

// Function: num_topologies
template <template <typename...> typename E>
size_t BasicTaskflow<E>::num_topologies() const {
  return std::distance(_topologies.begin(), _topologies.end());
}

// Function: share_executor
template <template <typename...> typename E>
std::shared_ptr<typename BasicTaskflow<E>::Executor> BasicTaskflow<E>::share_executor() {
  return _executor;
}

// Procedure: silent_dispatch 
template <template <typename...> typename E>
void BasicTaskflow<E>::silent_dispatch() {

  if(_graph.empty()) return;

  auto& topology = _topologies.emplace_front(std::move(_graph));

  _schedule(topology._sources);
}


// Procedure: silent_dispatch with registered callback
template <template <typename...> typename E>
template <typename C>
void BasicTaskflow<E>::silent_dispatch(C&& c) {

  if(_graph.empty()) {
    c();
    return;
  }

  auto& topology = _topologies.emplace_front(std::move(_graph), std::forward<C>(c));

  _schedule(topology._sources);
}

// Procedure: dispatch 
template <template <typename...> typename E>
std::shared_future<void> BasicTaskflow<E>::dispatch() {

  if(_graph.empty()) {
    return std::async(std::launch::deferred, [](){}).share();
  }

  auto& topology = _topologies.emplace_front(std::move(_graph));
 
  _schedule(topology._sources);

  return topology._future;
}


// Procedure: dispatch with registered callback
template <template <typename...> typename E>
template <typename C>
std::shared_future<void> BasicTaskflow<E>::dispatch(C&& c) {

  if(_graph.empty()) {
    c();
    return std::async(std::launch::deferred, [](){}).share();
  }

  auto& topology = _topologies.emplace_front(std::move(_graph), std::forward<C>(c));

  _schedule(topology._sources);

  return topology._future;
}

// Procedure: wait_for_all
template <template <typename...> typename E>
void BasicTaskflow<E>::wait_for_all() {
  if(!_graph.empty()) {
    silent_dispatch();
  }
  wait_for_topologies();
}

// Procedure: wait_for_topologies
template <template <typename...> typename E>
void BasicTaskflow<E>::wait_for_topologies() {
  for(auto& t: _topologies){
    t._future.get();
  }
  _topologies.clear();
}

// Procedure: _schedule
// The main procedure to schedule a give task node.
// Each task node has two types of tasks - regular and subflow.
template <template <typename...> typename E>
void BasicTaskflow<E>::_schedule(Node& node) {
  _executor->emplace(*this, node);
}


// Procedure: _schedule
// The main procedure to schedule a set of task nodes.
// Each task node has two types of tasks - regular and subflow.
template <template <typename...> typename E>
void BasicTaskflow<E>::_schedule(std::vector<Node*>& nodes) {
  std::vector<Closure> closures;
  closures.reserve(nodes.size());
  for(auto src : nodes) {
    closures.emplace_back(*this, *src);
  }
  _executor->batch(std::move(closures));
}

// Function: dump_topologies
template <template <typename...> typename E>
std::string BasicTaskflow<E>::dump_topologies() const {
  
  std::ostringstream os;

  for(const auto& tpg : _topologies) {
    tpg.dump(os);
  }
  
  return os.str();
}

// Function: dump_topologies
template <template <typename...> typename E>
void BasicTaskflow<E>::dump_topologies(std::ostream& os) const {
  for(const auto& tpg : _topologies) {
    tpg.dump(os);
  }
}

// Function: dump
template <template <typename...> typename E>
void BasicTaskflow<E>::dump(std::ostream& os) const {

  os << "digraph Taskflow {\n";
  
  for(const auto& node : _graph) {
    node.dump(os);
  }

  os << "}\n";
}

// Function: dump
// Dumps the taskflow in graphviz. 
// The result can be viewed at http://www.webgraphviz.com/.
template <template <typename...> typename E>
std::string BasicTaskflow<E>::dump() const {
  std::ostringstream os;
  dump(os); 
  return os.str();
}

}  // end of namespace tf ----------------------------------------------------

