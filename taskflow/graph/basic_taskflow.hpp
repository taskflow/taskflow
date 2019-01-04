#pragma once

#include "task.hpp"
#include "framework.hpp"

namespace tf {

/** @class BasicTaskflow

@brief The base class to derive a taskflow class.

@tparam E: executor type to use in this taskflow

This class is the base class to derive a taskflow class. 
It inherits all public methods of creating tasks from FlowBuilder
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
    @brief constructs the taskflow with @std_thread_hardware_concurrency worker threads
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

    @return a @std_shared_ptr of the executor
    */
    std::shared_ptr<Executor> share_executor();
    
    /**
    @brief dispatches the present graph to threads and returns immediately

    @return a @std_shared_future to access the execution status of the dispatched graph
    */
    std::shared_future<void> dispatch();
    
    /**
    @brief dispatches the present graph to threads and run a callback when the graph completes

    @return a @std_shared_future to access the execution status of the dispatched graph
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
    @brief dumps the present task dependency graph to a @std_ostream in DOT format

    @param ostream a @std_ostream target
    */
    void dump(std::ostream& ostream) const;

    /**
    @brief dumps the present topologies to a @std_ostream in DOT format

    @param ostream a @std_ostream target
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
    @brief dumps the present task dependency graph in DOT format to a @std_string
    */
    std::string dump() const;
    
    /**
    @brief dumps the existing topologies in DOT format to a @std_string
    */
    std::string dump_topologies() const;

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

// Constructor
template <template <typename...> typename E>
BasicTaskflow<E>::Closure::Closure(BasicTaskflow& t, Node& n) : 
  taskflow{&t}, node {&n} {
}

// Operator ()
template <template <typename...> typename E>
void BasicTaskflow<E>::Closure::operator () () const {
  
  //assert(taskflow && node);

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

    bool first_time {false};  // To stop creating subflow in second time 
    
    // The first time we enter into the subflow context, subgraph must be nullopt.
    if(!(node->_subgraph)){
      node->_subgraph.emplace();  // Initialize the _subgraph   
      first_time = true;
    }
    
    SubflowBuilder fb(*(node->_subgraph));

    std::invoke(std::get<DynamicWork>(node->_work), fb);
    
    // Need to create a subflow if first time & subgraph is not empty
    if(first_time && !(node->_subgraph->empty())) {
      //// small optimization on one task
      //if(std::next(node->_subgraph->begin()) == node->_subgraph->end()){
      //  // If the subgraph has only one node, directly execute this static task 
      //  // regardless of detached or join since this task will be eventually executed 
      //  // some time before the end of the graph.
      //  if(auto &f = node->_subgraph->front(); f._work.index() == 0) { 
      //    std::invoke(std::get<StaticWork>(f._work));
      //  }
      //  else {
      //    f.precede(fb.detached() ? node->_topology->_target : *node);
      //    f._topology = node->_topology;
      //    Closure c(*taskflow, f);
      //    c();
      //    // The reason to return here is this f might spawn new subflows (grandchildren)
      //    // and we need to make sure grandchildren finish before f's parent. So we need to 
      //    // return here. Otherwise, we might execute f's parent even grandchildren are not scheduled 
      //    if(!fb.detached()) {
      //      return;
      //    }
      //  }
      //}
      //else {
        // For storing the source nodes
        std::vector<Node*> src; 
        for(auto n = node->_subgraph->begin(); n != node->_subgraph->end(); ++n) {
          n->_topology = node->_topology;
          if(n->num_successors() == 0) {
            n->precede(fb.detached() ? node->_topology->_target : *node);
          }
          if(n->num_dependents() == 0) {
            src.emplace_back(&(*n));
          }
        }

        for(auto& n: src) {
          taskflow->_schedule(*n);
        }

        if(!fb.detached()) {
          return;
        }
      //}
    }
  }
  
  // At this point, the node/node storage might be destructed.
  for(size_t i=0; i<num_successors; ++i) {
    if(--(node->_successors[i]->_dependents) == 0) {
      taskflow->_schedule(*(node->_successors[i]));
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

};  // end of namespace tf ----------------------------------------------------

