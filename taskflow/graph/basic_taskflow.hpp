#pragma once

#include "../threadpool/threadpool.hpp"
#include "flow_builder.hpp"
#include "framework.hpp"

namespace tf {

// Class: BasicTaskflow
// template argument E : executor, the threadpool implementation
template <template <typename...> typename E>
class BasicTaskflow : public FlowBuilder {
  
  using StaticWork  = typename Node::StaticWork;
  using DynamicWork = typename Node::DynamicWork;
  
  // Closure
  struct Closure {
  
    Closure() = default;
    Closure(const Closure&) = delete;
    Closure(Closure&&);
    Closure(BasicTaskflow&, Node&);

    Closure& operator = (Closure&&);
    Closure& operator = (const Closure&) = delete;
    
    void operator ()() const;

    BasicTaskflow* taskflow {nullptr};
    Node*          node     {nullptr};
  };

  public:

  using Executor = E<Closure>;

    explicit BasicTaskflow();
    explicit BasicTaskflow(unsigned);
    explicit BasicTaskflow(std::shared_ptr<Executor>);

    ~BasicTaskflow();
    
    std::shared_ptr<Executor> share_executor();
 
    std::shared_future<void> dispatch();

    template <typename C>
    std::shared_future<void> dispatch(C&&);

    void silent_dispatch();

    template <typename C>
    void silent_dispatch(C&&);

    void wait_for_all();
    void wait_for_topologies();
    void dump(std::ostream&) const;
    void dump_topologies(std::ostream&) const;

    size_t num_nodes() const;
    size_t num_workers() const;
    size_t num_topologies() const;

    std::string dump() const;
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
BasicTaskflow<E>::Closure::Closure(Closure&& rhs) : 
  taskflow {rhs.taskflow}, node {rhs.node} { 
  rhs.taskflow = nullptr;
  rhs.node     = nullptr;
}

// Constructor
template <template <typename...> typename E>
BasicTaskflow<E>::Closure::Closure(BasicTaskflow& t, Node& n) : 
  taskflow{&t}, node {&n} {
}

// Move assignment
template <template <typename...> typename E>
typename BasicTaskflow<E>::Closure& BasicTaskflow<E>::Closure::operator = (Closure&& rhs) {
  taskflow = rhs.taskflow;
  node     = rhs.node;
  rhs.taskflow = nullptr;
  rhs.node     = nullptr;
  return *this;
}

// Operator ()
template <template <typename...> typename E>
void BasicTaskflow<E>::Closure::operator () () const {
  
  assert(taskflow && node);

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
  // The first time we enter into the subflow context, 
  // "subnodes" must be empty.
  // After executing the user's callback on subflow, 
  // there will be at least one node node used as "super source". 
  // The second time we enter this context there is no need
  // to re-execute the work.
  else {
    assert(std::holds_alternative<DynamicWork>(node->_work));
		
    if(!node->_subgraph.has_value()){
      node->_subgraph.emplace();  // Initialize the _subgraph		
		}
    
    SubflowBuilder fb(*(node->_subgraph));

    bool empty_graph = node->_subgraph->empty();

    std::invoke(std::get<DynamicWork>(node->_work), fb);
    
    // Need to create a subflow
    if(empty_graph) {

      auto& S = node->_subgraph->emplace_front([](){});

      S._topology = node->_topology;

      for(auto i = std::next(node->_subgraph->begin()); i != node->_subgraph->end(); ++i) {

        i->_topology = node->_topology;

        if(i->num_successors() == 0) {
          i->precede(fb.detached() ? node->_topology->_target : *node);
        }

        if(i->num_dependents() == 0) {
          S.precede(*i);
        }
      }
      
      // this is for the case where subflow graph might be empty
      if(!fb.detached()) {
        S.precede(*node);
      }

      taskflow->_schedule(S);

      if(!fb.detached()) {
        return;
      }
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
  return std::distance(_graph.begin(), _graph.end());
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

  // Start the taskflow 
  //for(auto src : topology._sources) {
  //  _schedule(*src);
  //} 

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

  // Start the taskflow
  //for(auto src : topology._sources) {
  //  _schedule(*src);
  //}

  _schedule(topology._sources);
}

// Procedure: dispatch 
template <template <typename...> typename E>
std::shared_future<void> BasicTaskflow<E>::dispatch() {

  if(_graph.empty()) {
    return std::async(std::launch::deferred, [](){}).share();
  }

  auto& topology = _topologies.emplace_front(std::move(_graph));

  // Start the taskflow
  //for(auto src : topology._sources) {
  //  _schedule(*src);
  //}
 
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

  // Start the taskflow
  //for(auto src : topology._sources) {
  //  _schedule(*src);
  //}
  
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

  _executor->emplace(std::move(closures));
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

