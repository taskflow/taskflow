#pragma once

#include <stack>

#include "flow_builder.hpp"
#include "topology.hpp"

/** 
@file core/taskflow.hpp
@brief taskflow include file
*/

namespace tf {

// ----------------------------------------------------------------------------

/**
@class Taskflow 

@brief main entry to create a task dependency graph

A %taskflow manages a task dependency graph where each task represents a 
callable object (e.g., @std_lambda, @std_function) and an edge represents a 
dependency between two tasks. A task is one of the following five types:
  
  1. static task: the callable constructible from 
                  @c std::function<void()>
  2. dynamic task: the callable constructible from 
                   @c std::function<void(tf::Subflow&)>
  3. condition task: the callable constructible from 
                     @c std::function<int()>
  4. module task: the task constructed from tf::Taskflow::composed_of
  5. %cudaFlow task: the callable constructible from 
                     @c std::function<void(tf::cudaFlow)> or
                     @c std::function<void(tf::cudaFlowCapturer)>

The following example creates a simple taskflow graph of four static tasks, 
@c A, @c B, @c C, and @c D, where
@c A runs before @c B and @c C and 
@c D runs after  @c B and @c C.

@code{.cpp}
tf::Executor executor;
tf::Taskflow taskflow("simple");

auto [A, B, C, D] = taskflow.emplace(
  []() { std::cout << "TaskA\n"; },
  []() { std::cout << "TaskB\n"; },
  []() { std::cout << "TaskC\n"; },
  []() { std::cout << "TaskD\n"; }
);

A.precede(B, C);  // A runs before B and C
D.succeed(B, C);  // D runs after  B and C
                                   
executor.run(taskflow).wait();     
@endcode

Please refer to @ref Cookbook to learn more about each task type.

*/
class Taskflow : public FlowBuilder {

  friend class Topology;
  friend class Executor;
  friend class FlowBuilder;

  struct Dumper {
    std::stack<const Taskflow*> stack;
    std::unordered_set<const Taskflow*> visited;
  };

  public:

    /**
    @brief constructs a taskflow with the given name
    */
    Taskflow(const std::string& name);

    /**
    @brief constructs a taskflow
    */
    Taskflow();

    /**
    @brief dumps the taskflow to a DOT format through an output stream
           using the stream insertion operator @c <<
    */
    void dump(std::ostream& ostream) const;
    
    /**
    @brief dumps the taskflow to a std::string of DOT format
    */
    std::string dump() const;
    
    /**
    @brief queries the number of tasks in the taskflow
    */
    size_t num_tasks() const;
    
    /**
    @brief queries the emptiness of the taskflow
    */
    bool empty() const;

    /**
    @brief sets the name of the taskflow
    */
    void name(const std::string&); 

    /**
    @brief queries the name of the taskflow
    */
    const std::string& name() const ;
    
    /**
    @brief clears the associated task dependency graph
    */
    void clear();

    /**
    @brief applies an visitor callable to each task in the taskflow

    The visitor is a callable that takes an argument of type tf::Task
    and returns nothing. The following example iterates each task in a
    taskflow and prints its name:

    @code{.cpp}
    taskflow.for_each_task([](tf::Task task){
      std::cout << task.name() << '\n';
    });
    @endcode
    */
    template <typename V>
    void for_each_task(V&& visitor) const;

  private:
 
    std::string _name;
   
    Graph _graph;

    std::mutex _mtx;

    std::list<Topology> _topologies;
    
    void _dump(std::ostream&, const Taskflow*) const;
    void _dump(std::ostream&, const Node*, Dumper&) const;
    void _dump(std::ostream&, const Graph&, Dumper&) const;
};

// Constructor
inline Taskflow::Taskflow(const std::string& name) : 
  FlowBuilder {_graph},
  _name       {name} {
}

// Constructor
inline Taskflow::Taskflow() : FlowBuilder{_graph} {
}

// Procedure:
inline void Taskflow::clear() {
  _graph.clear();
}

// Function: num_tasks
inline size_t Taskflow::num_tasks() const {
  return _graph.size();
}

// Function: empty
inline bool Taskflow::empty() const {
  return _graph.empty();
}

// Function: name
inline void Taskflow::name(const std::string &name) {
  _name = name;
}

// Function: name
inline const std::string& Taskflow::name() const {
  return _name;
}

// Function: for_each_task
template <typename V>
void Taskflow::for_each_task(V&& visitor) const {
  for(size_t i=0; i<_graph._nodes.size(); ++i) {
    visitor(Task(_graph._nodes[i]));
  }
}

// Procedure: dump
inline std::string Taskflow::dump() const {
  std::ostringstream oss;
  dump(oss);
  return oss.str();
}

// Function: dump
inline void Taskflow::dump(std::ostream& os) const {
  os << "digraph Taskflow {\n";
  _dump(os, this);
  os << "}\n";
}

// Procedure: _dump
inline void Taskflow::_dump(std::ostream& os, const Taskflow* top) const {
  
  Dumper dumper;
  
  dumper.stack.push(top);
  dumper.visited.insert(top);

  while(!dumper.stack.empty()) {
    
    auto f = dumper.stack.top();
    dumper.stack.pop();
    
    os << "subgraph cluster_p" << f << " {\nlabel=\"Taskflow: ";
    if(f->_name.empty()) os << 'p' << f;
    else os << f->_name;
    os << "\";\n";
    _dump(os, f->_graph, dumper);
    os << "}\n";
  }
}

// Procedure: _dump
inline void Taskflow::_dump(
  std::ostream& os, const Node* node, Dumper& dumper
) const {

  os << 'p' << node << "[label=\"";
  if(node->_name.empty()) os << 'p' << node;
  else os << node->_name;
  os << "\" ";

  // shape for node
  switch(node->_handle.index()) {

    case Node::CONDITION_TASK:
      os << "shape=diamond color=black fillcolor=aquamarine style=filled";
    break;

    case Node::CUDAFLOW_TASK:
      os << " style=\"filled\""
         << " color=\"black\" fillcolor=\"purple\""
         << " fontcolor=\"white\""
         << " shape=\"folder\"";
    break;

    default:
    break;
  }

  os << "];\n";
  
  for(size_t s=0; s<node->_successors.size(); ++s) {
    if(node->_handle.index() == Node::CONDITION_TASK) {
      // case edge is dashed
      os << 'p' << node << " -> p" << node->_successors[s] 
         << " [style=dashed label=\"" << s << "\"];\n";
    }
    else {
      os << 'p' << node << " -> p" << node->_successors[s] << ";\n";
    }
  }
  
  // subflow join node
  if(node->_parent && node->_successors.size() == 0) {
    os << 'p' << node << " -> p" << node->_parent << ";\n";
  }

  switch(node->_handle.index()) {

    case Node::DYNAMIC_TASK: {
      auto& sbg = std::get<Node::DynamicTask>(node->_handle).subgraph;
      if(!sbg.empty()) {
        os << "subgraph cluster_p" << node << " {\nlabel=\"Subflow: ";
        if(node->_name.empty()) os << 'p' << node;
        else os << node->_name;

        os << "\";\n" << "color=blue\n";
        _dump(os, sbg, dumper);
        os << "}\n";
      }
    }
    break;
    
    case Node::CUDAFLOW_TASK: {
      std::get<Node::cudaFlowTask>(node->_handle).graph->dump(
        os, node, node->_name
      );
    }
    break;

    default:
    break;
  }
}

// Procedure: _dump
inline void Taskflow::_dump(
  std::ostream& os, const Graph& graph, Dumper& dumper
) const {
    
  for(const auto& n : graph._nodes) {

    // regular task
    if(n->_handle.index() != Node::MODULE_TASK) {
      _dump(os, n, dumper);
    }
    // module task
    else {

      auto module = std::get<Node::ModuleTask>(n->_handle).module;

      os << 'p' << n << "[shape=box3d, color=blue, label=\"";
      if(n->_name.empty()) os << n;
      else os << n->_name;
      os << " [Taskflow: ";
      if(module->_name.empty()) os << 'p' << module;
      else os << module->_name;
      os << "]\"];\n";

      if(dumper.visited.find(module) == dumper.visited.end()) {
        dumper.visited.insert(module);
        dumper.stack.push(module);
      }

      for(const auto s : n->_successors) {
        os << 'p' << n << "->" << 'p' << s << ";\n";
      }
    }
  }
}


template <typename T>
class Future : public std::future<T> {
  friend class Node;
  friend class Topology;
  public:
    //future object to store futre object returned by executor.run_until
    std::future<T> future_obj;

    //function for running wait_until method of future object
    template <class Clock, class Duration>
    auto wait_until (const std::chrono::time_point<Clock,Duration>& abs_time) const {
      return future_obj.wait_until(abs_time);
    }

    //function for running wait_for method of future object
    template <class Rep, class Period>
    auto wait_for (const std::chrono::duration<Rep,Period>& rel_time) const{
      return future_obj.wait_for( rel_time);
    }
    
    //function for running wait method of future object
    void wait() const{
      future_obj.wait();
      return;
    }

    //function for running valid method of future object
    bool valid() const noexcept{
      return future_obj.valid();
    }

    //function for running get method of future object
    //template<typename T>
    //template <class _Res>
    T get() {
      //future_obj.wait();
      return future_obj.get();
    }

    //function for running share method of future object
    auto share(){
      return future_obj.share();
    }

    //operator equivalent to "=" operator of future object
    auto operator=(std::future<T>&& rhs) noexcept{
      future_obj=rhs;
    }

    //method for setting is_cancel variable of the topology to true
    void cancel(){
      _topology->is_cancel=true;
      return;
    }

    //method for setting the topology for this class
    void set_tpg(Topology* tpg){
      _topology=tpg;
      return;
    }

    //method for setting async task node for this class
    void set_async_node(Node* node){
      _node=node;
      return;
    }

    //method for setting async_cancelled of an async node to true.
    void cancel_async(){
      _node->async_cancelled=true;
      return;
    }

  private:
    Topology* _topology {nullptr};
    Node* _node {nullptr};

};

}  // end of namespace tf. ---------------------------------------------------

