#pragma once

#include "flow_builder.hpp"

/**
@file taskflow/core/taskflow.hpp
@brief taskflow include file
*/

namespace tf {

// ----------------------------------------------------------------------------

/**
@class Taskflow

@brief class to create a taskflow object

A %taskflow manages a task dependency graph where each task represents a
callable object (e.g., @std_lambda, @std_function) and an edge represents a
dependency between two tasks. A task is one of the following types:

  1. static task         : the callable constructible from
                           @c std::function<void()>
  2. dynamic task        : the callable constructible from
                           @c std::function<void(tf::Subflow&)>
  3. condition task      : the callable constructible from
                           @c std::function<int()>
  4. multi-condition task: the callable constructible from
                           @c %std::function<tf::SmallVector<int>()>
  5. module task         : the task constructed from tf::Taskflow::composed_of
  6. runtime task        : the callable constructible from
                           @c std::function<void(tf::Runtime&)>
  7. %cudaFlow task      : the callable constructible from
                           @c std::function<void(tf::cudaFlow&)> or
                           @c std::function<void(tf::cudaFlowCapturer&)>
  8. %syclFlow task      : the callable constructible from
                           @c std::function<void(tf::syclFlow&)>

Each task is a basic computation unit and is run by one worker thread
from an executor.
The following example creates a simple taskflow graph of four static tasks,
@c A, @c B, @c C, and @c D, where
@c A runs before @c B and @c C and
@c D runs after  @c B and @c C.

@code{.cpp}
tf::Executor executor;
tf::Taskflow taskflow("simple");

tf::Task A = taskflow.emplace([](){ std::cout << "TaskA\n"; });
tf::Task B = taskflow.emplace([](){ std::cout << "TaskB\n"; });
tf::Task C = taskflow.emplace([](){ std::cout << "TaskC\n"; });
tf::Task D = taskflow.emplace([](){ std::cout << "TaskD\n"; });

A.precede(B, C);  // A runs before B and C
D.succeed(B, C);  // D runs after  B and C

executor.run(taskflow).wait();
@endcode

The taskflow object itself is NOT thread-safe. You should not
modifying the graph while it is running,
such as adding new tasks, adding new dependencies, and moving
the taskflow to another.
To minimize the overhead of task creation,
our runtime leverages a global object pool to recycle
tasks in a thread-safe manner.

Please refer to @ref Cookbook to learn more about each task type
and how to submit a taskflow to an executor.
*/
class Taskflow : public FlowBuilder {

  friend class Topology;
  friend class Executor;
  friend class FlowBuilder;

  struct Dumper {
    size_t id;
    std::stack<std::pair<const Node*, const Graph*>> stack;
    std::unordered_map<const Graph*, size_t> visited;
  };

  public:

    /**
    @brief constructs a taskflow with the given name

    @code{.cpp}
    tf::Taskflow taskflow("My Taskflow");
    std::cout << taskflow.name();         // "My Taskflow"
    @endcode
    */
    Taskflow(const std::string& name);

    /**
    @brief constructs a taskflow
    */
    Taskflow();

    /**
    @brief constructs a taskflow from a moved taskflow

    Constructing a taskflow @c taskflow1 from a moved taskflow @c taskflow2 will
    migrate the graph of @c taskflow2 to @c taskflow1.
    After the move, @c taskflow2 will become empty.

    @code{.cpp}
    tf::Taskflow taskflow1(std::move(taskflow2));
    assert(taskflow2.empty());
    @endcode

    Notice that @c taskflow2 should not be running in an executor
    during the move operation, or the behavior is undefined.
    */
    Taskflow(Taskflow&& rhs);

    /**
    @brief move assignment operator

    Moving a taskflow @c taskflow2 to another taskflow @c taskflow1 will destroy
    the existing graph of @c taskflow1 and assign it the graph of @c taskflow2.
    After the move, @c taskflow2 will become empty.

    @code{.cpp}
    taskflow1 = std::move(taskflow2);
    assert(taskflow2.empty());
    @endcode

    Notice that both @c taskflow1 and @c taskflow2 should not be running
    in an executor during the move operation, or the behavior is undefined.
    */
    Taskflow& operator = (Taskflow&& rhs);

    /**
    @brief default destructor

    When the destructor is called, all tasks and their associated data
    (e.g., captured data) will be destroyed.
    It is your responsibility to ensure all submitted execution of this
    taskflow have completed before destroying it.
    For instance, the following code results in undefined behavior
    since the executor may still be running the taskflow while
    it is destroyed after the block.

    @code{.cpp}
    {
      tf::Taskflow taskflow;
      executor.run(taskflow);
    }
    @endcode

    To fix the problem, we must wait for the execution to complete
    before destroying the taskflow.

    @code{.cpp}
    {
      tf::Taskflow taskflow;
      executor.run(taskflow).wait();
    }
    @endcode
    */
    ~Taskflow() = default;

    /**
    @brief dumps the taskflow to a DOT format through a std::ostream target

    @code{.cpp}
    taskflow.dump(std::cout);  // dump the graph to the standard output

    std::ofstream ofs("output.dot");
    taskflow.dump(ofs);        // dump the graph to the file output.dot
    @endcode

    For dynamically spawned tasks, such as module tasks, subflow tasks,
    and GPU tasks, you need to run the taskflow first before you can
    dump the entire graph.

    @code{.cpp}
    tf::Task parent = taskflow.emplace([](tf::Subflow sf){
      sf.emplace([](){ std::cout << "child\n"; });
    });
    taskflow.dump(std::cout);      // this dumps only the parent tasks
    executor.run(taskflow).wait();
    taskflow.dump(std::cout);      // this dumps both parent and child tasks
    @endcode
    */
    void dump(std::ostream& ostream) const;

    /**
    @brief dumps the taskflow to a std::string of DOT format

    This method is similar to tf::Taskflow::dump(std::ostream& ostream),
    but returning a string of the graph in DOT format.
    */
    std::string dump() const;

    /**
    @brief queries the number of tasks
    */
    size_t num_tasks() const;

    /**
    @brief queries the emptiness of the taskflow

    An empty taskflow has no tasks. That is the return of
    tf::Taskflow::num_tasks is zero.
    */
    bool empty() const;

    /**
    @brief assigns a name to the taskflow

    @code{.cpp}
    taskflow.name("assign another name");
    @endcode
    */
    void name(const std::string&);

    /**
    @brief queries the name of the taskflow

    @code{.cpp}
    std::cout << "my name is: " << taskflow.name();
    @endcode
    */
    const std::string& name() const;

    /**
    @brief clears the associated task dependency graph

    When you clear a taskflow, all tasks and their associated data
    (e.g., captured data in task callables) will be destroyed.
    The behavior of clearing a running taskflow is undefined.
    */
    void clear();

    /**
    @brief applies a visitor to each task in the taskflow

    A visitor is a callable that takes an argument of type tf::Task
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

    /**
    @brief returns a reference to the underlying graph object

    A graph object (of type tf::Graph) is the ultimate storage for the
    task dependency graph and should only be used as an opaque
    data structure to interact with the executor (e.g., composition).
    */
    Graph& graph();

  private:

    mutable std::mutex _mutex;

    std::string _name;

    Graph _graph;

    std::queue<std::shared_ptr<Topology>> _topologies;

    std::optional<std::list<Taskflow>::iterator> _satellite;

    void _dump(std::ostream&, const Graph*) const;
    void _dump(std::ostream&, const Node*, Dumper&) const;
    void _dump(std::ostream&, const Graph*, Dumper&) const;
};

// Constructor
inline Taskflow::Taskflow(const std::string& name) :
  FlowBuilder {_graph},
  _name       {name} {
}

// Constructor
inline Taskflow::Taskflow() : FlowBuilder{_graph} {
}

// Move constructor
inline Taskflow::Taskflow(Taskflow&& rhs) : FlowBuilder{_graph} {

  std::scoped_lock<std::mutex> lock(rhs._mutex);

  _name = std::move(rhs._name);
  _graph = std::move(rhs._graph);
  _topologies = std::move(rhs._topologies);
  _satellite = rhs._satellite;

  rhs._satellite.reset();
}

// Move assignment
inline Taskflow& Taskflow::operator = (Taskflow&& rhs) {
  if(this != &rhs) {
    std::scoped_lock<std::mutex, std::mutex> lock(_mutex, rhs._mutex);
    _name = std::move(rhs._name);
    _graph = std::move(rhs._graph);
    _topologies = std::move(rhs._topologies);
    _satellite = rhs._satellite;
    rhs._satellite.reset();
  }
  return *this;
}

// Procedure:
inline void Taskflow::clear() {
  _graph._clear();
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

// Function: graph
inline Graph& Taskflow::graph() {
  return _graph;
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
  _dump(os, &_graph);
  os << "}\n";
}

// Procedure: _dump
inline void Taskflow::_dump(std::ostream& os, const Graph* top) const {

  Dumper dumper;

  dumper.id = 0;
  dumper.stack.push({nullptr, top});
  dumper.visited[top] = dumper.id++;

  while(!dumper.stack.empty()) {

    auto [p, f] = dumper.stack.top();
    dumper.stack.pop();

    os << "subgraph cluster_p" << f << " {\nlabel=\"";

    // n-level module
    if(p) {
      os << 'm' << dumper.visited[f];
    }
    // top-level taskflow graph
    else {
      os << "Taskflow: ";
      if(_name.empty()) os << 'p' << this;
      else os << _name;
    }

    os << "\";\n";

    _dump(os, f, dumper);
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

    case Node::CONDITION:
    case Node::MULTI_CONDITION:
      os << "shape=diamond color=black fillcolor=aquamarine style=filled";
    break;

    case Node::RUNTIME:
      os << "shape=component";
    break;

    case Node::CUDAFLOW:
      os << " style=\"filled\""
         << " color=\"black\" fillcolor=\"purple\""
         << " fontcolor=\"white\""
         << " shape=\"folder\"";
    break;

    case Node::SYCLFLOW:
      os << " style=\"filled\""
         << " color=\"black\" fillcolor=\"red\""
         << " fontcolor=\"white\""
         << " shape=\"folder\"";
    break;

    default:
    break;
  }

  os << "];\n";

  for(size_t s=0; s<node->_successors.size(); ++s) {
    if(node->_is_conditioner()) {
      // case edge is dashed
      os << 'p' << node << " -> p" << node->_successors[s]
         << " [style=dashed label=\"" << s << "\"];\n";
    } else {
      os << 'p' << node << " -> p" << node->_successors[s] << ";\n";
    }
  }

  // subflow join node
  if(node->_parent && node->_parent->_handle.index() == Node::DYNAMIC &&
     node->_successors.size() == 0
    ) {
    os << 'p' << node << " -> p" << node->_parent << ";\n";
  }

  // node info
  switch(node->_handle.index()) {

    case Node::DYNAMIC: {
      auto& sbg = std::get_if<Node::Dynamic>(&node->_handle)->subgraph;
      if(!sbg.empty()) {
        os << "subgraph cluster_p" << node << " {\nlabel=\"Subflow: ";
        if(node->_name.empty()) os << 'p' << node;
        else os << node->_name;

        os << "\";\n" << "color=blue\n";
        _dump(os, &sbg, dumper);
        os << "}\n";
      }
    }
    break;

    case Node::CUDAFLOW: {
      std::get_if<Node::cudaFlow>(&node->_handle)->graph->dump(
        os, node, node->_name
      );
    }
    break;

    case Node::SYCLFLOW: {
      std::get_if<Node::syclFlow>(&node->_handle)->graph->dump(
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
  std::ostream& os, const Graph* graph, Dumper& dumper
) const {

  for(const auto& n : graph->_nodes) {

    // regular task
    if(n->_handle.index() != Node::MODULE) {
      _dump(os, n, dumper);
    }
    // module task
    else {
      //auto module = &(std::get_if<Node::Module>(&n->_handle)->module);
      auto module = &(std::get_if<Node::Module>(&n->_handle)->graph);

      os << 'p' << n << "[shape=box3d, color=blue, label=\"";
      if(n->_name.empty()) os << 'p' << n;
      else os << n->_name;

      if(dumper.visited.find(module) == dumper.visited.end()) {
        dumper.visited[module] = dumper.id++;
        dumper.stack.push({n, module});
      }

      os << " [m" << dumper.visited[module] << "]\"];\n";

      for(const auto s : n->_successors) {
        os << 'p' << n << "->" << 'p' << s << ";\n";
      }
    }
  }
}

// ----------------------------------------------------------------------------
// class definition: Future
// ----------------------------------------------------------------------------

/**
@class Future

@brief class to access the result of an execution

tf::Future is a derived class from std::future that will eventually hold the
execution result of a submitted taskflow (tf::Executor::run)
or an asynchronous task (tf::Executor::async, tf::Executor::silent_async).
In addition to the base methods inherited from std::future,
you can call tf::Future::cancel to cancel the execution of the running taskflow
associated with this future object.
The following example cancels a submission of a taskflow that contains
1000 tasks each running one second.

@code{.cpp}
tf::Executor executor;
tf::Taskflow taskflow;

for(int i=0; i<1000; i++) {
  taskflow.emplace([](){
    std::this_thread::sleep_for(std::chrono::seconds(1));
  });
}

// submit the taskflow
tf::Future fu = executor.run(taskflow);

// request to cancel the submitted execution above
fu.cancel();

// wait until the cancellation finishes
fu.get();
@endcode
*/
template <typename T>
class Future : public std::future<T>  {

  friend class Executor;
  friend class Subflow;

  using handle_t = std::variant<
    std::monostate, std::weak_ptr<Topology>, std::weak_ptr<AsyncTopology>
  >;

  // variant index
  constexpr static auto ASYNC = get_index_v<std::weak_ptr<AsyncTopology>, handle_t>;
  constexpr static auto TASKFLOW = get_index_v<std::weak_ptr<Topology>, handle_t>;

  public:

    /**
    @brief default constructor
    */
    Future() = default;

    /**
    @brief disabled copy constructor
    */
    Future(const Future&) = delete;

    /**
    @brief default move constructor
    */
    Future(Future&&) = default;

    /**
    @brief disabled copy assignment
    */
    Future& operator = (const Future&) = delete;

    /**
    @brief default move assignment
    */
    Future& operator = (Future&&) = default;

    /**
    @brief cancels the execution of the running taskflow associated with
           this future object

    @return @c true if the execution can be cancelled or
            @c false if the execution has already completed

    When you request a cancellation, the executor will stop scheduling
    any tasks onwards. Tasks that are already running will continue to finish
    (non-preemptive).
    You can call tf::Future::wait to wait for the cancellation to complete.
    */
    bool cancel();

  private:

    handle_t _handle;

    template <typename P>
    Future(std::future<T>&&, P&&);
};

template <typename T>
template <typename P>
Future<T>::Future(std::future<T>&& fu, P&& p) :
  std::future<T> {std::move(fu)},
  _handle        {std::forward<P>(p)} {
}

// Function: cancel
template <typename T>
bool Future<T>::cancel() {
  return std::visit([](auto&& arg){
    using P = std::decay_t<decltype(arg)>;
    if constexpr(std::is_same_v<P, std::monostate>) {
      return false;
    }
    else {
      auto ptr = arg.lock();
      if(ptr) {
        ptr->_is_cancelled.store(true, std::memory_order_relaxed);
        return true;
      }
      return false;
    }
  }, _handle);
}


}  // end of namespace tf. ---------------------------------------------------
