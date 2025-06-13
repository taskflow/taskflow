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
  2. subflow task        : the callable constructible from
                           @c std::function<void(tf::Subflow&)>
  3. condition task      : the callable constructible from
                           @c std::function<int()>
  4. multi-condition task: the callable constructible from
                           @c %std::function<tf::SmallVector<int>()>
  5. module task         : the task constructed from tf::Taskflow::composed_of
                           @c std::function<void(tf::Runtime&)>

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
  friend class Subflow;

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

    @attention You should avoid moving a taskflow that is currently running on an executor.
    Doing so results in undefined behavior.
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

    @attention You should avoid moving a taskflow that is currently running on an executor.
    Doing so results in undefined behavior.
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
    @brief queries the number of tasks in this taskflow
    
    The number of tasks in this taskflow is defined at the first level of hierarchy.
    Tasks that are created dynamically, such as those via tf::Subflow, are not counted.

    @code{.cpp}
    tf::Taskflow taskflow;
    auto my_task = taskflow.emplace([](){});
    assert(taskflow.num_tasks() == 1);
    
    // reassign my_task to a subflow of four tasks
    my_task.work([](tf::Subflow& sf){
      sf.emplace(
        [](){ std::cout << "Task A\n"; },
        [](){ std::cout << "Task B\n"; },
        [](){ std::cout << "Task C\n"; },
        [](){ std::cout << "Task D\n"; }
      );
    });
    
    // subflow tasks will not be counted
    assert(taskflow.num_tasks() == 1);
    @endcode
    */
    size_t num_tasks() const;

    /**
    @brief queries if this taskflow is empty (has no tasks)

    An empty taskflow has no tasks, i.e., the return of tf::Taskflow::num_tasks is `0`.
    
    @code{.cpp}
    tf::Taskflow taskflow;
    assert(taskflow.empty() == true);
    taskflow.emplace([](){});
    assert(taskflow.empty() == false);
    @endcode
    */
    bool empty() const;

    /**
    @brief assigns a new name to this taskflow

    @code{.cpp}
    taskflow.name("foo");
    assert(taskflow.name() == "foo");
    @endcode
    */
    void name(const std::string&);

    /**
    @brief queries the name of this taskflow

    @code{.cpp}
    tf::Taskflow taskflow("foo");
    assert(taskflow.name() == "foo");
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
    @brief applies a visitor to each task in this taskflow

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
    @brief removes dependencies that go from task @c from to task @c to

    @param from from task (dependent)
    @param to to task (successor)

    Removing the depencency from task `from` to task `to` is equivalent to 
    removing `to` from the succcessor list of `from` and 
    removing `from` from the predecessor list of `to`.

    @code{.cpp}
    tf::Taskflow taskflow;
    auto a = taskflow.placeholder().name("a");
    auto b = taskflow.placeholder().name("b");
    auto c = taskflow.placeholder().name("c");
    auto d = taskflow.placeholder().name("d");

    a.precede(b, c, d);
    assert(a.num_successors() == 3);
    assert(b.num_predecessors() == 1);
    assert(c.num_predecessors() == 1);
    assert(d.num_predecessors() == 1);
  
    taskflow.remove_dependency(a, b);
    assert(a.num_successors() == 2);
    assert(b.num_predecessors() == 0);
    @endcode

    @attention For performance reason, %Taskflow does not store the graph using linked lists but 
    vectors with contiguous space. 
    Therefore, removing tasks or dependencies incurs linear time complexity proportional
    to the size of the graph and the dependency count of a task.
    */
    void remove_dependency(Task from, Task to);

    /**
    @brief returns a reference to the underlying graph object

    A graph object is of type tf::Graph and stores a task dependency graph that can be executed
    by an tf::Executor.

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

// Function: graph
inline Graph& Taskflow::graph() {
  return _graph;
}

// Function: for_each_task
template <typename V>
void Taskflow::for_each_task(V&& visitor) const {
  for(auto itr = _graph.begin(); itr != _graph.end(); ++itr) {
    visitor(Task(itr->get()));
  }
}

// Procedure: remove_dependency
inline void Taskflow::remove_dependency(Task from, Task to) {
  // remove "to" from the succcessor list of "from"
  from._node->_remove_successors(to._node);

  // remove "from" from the predecessor list of "to"
  to._node->_remove_predecessors(from._node);
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

  // label of the node
  os << 'p' << node << "[label=\"";
  if(node->_name.empty()) os << 'p' << node;
  else os << node->_name;
  os << "\" ";

  // shape of the node
  switch(node->_handle.index()) {

    case Node::CONDITION:
    case Node::MULTI_CONDITION:
      os << "shape=diamond color=black fillcolor=aquamarine style=filled";
    break;

    default:
    break;
  }

  os << "];\n";

  for(size_t s=0; s<node->_num_successors; ++s) {
    if(node->_is_conditioner()) {
      // case edge is dashed
      os << 'p' << node << " -> p" << node->_edges[s]
         << " [style=dashed label=\"" << s << "\"];\n";
    } else {
      os << 'p' << node << " -> p" << node->_edges[s] << ";\n";
    }
  }

  // subflow join node
  if(node->_parent && node->_parent->_handle.index() == Node::SUBFLOW &&
     node->_num_successors == 0
    ) {
    os << 'p' << node << " -> p" << node->_parent << " [style=dashed color=blue];\n";
  }

  // node info
  switch(node->_handle.index()) {

    case Node::SUBFLOW: {
      auto& sbg = std::get_if<Node::Subflow>(&node->_handle)->subgraph;
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

    default:
    break;
  }
}

// Procedure: _dump
inline void Taskflow::_dump(
  std::ostream& os, const Graph* graph, Dumper& dumper
) const {

  for(auto itr = graph->begin(); itr != graph->end(); ++itr) {

    Node* n = itr->get();

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

      //for(const auto s : n->_successors) {
      for(size_t i=0; i<n->_num_successors; ++i) {
        os << 'p' << n << "->" << 'p' << n->_edges[i] << ";\n";
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
execution result of a submitted taskflow (tf::Executor::run series).
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
  friend class Runtime;

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

    When you request a cancellation, the executor will stop scheduling any tasks onwards. 
    Tasks that are already running will continue to finish as their executions are non-preemptive.
    You can call tf::Future::wait to wait for the cancellation to complete.

    @code{.cpp}
    // create a taskflow of four tasks and submit it to an executor
    taskflow.emplace(
      [](){ std::cout << "Task A\n"; },
      [](){ std::cout << "Task B\n"; },
      [](){ std::cout << "Task C\n"; },
      [](){ std::cout << "Task D\n"; }
    );
    auto future = executor.run(taskflow);

    // cancel the execution of the taskflow and wait until it finishes all running tasks
    future.cancel();
    future.wait();
    @endcode

    In the above example, we submit a taskflow of four tasks to the executor and then
    issue a cancellation to stop its execution.
    Since the cancellation is non-deterministic with the executor runtime, 
    we may still see some tasks complete their executions or none.

    */
    bool cancel();

  private:
    
    std::weak_ptr<Topology> _topology;

    Future(std::future<T>&&, std::weak_ptr<Topology> = std::weak_ptr<Topology>());
};

template <typename T>
Future<T>::Future(std::future<T>&& f, std::weak_ptr<Topology> p) :
  std::future<T> {std::move(f)},
  _topology      {std::move(p)} {
}

// Function: cancel
template <typename T>
bool Future<T>::cancel() {
  if(auto ptr = _topology.lock(); ptr) {
    ptr->_estate.fetch_or(ESTATE::CANCELLED, std::memory_order_relaxed);
    return true;
  }
  return false;
}


}  // end of namespace tf. ---------------------------------------------------
