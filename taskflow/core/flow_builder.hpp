#pragma once

#include "task.hpp"

/**
@file flow_builder.hpp
@brief flow builder include file
*/

namespace tf {

/**
@class FlowBuilder

@brief class to build a task dependency graph

The class provides essential methods to construct a task dependency graph
from which tf::Taskflow and tf::Subflow are derived.

*/
class FlowBuilder {

  friend class Executor;

  public:

    /**
    @brief constructs a flow builder with a graph
    */
    FlowBuilder(Graph& graph);

    /**
    @brief creates a static task

    @tparam C callable type constructible from std::function<void()>

    @param callable callable to construct a static task

    @return a tf::Task handle

    The following example creates a static task.

    @code{.cpp}
    tf::Task static_task = taskflow.emplace([](){});
    @endcode

    Please refer to @ref StaticTasking for details.
    */
    template <typename C,
      std::enable_if_t<is_static_task_v<C>, void>* = nullptr
    >
    Task emplace(C&& callable);

    /**
    @brief creates a dynamic task

    @tparam C callable type constructible from std::function<void(tf::Subflow&)>

    @param callable callable to construct a dynamic task

    @return a tf::Task handle

    The following example creates a dynamic task (tf::Subflow)
    that spawns two static tasks.

    @code{.cpp}
    tf::Task dynamic_task = taskflow.emplace([](tf::Subflow& sf){
      tf::Task static_task1 = sf.emplace([](){});
      tf::Task static_task2 = sf.emplace([](){});
    });
    @endcode

    Please refer to @ref DynamicTasking for details.
    */
    template <typename C,
      std::enable_if_t<is_dynamic_task_v<C>, void>* = nullptr
    >
    Task emplace(C&& callable);

    /**
    @brief creates a condition task

    @tparam C callable type constructible from std::function<int()>

    @param callable callable to construct a condition task

    @return a tf::Task handle

    The following example creates an if-else block using one condition task
    and three static tasks.

    @code{.cpp}
    tf::Taskflow taskflow;

    auto [init, cond, yes, no] = taskflow.emplace(
      [] () { },
      [] () { return 0; },
      [] () { std::cout << "yes\n"; },
      [] () { std::cout << "no\n"; }
    );

    // executes yes if cond returns 0, or no if cond returns 1
    cond.precede(yes, no);
    cond.succeed(init);
    @endcode

    Please refer to @ref ConditionalTasking for details.
    */
    template <typename C,
      std::enable_if_t<is_condition_task_v<C>, void>* = nullptr
    >
    Task emplace(C&& callable);

    /**
    @brief creates a multi-condition task

    @tparam C callable type constructible from
            std::function<tf::SmallVector<int>()>

    @param callable callable to construct a multi-condition task

    @return a tf::Task handle

    The following example creates a multi-condition task that selectively
    jumps to two successor tasks.

    @code{.cpp}
    tf::Taskflow taskflow;

    auto [init, cond, branch1, branch2, branch3] = taskflow.emplace(
      [] () { },
      [] () { return tf::SmallVector{0, 2}; },
      [] () { std::cout << "branch1\n"; },
      [] () { std::cout << "branch2\n"; },
      [] () { std::cout << "branch3\n"; }
    );

    // executes branch1 and branch3 when cond returns 0 and 2
    cond.precede(branch1, branch2, branch3);
    cond.succeed(init);
    @endcode

    Please refer to @ref ConditionalTasking for details.
    */
    template <typename C,
      std::enable_if_t<is_multi_condition_task_v<C>, void>* = nullptr
    >
    Task emplace(C&& callable);

    /**
    @brief creates multiple tasks from a list of callable objects

    @tparam C callable types

    @param callables one or multiple callable objects constructible from each task category

    @return a tf::Task handle

    The method returns a tuple of tasks each corresponding to the given
    callable target. You can use structured binding to get the return tasks
    one by one.
    The following example creates four static tasks and assign them to
    @c A, @c B, @c C, and @c D using structured binding.

    @code{.cpp}
    auto [A, B, C, D] = taskflow.emplace(
      [] () { std::cout << "A"; },
      [] () { std::cout << "B"; },
      [] () { std::cout << "C"; },
      [] () { std::cout << "D"; }
    );
    @endcode
    */
    template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
    auto emplace(C&&... callables);

    /**
    @brief removes a task from a taskflow

    @param task task to remove

    Removes a task and its input and output dependencies from the graph
    associated with the flow builder.
    If the task does not belong to the graph, nothing will happen.

    @code{.cpp}
    tf::Task A = taskflow.emplace([](){ std::cout << "A"; });
    tf::Task B = taskflow.emplace([](){ std::cout << "B"; });
    tf::Task C = taskflow.emplace([](){ std::cout << "C"; });
    tf::Task D = taskflow.emplace([](){ std::cout << "D"; });
    A.precede(B, C, D);

    // erase A from the taskflow and its dependencies to B, C, and D
    taskflow.erase(A);
    @endcode
    */
    void erase(Task task);

    /**
    @brief creates a module task for the target object

    @tparam T target object type
    @param object a custom object that defines the method @c T::graph()

    @return a tf::Task handle

    The example below demonstrates a taskflow composition using
    the @c composed_of method.

    @code{.cpp}
    tf::Taskflow t1, t2;
    t1.emplace([](){ std::cout << "t1"; });

    // t2 is partially composed of t1
    tf::Task comp = t2.composed_of(t1);
    tf::Task init = t2.emplace([](){ std::cout << "t2"; });
    init.precede(comp);
    @endcode

    The taskflow object @c t2 is composed of another taskflow object @c t1,
    preceded by another static task @c init.
    When taskflow @c t2 is submitted to an executor,
    @c init will run first and then @c comp which spwans its definition
    in taskflow @c t1.

    The target @c object being composed must define the method
    <tt>T::graph()</tt> that returns a reference to a graph object of
    type tf::Graph such that it can interact with the executor.
    For example:

    @code{.cpp}
    // custom struct
    struct MyObj {
      tf::Graph graph;
      MyObj() {
        tf::FlowBuilder builder(graph);
        tf::Task task = builder.emplace([](){
          std::cout << "a task\n";  // static task
        });
      }
      Graph& graph() { return graph; }
    };

    MyObj obj;
    tf::Task comp = taskflow.composed_of(obj);
    @endcode

    Please refer to @ref ComposableTasking for details.
    */
    template <typename T>
    Task composed_of(T& object);

    /**
    @brief creates a placeholder task

    @return a tf::Task handle

    A placeholder task maps to a node in the taskflow graph, but
    it does not have any callable work assigned yet.
    A placeholder task is different from an empty task handle that
    does not point to any node in a graph.

    @code{.cpp}
    // create a placeholder task with no callable target assigned
    tf::Task placeholder = taskflow.placeholder();
    assert(placeholder.empty() == false && placeholder.has_work() == false);

    // create an empty task handle
    tf::Task task;
    assert(task.empty() == true);

    // assign the task handle to the placeholder task
    task = placeholder;
    assert(task.empty() == false && task.has_work() == false);
    @endcode
    */
    Task placeholder();

    /**
    @brief creates a %cudaFlow task on the caller's GPU device context

    @tparam C callable type constructible from @c std::function<void(tf::cudaFlow&)>

    @return a tf::Task handle

    This method is equivalent to calling tf::FlowBuilder::emplace_on(callable, d)
    where @c d is the caller's device context.
    The following example creates a %cudaFlow of two kernel tasks, @c task1 and
    @c task2, where @c task1 runs before @c task2.

    @code{.cpp}
    taskflow.emplace([&](tf::cudaFlow& cf){
      // create two kernel tasks
      tf::cudaTask task1 = cf.kernel(grid1, block1, shm1, kernel1, args1);
      tf::cudaTask task2 = cf.kernel(grid2, block2, shm2, kernel2, args2);

      // kernel1 runs before kernel2
      task1.precede(task2);
    });
    @endcode

    Please refer to @ref GPUTaskingcudaFlow and @ref GPUTaskingcudaFlowCapturer
    for details.
    */
    template <typename C,
      std::enable_if_t<is_cudaflow_task_v<C>, void>* = nullptr
    >
    Task emplace(C&& callable);

    /**
    @brief creates a %cudaFlow task on the given device

    @tparam C callable type constructible from std::function<void(tf::cudaFlow&)>
    @tparam D device type, either @c int or @c std::ref<int> (stateful)

    @return a tf::Task handle

    The following example creates a %cudaFlow of two kernel tasks, @c task1 and
    @c task2 on GPU @c 2, where @c task1 runs before @c task2

    @code{.cpp}
    taskflow.emplace_on([&](tf::cudaFlow& cf){
      // create two kernel tasks
      tf::cudaTask task1 = cf.kernel(grid1, block1, shm1, kernel1, args1);
      tf::cudaTask task2 = cf.kernel(grid2, block2, shm2, kernel2, args2);

      // kernel1 runs before kernel2
      task1.precede(task2);
    }, 2);
    @endcode
    */
    template <typename C, typename D,
      std::enable_if_t<is_cudaflow_task_v<C>, void>* = nullptr
    >
    Task emplace_on(C&& callable, D&& device);

    /**
    @brief creates a %syclFlow task on the default queue

    @tparam C callable type constructible from std::function<void(tf::syclFlow&)>

    @param callable a callable that takes a referenced tf::syclFlow object

    @return a tf::Task handle

    The following example creates a %syclFlow on the default queue to submit
    two kernel tasks, @c task1 and @c task2, where @c task1 runs before @c task2.

    @code{.cpp}
    taskflow.emplace([&](tf::syclFlow& cf){
      // create two single-thread kernel tasks
      tf::syclTask task1 = cf.single_task([](){});
      tf::syclTask task2 = cf.single_task([](){});

      // kernel1 runs before kernel2
      task1.precede(task2);
    });
    @endcode
    */
    template <typename C, std::enable_if_t<is_syclflow_task_v<C>, void>* = nullptr>
    Task emplace(C&& callable);

    /**
    @brief creates a %syclFlow task on the given queue

    @tparam C callable type constructible from std::function<void(tf::syclFlow&)>
    @tparam Q queue type

    @param callable a callable that takes a referenced tf::syclFlow object
    @param queue a queue of type sycl::queue

    @return a tf::Task handle

    The following example creates a %syclFlow on the given queue to submit
    two kernel tasks, @c task1 and @c task2, where @c task1 runs before @c task2.

    @code{.cpp}
    taskflow.emplace_on([&](tf::syclFlow& cf){
      // create two single-thread kernel tasks
      tf::syclTask task1 = cf.single_task([](){});
      tf::syclTask task2 = cf.single_task([](){});

      // kernel1 runs before kernel2
      task1.precede(task2);
    }, queue);
    @endcode
    */
    template <typename C, typename Q,
      std::enable_if_t<is_syclflow_task_v<C>, void>* = nullptr
    >
    Task emplace_on(C&& callable, Q&& queue);

    /**
    @brief creates a runtime task

    @tparam C callable type constructible from std::function<void(tf::Runtime&)>

    @param callable callable to construct a runtime task

    @return a tf::Task handle

    The following example creates a runtime task that enables in-task
    control over the running executor.

    @code{.cpp}
    tf::Task runtime_task = taskflow.emplace([](tf::Runtime& rt){
      auto& executor = rt.executor();
      std::cout << executor.num_workers() << '\n';
    });
    @endcode

    Please refer to @ref RuntimeTasking for details.
    */
    template <typename C,
      std::enable_if_t<is_runtime_task_v<C>, void>* = nullptr
    >
    Task emplace(C&& callable);

    /**
    @brief adds adjacent dependency links to a linear list of tasks

    @param tasks a vector of tasks

    This member function creates linear dependencies over a vector of tasks.

    @code{.cpp}
    tf::Task A = taskflow.emplace([](){ std::cout << "A"; });
    tf::Task B = taskflow.emplace([](){ std::cout << "B"; });
    tf::Task C = taskflow.emplace([](){ std::cout << "C"; });
    tf::Task D = taskflow.emplace([](){ std::cout << "D"; });
    std::vector<tf::Task> tasks {A, B, C, D}
    taskflow.linearize(tasks);  // A->B->C->D
    @endcode

    */
    void linearize(std::vector<Task>& tasks);

    /**
    @brief adds adjacent dependency links to a linear list of tasks

    @param tasks an initializer list of tasks

    This member function creates linear dependencies over a list of tasks.

    @code{.cpp}
    tf::Task A = taskflow.emplace([](){ std::cout << "A"; });
    tf::Task B = taskflow.emplace([](){ std::cout << "B"; });
    tf::Task C = taskflow.emplace([](){ std::cout << "C"; });
    tf::Task D = taskflow.emplace([](){ std::cout << "D"; });
    taskflow.linearize({A, B, C, D});  // A->B->C->D
    @endcode
    */
    void linearize(std::initializer_list<Task> tasks);

    // ------------------------------------------------------------------------
    // parallel iterations
    // ------------------------------------------------------------------------

    /**
    @brief constructs a STL-styled parallel-for task

    @tparam B beginning iterator type
    @tparam E ending iterator type
    @tparam C callable type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param callable a callable object to apply to the dereferenced iterator

    @return a tf::Task handle

    The task spawns a subflow that applies the callable object to each object
    obtained by dereferencing every iterator in the range <tt>[first, last)</tt>.
    This method is equivalent to the parallel execution of the following loop:

    @code{.cpp}
    for(auto itr=first; itr!=last; itr++) {
      callable(*itr);
    }
    @endcode

    Arguments templated to enable stateful range using std::reference_wrapper.
    The callable needs to take a single argument of
    the dereferenced iterator type.

    Please refer to @ref ParallelIterations for details.
    */
    template <typename B, typename E, typename C>
    Task for_each(B first, E last, C callable);

    /**
    @brief constructs a parallel-transform task

    @tparam B beginning index type (must be integral)
    @tparam E ending index type (must be integral)
    @tparam S step type (must be integral)
    @tparam C callable type

    @param first index of the beginning (inclusive)
    @param last index of the end (exclusive)
    @param step step size
    @param callable a callable object to apply to each valid index

    @return a tf::Task handle

    The task spawns a subflow that applies the callable object to each index
    in the range <tt>[first, last)</tt> with the step size.
    This method is equivalent to the parallel execution of the following loop:

    @code{.cpp}
    // case 1: step size is positive
    for(auto i=first; i<last; i+=step) {
      callable(i);
    }

    // case 2: step size is negative
    for(auto i=first, i>last; i+=step) {
      callable(i);
    }
    @endcode

    Arguments are templated to enable stateful range using std::reference_wrapper.
    The callable needs to take a single argument of the integral index type.

    Please refer to @ref ParallelIterations for details.
    */
    template <typename B, typename E, typename S, typename C>
    Task for_each_index(B first, E last, S step, C callable);

    // ------------------------------------------------------------------------
    // transform
    // ------------------------------------------------------------------------

    /**
    @brief constructs a parallel-transform task

    @tparam B beginning input iterator type
    @tparam E ending input iterator type
    @tparam O output iterator type
    @tparam C callable type

    @param first1 iterator to the beginning of the first range
    @param last1 iterator to the end of the first range
    @param d_first iterator to the beginning of the output range
    @param c an unary callable to apply to dereferenced input elements

    @return a tf::Task handle

    The task spawns a subflow that applies the callable object to an
    input range and stores the result in another output range.
    This method is equivalent to the parallel execution of the following loop:

    @code{.cpp}
    while (first1 != last1) {
      *d_first++ = c(*first1++);
    }
    @endcode

    Arguments are templated to enable stateful range using std::reference_wrapper.
    The callable needs to take a single argument of the dereferenced
    iterator type.
    */
    template <typename B, typename E, typename O, typename C>
    Task transform(B first1, E last1, O d_first, C c);

    /**
    @brief constructs a parallel-transform task

    @tparam B1 beginning input iterator type for the first input range
    @tparam E1 ending input iterator type for the first input range
    @tparam B2 beginning input iterator type for the first second range
    @tparam O output iterator type
    @tparam C callable type

    @param first1 iterator to the beginning of the first input range
    @param last1 iterator to the end of the first input range
    @param first2 iterator to the beginning of the second input range
    @param d_first iterator to the beginning of the output range
    @param c a binary operator to apply to dereferenced input elements

    @return a tf::Task handle

    The task spawns a subflow that applies the callable object to two
    input ranges and stores the result in another output range.
    This method is equivalent to the parallel execution of the following loop:

    @code{.cpp}
    while (first1 != last1) {
      *d_first++ = c(*first1++, *first2++);
    }
    @endcode

    Arguments are templated to enable stateful range using std::reference_wrapper.
    The callable needs to take two arguments of dereferenced elements
    from the two input ranges.
    */
    template <typename B1, typename E1, typename B2, typename O, typename C>
    Task transform(B1 first1, E1 last1, B2 first2, O d_first, C c);

    // ------------------------------------------------------------------------
    // reduction
    // ------------------------------------------------------------------------

    /**
    @brief constructs a STL-styled parallel-reduce task

    @tparam B beginning iterator type
    @tparam E ending iterator type
    @tparam T result type
    @tparam O binary reducer type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param init initial value of the reduction and the storage for the reduced result
    @param bop binary operator that will be applied

    @return a tf::Task handle

    The task spawns a subflow to perform parallel reduction over @c init
    and the elements in the range <tt>[first, last)</tt>.
    The reduced result is store in @c init.
    This method is equivalent to the parallel execution of the following loop:

    @code{.cpp}
    for(auto itr=first; itr!=last; itr++) {
      init = bop(init, *itr);
    }
    @endcode

    Arguments are templated to enable stateful range using std::reference_wrapper.

    Please refer to @ref ParallelReduction for details.
    */
    template <typename B, typename E, typename T, typename O>
    Task reduce(B first, E last, T& init, O bop);

    // ------------------------------------------------------------------------
    // transfrom and reduction
    // ------------------------------------------------------------------------

    /**
    @brief constructs a STL-styled parallel transform-reduce task

    @tparam B beginning iterator type
    @tparam E ending iterator type
    @tparam T result type
    @tparam BOP binary reducer type
    @tparam UOP unary transformion type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param init initial value of the reduction and the storage for the reduced result
    @param bop binary operator that will be applied in unspecified order to the results of @c uop
    @param uop unary operator that will be applied to transform each element in the range to the result type

    @return a tf::Task handle

    The task spawns a subflow to perform parallel reduction over @c init and
    the transformed elements in the range <tt>[first, last)</tt>.
    The reduced result is store in @c init.
    This method is equivalent to the parallel execution of the following loop:

    @code{.cpp}
    for(auto itr=first; itr!=last; itr++) {
      init = bop(init, uop(*itr));
    }
    @endcode

    Arguments are templated to enable stateful range using std::reference_wrapper.

    Please refer to @ref ParallelReduction for details.
    */
    template <typename B, typename E, typename T, typename BOP, typename UOP>
    Task transform_reduce(B first, E last, T& init, BOP bop, UOP uop);

    // ------------------------------------------------------------------------
    // sort
    // ------------------------------------------------------------------------

    /**
    @brief constructs a dynamic task to perform STL-styled parallel sort

    @tparam B beginning iterator type (random-accessible)
    @tparam E ending iterator type (random-accessible)
    @tparam C comparator type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param cmp comparison function object

    The task spawns a subflow to parallelly sort elements in the range
    <tt>[first, last)</tt>.

    Arguments are templated to enable stateful range using std::reference_wrapper.

    Please refer to @ref ParallelSort for details.
    */
    template <typename B, typename E, typename C>
    Task sort(B first, E last, C cmp);

    /**
    @brief constructs a dynamic task to perform STL-styled parallel sort using
           the @c std::less<T> comparator, where @c T is the element type

    @tparam B beginning iterator type (random-accessible)
    @tparam E ending iterator type (random-accessible)

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)

    The task spawns a subflow to parallelly sort elements in the range
    <tt>[first, last)</tt> using the @c std::less<T> comparator,
    where @c T is the dereferenced iterator type.

    Arguments are templated to enable stateful range using std::reference_wrapper.

    Please refer to @ref ParallelSort for details.
     */
    template <typename B, typename E>
    Task sort(B first, E last);

  protected:

    /**
    @brief associated graph object
    */
    Graph& _graph;

  private:

    template <typename L>
    void _linearize(L&);
};

// Constructor
inline FlowBuilder::FlowBuilder(Graph& graph) :
  _graph {graph} {
}

// Function: emplace
template <typename C, std::enable_if_t<is_static_task_v<C>, void>*>
Task FlowBuilder::emplace(C&& c) {
  return Task(_graph._emplace_back(
    std::in_place_type_t<Node::Static>{}, std::forward<C>(c)
  ));
}

// Function: emplace
template <typename C, std::enable_if_t<is_dynamic_task_v<C>, void>*>
Task FlowBuilder::emplace(C&& c) {
  return Task(_graph._emplace_back(
    std::in_place_type_t<Node::Dynamic>{}, std::forward<C>(c)
  ));
}

// Function: emplace
template <typename C, std::enable_if_t<is_condition_task_v<C>, void>*>
Task FlowBuilder::emplace(C&& c) {
  return Task(_graph._emplace_back(
    std::in_place_type_t<Node::Condition>{}, std::forward<C>(c)
  ));
}

// Function: emplace
template <typename C, std::enable_if_t<is_multi_condition_task_v<C>, void>*>
Task FlowBuilder::emplace(C&& c) {
  return Task(_graph._emplace_back(
    std::in_place_type_t<Node::MultiCondition>{}, std::forward<C>(c)
  ));
}

// Function: emplace
template <typename C, std::enable_if_t<is_runtime_task_v<C>, void>*>
Task FlowBuilder::emplace(C&& c) {
  return Task(_graph._emplace_back(
    std::in_place_type_t<Node::Runtime>{}, std::forward<C>(c)
  ));
}

// Function: emplace
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto FlowBuilder::emplace(C&&... cs) {
  return std::make_tuple(emplace(std::forward<C>(cs))...);
}

// Function: erase
inline void FlowBuilder::erase(Task task) {

  if (!task._node) {
    return;
  }

  task.for_each_dependent([&] (Task dependent) {
    auto& S = dependent._node->_successors;
    if(auto I = std::find(S.begin(), S.end(), task._node); I != S.end()) {
      S.erase(I);
    }
  });

  task.for_each_successor([&] (Task dependent) {
    auto& D = dependent._node->_dependents;
    if(auto I = std::find(D.begin(), D.end(), task._node); I != D.end()) {
      D.erase(I);
    }
  });

  _graph._erase(task._node);
}

// Function: composed_of
template <typename T>
Task FlowBuilder::composed_of(T& object) {
  auto node = _graph._emplace_back(
    std::in_place_type_t<Node::Module>{}, object
  );
  return Task(node);
}

// Function: placeholder
inline Task FlowBuilder::placeholder() {
  auto node = _graph._emplace_back();
  return Task(node);
}

// Procedure: _linearize
template <typename L>
void FlowBuilder::_linearize(L& keys) {

  auto itr = keys.begin();
  auto end = keys.end();

  if(itr == end) {
    return;
  }

  auto nxt = itr;

  for(++nxt; nxt != end; ++nxt, ++itr) {
    itr->_node->_precede(nxt->_node);
  }
}

// Procedure: linearize
inline void FlowBuilder::linearize(std::vector<Task>& keys) {
  _linearize(keys);
}

// Procedure: linearize
inline void FlowBuilder::linearize(std::initializer_list<Task> keys) {
  _linearize(keys);
}

// ----------------------------------------------------------------------------

/**
@class Subflow

@brief class to construct a subflow graph from the execution of a dynamic task

By default, a subflow automatically @em joins its parent node.
You may explicitly join or detach a subflow by calling tf::Subflow::join
or tf::Subflow::detach, respectively.
The following example creates a taskflow graph that spawns a subflow from
the execution of task @c B, and the subflow contains three tasks, @c B1,
@c B2, and @c B3, where @c B3 runs after @c B1 and @c B2.

@code{.cpp}
// create three static tasks
tf::Task A = taskflow.emplace([](){}).name("A");
tf::Task C = taskflow.emplace([](){}).name("C");
tf::Task D = taskflow.emplace([](){}).name("D");

// create a subflow graph (dynamic tasking)
tf::Task B = taskflow.emplace([] (tf::Subflow& subflow) {
  tf::Task B1 = subflow.emplace([](){}).name("B1");
  tf::Task B2 = subflow.emplace([](){}).name("B2");
  tf::Task B3 = subflow.emplace([](){}).name("B3");
  B1.precede(B3);
  B2.precede(B3);
}).name("B");

A.precede(B);  // B runs after A
A.precede(C);  // C runs after A
B.precede(D);  // D runs after B
C.precede(D);  // D runs after C
@endcode

*/
class Subflow : public FlowBuilder {

  friend class Executor;
  friend class FlowBuilder;
  friend class Runtime;

  public:

    /**
    @brief enables the subflow to join its parent task

    Performs an immediate action to join the subflow. Once the subflow is joined,
    it is considered finished and you may not modify the subflow anymore.

    @code{.cpp}
    taskflow.emplace([](tf::Subflow& sf){
      sf.emplace([](){});
      sf.join();  // join the subflow of one task
    });
    @endcode

    Only the worker that spawns this subflow can join it.
    */
    void join();

    /**
    @brief enables the subflow to detach from its parent task

    Performs an immediate action to detach the subflow. Once the subflow is detached,
    it is considered finished and you may not modify the subflow anymore.

    @code{.cpp}
    taskflow.emplace([](tf::Subflow& sf){
      sf.emplace([](){});
      sf.detach();
    });
    @endcode

    Only the worker that spawns this subflow can detach it.
    */
    void detach();

    /**
    @brief resets the subflow to a joinable state

    @param clear_graph specifies whether to clear the associated graph (default @c true)

    Clears the underlying task graph depending on the 
    given variable @c clear_graph (default @c true) and then
    updates the subflow to a joinable state.
    */
    void reset(bool clear_graph = true);

    /**
    @brief queries if the subflow is joinable

    This member function queries if the subflow is joinable.
    When a subflow is joined or detached, it becomes not joinable.

    @code{.cpp}
    taskflow.emplace([](tf::Subflow& sf){
      sf.emplace([](){});
      std::cout << sf.joinable() << '\n';  // true
      sf.join();
      std::cout << sf.joinable() << '\n';  // false
    });
    @endcode
    */
    bool joinable() const noexcept;

    /**
    @brief runs a given function asynchronously

    @tparam F callable type
    @tparam ArgsT parameter types

    @param f callable object to call
    @param args parameters to pass to the callable

    @return a tf::Future that will holds the result of the execution

    The method creates an asynchronous task to launch the given
    function on the given arguments.
    The difference to tf::Executor::async is that the created asynchronous task
    pertains to the subflow.
    When the subflow joins, all asynchronous tasks created from the subflow
    are guaranteed to finish before the join.
    For example:

    @code{.cpp}
    std::atomic<int> counter(0);
    taskflow.empalce([&](tf::Subflow& sf){
      for(int i=0; i<100; i++) {
        sf.async([&](){ counter++; });
      }
      sf.join();
      assert(counter == 100);
    });
    @endcode

    This method is thread-safe and can be called by multiple tasks in the
    subflow at the same time.

    @attention
    You cannot create asynchronous tasks from a detached subflow.
    Doing this results in undefined behavior.
    */
    template <typename F, typename... ArgsT>
    auto async(F&& f, ArgsT&&... args);

    /**
    @brief runs the given function asynchronously and assigns the task a name

    @tparam F callable type
    @tparam ArgsT parameter types

    @param name name of the asynchronous task
    @param f callable object to call
    @param args parameters to pass to the callable

    @return a tf::Future that will holds the result of the execution

    The method creates a named asynchronous task to launch the given
    function on the given arguments.
    The difference from tf::Executor::async is that the created asynchronous task
    pertains to the subflow.
    When the subflow joins, all asynchronous tasks created from the subflow
    are guaranteed to finish before the join.
    For example:

    @code{.cpp}
    std::atomic<int> counter(0);
    taskflow.empalce([&](tf::Subflow& sf){
      for(int i=0; i<100; i++) {
        sf.async("name", [&](){ counter++; });
      }
      sf.join();
      assert(counter == 100);
    });
    @endcode

    This method is thread-safe and can be called by multiple tasks in the
    subflow at the same time.

    @attention
    You cannot create named asynchronous tasks from a detached subflow.
    Doing this results in undefined behavior.
    */
    template <typename F, typename... ArgsT>
    auto named_async(const std::string& name, F&& f, ArgsT&&... args);

    /**
    @brief similar to tf::Subflow::async but does not return a future object

    This member function is more efficient than tf::Subflow::async
    and is encouraged to use when there is no data returned.

    @code{.cpp}
    taskflow.empalce([&](tf::Subflow& sf){
      for(int i=0; i<100; i++) {
        sf.silent_async([&](){ counter++; });
      }
      sf.join();
      assert(counter == 100);
    });
    @endcode

    This member function is thread-safe.
    */
    template <typename F, typename... ArgsT>
    void silent_async(F&& f, ArgsT&&... args);

    /**
    @brief similar to tf::Subflow::named_async but does not return a future object

    This member function is more efficient than tf::Subflow::named_async
    and is encouraged to use when there is no data returned.

    @code{.cpp}
    taskflow.empalce([&](tf::Subflow& sf){
      for(int i=0; i<100; i++) {
        sf.named_silent_async("name", [&](){ counter++; });
      }
      sf.join();
      assert(counter == 100);
    });
    @endcode

    This member function is thread-safe.
    */
    template <typename F, typename... ArgsT>
    void named_silent_async(const std::string& name, F&& f, ArgsT&&... args);

    /**
    @brief returns the executor that runs this subflow
    */
    inline Executor& executor();

  private:

    Executor& _executor;
    Worker& _worker;
    Node* _parent;
    bool _joinable {true};

    Subflow(Executor&, Worker&, Node*, Graph&);

    template <typename F, typename... ArgsT>
    auto _named_async(Worker& w, const std::string& name, F&& f, ArgsT&&... args);

    template <typename F, typename... ArgsT>
    void _named_silent_async(Worker& w, const std::string& name, F&& f, ArgsT&&... args);
};

// Constructor
inline Subflow::Subflow(
  Executor& executor, Worker& worker, Node* parent, Graph& graph
) :
  FlowBuilder {graph},
  _executor   {executor},
  _worker     {worker},
  _parent     {parent} {
  // assert(_parent != nullptr);
}

// Function: joined
inline bool Subflow::joinable() const noexcept {
  return _joinable;
}

// Function: executor
inline Executor& Subflow::executor() {
  return _executor;
}

// Procedure: reset
inline void Subflow::reset(bool clear_graph) {
  if(clear_graph) {
    _graph._clear();
  }
  _joinable = true;
}

}  // end of namespace tf. ---------------------------------------------------










