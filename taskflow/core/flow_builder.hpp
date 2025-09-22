﻿#pragma once

#include "task.hpp"
#include "../algorithm/partitioner.hpp"

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
#ifdef TF_ENABLE_DELEGATE
    template <auto C>
        requires is_static_task_v<decltype(C)>
    Task emplace();
    template <auto C, typename T>
        requires is_static_task_with_arg_v<decltype(C), T>
    Task emplace(T&& value_or_value_or_instance);


    // Runtime without value_or_instance
    template <auto C>
        requires is_runtime_task_v<decltype(C)>
    Task emplace();
    // Runtime with value_or_instance
    template <auto C, typename T>
        requires is_runtime_task_with_arg_v<decltype(C), T>
    Task emplace(T&& value_or_instance);

    // Subflow without value_or_instance
    template <auto C>
        requires is_subflow_task_v<decltype(C)>
    Task emplace();
    // Subflow with value_or_instance
    template <auto C, typename T>
        requires is_subflow_task_with_arg_v<decltype(C), T>
    Task emplace(T&& value_or_instance);

    // Condition without value_or_instance
    template <auto C>
        requires is_condition_task_v<decltype(C)>
    Task emplace();
    // Condition with value_or_instance
    template <auto C, typename T>
        requires is_condition_task_with_arg_v<decltype(C), T>
    Task emplace(T&& value_or_instance);

    // Multi-condition without value_or_instance
    template <auto C>
        requires is_multi_condition_task_v<decltype(C)>
    Task emplace();
    // Multi-condition with value_or_instance
    template <auto C, typename T>
        requires is_multi_condition_task_with_arg_v<decltype(C), T>
    Task emplace(T&& value_or_instance);


    template <auto... C>
        requires (sizeof...(C) > 1)
    auto emplace();

    template <auto... C, typename... T>
        requires (sizeof...(C) > 1 && sizeof...(C) == sizeof...(T))
    auto emplace(T&&... ts);


#else
    /**
  @brief creates a static task

  @tparam C callable type constructible from std::function<void()>

  @param callable callable to construct a static task

  @return a tf::Task handle

  The following example creates a static task.

  @code{.cpp}
  tf::Task static_task = taskflow.emplace([](){});
  @endcode

  @note
  Please refer to @ref StaticTasking for details.
  */
    template <typename C,
             std::enable_if_t<is_static_task_v<C>, void>* = nullptr
             >
    Task emplace(C&& callable);

    /**
  @brief creates a runtime task

  @tparam C callable type constructible from std::function<void(tf::Runtime&)>

  @param callable callable to construct a runtime task

  @return a tf::Task handle

  The following example creates a runtime task.

  @code{.cpp}
  tf::Task static_task = taskflow.emplace([](tf::Runtime&){});
  @endcode

  @note
  Please refer to @ref RuntimeTasking for details.
  */
    template <typename C,
             std::enable_if_t<is_runtime_task_v<C>, void>* = nullptr
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

  @note
  Please refer to @ref SubflowTasking for details.
  */
    template <typename C,
             std::enable_if_t<is_subflow_task_v<C>, void>* = nullptr
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

  @note
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

  @note
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
#endif
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
  @c init will run first and then @c comp which spawns its definition
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

  @note
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
  @brief constructs an STL-styled parallel-for task

  @tparam B beginning iterator type
  @tparam E ending iterator type
  @tparam C callable type
  @tparam P partitioner type (default tf::DefaultPartitioner)

  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  @param callable callable object to apply to the dereferenced iterator
  @param part partitioning algorithm to schedule parallel iterations

  @return a tf::Task handle

  The task spawns asynchronous tasks that applies the callable object to each object
  obtained by dereferencing every iterator in the range <tt>[first, last)</tt>.
  This method is equivalent to the parallel execution of the following loop:

  @code{.cpp}
  for(auto itr=first; itr!=last; itr++) {
    callable(*itr);
  }
  @endcode

  Iterators can be made stateful by using std::reference_wrapper
  The callable needs to take a single argument of
  the dereferenced iterator type.

  @note
  Please refer to @ref ParallelIterations for details.
  */
    template <typename B, typename E, typename C, typename P = DefaultPartitioner>
    Task for_each(B first, E last, C callable, P part = P());

    /**
  @brief constructs an index-based parallel-for task

  @tparam B beginning index type (must be integral)
  @tparam E ending index type (must be integral)
  @tparam S step type (must be integral)
  @tparam C callable type
  @tparam P partitioner type (default tf::DefaultPartitioner)

  @param first index of the beginning (inclusive)
  @param last index of the end (exclusive)
  @param step step size
  @param callable callable object to apply to each valid index
  @param part partitioning algorithm to schedule parallel iterations

  @return a tf::Task handle

  The task spawns asynchronous tasks that applies the callable object to each index
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

  Iterators can be made stateful by using std::reference_wrapper
  The callable needs to take a single argument of the integral index type.

  @note
  Please refer to @ref ParallelIterations for details.
  */
    template <typename B, typename E, typename S, typename C, typename P = DefaultPartitioner>
    Task for_each_index(B first, E last, S step, C callable, P part = P());

    /**
  @brief constructs an index range-based parallel-for task

  @tparam R index range type (tf::IndexRange)
  @tparam C callable type
  @tparam P partitioner type (default tf::DefaultPartitioner)

  @param range index range
  @param callable callable object to apply to each valid index
  @param part partitioning algorithm to schedule parallel iterations

  @return a tf::Task handle

  The task spawns asynchronous tasks that applies the callable object to
  in the range <tt>[first, last)</tt> with the step size.

  @code{.cpp}
  // [0, 17) with a step size of 2 using tf::IndexRange
  tf::IndexRange<int> range(0, 17, 2);

  // parallelize the sequence [0, 2, 4, 6, 8, 10, 12, 14, 16]
  taskflow.for_each_by_index(range, [](tf::IndexRange<int> range) {
    // iterate each index in the subrange
    for(int i=range.begin(); i<range.end(); i+=range.step_size()) {
      printf("iterate %d\n", i);
    }
  });

  executor.run(taskflow).wait();
  @endcode

  The callable needs to take a single argument of type tf::IndexRange.

  @note
  Please refer to @ref ParallelIterations for details.
  */
    template <typename R, typename C, typename P = DefaultPartitioner>
    Task for_each_by_index(R range, C callable, P part = P());

    // ------------------------------------------------------------------------
    // transform
    // ------------------------------------------------------------------------

    /**
  @brief constructs a parallel-transform task

  @tparam B beginning input iterator type
  @tparam E ending input iterator type
  @tparam O output iterator type
  @tparam C callable type
  @tparam P partitioner type (default tf::DefaultPartitioner)

  @param first1 iterator to the beginning of the first range
  @param last1 iterator to the end of the first range
  @param d_first iterator to the beginning of the output range
  @param c an unary callable to apply to dereferenced input elements
  @param part partitioning algorithm to schedule parallel iterations

  @return a tf::Task handle

  The task spawns asynchronous tasks that applies the callable object to an
  input range and stores the result in another output range.
  This method is equivalent to the parallel execution of the following loop:

  @code{.cpp}
  while (first1 != last1) {
    *d_first++ = c(*first1++);
  }
  @endcode

  Iterators can be made stateful by using std::reference_wrapper
  The callable needs to take a single argument of the dereferenced
  iterator type.

  @note
  Please refer to @ref ParallelTransforms for details.
  */
    template <
        typename B, typename E, typename O, typename C, typename P = DefaultPartitioner,
        std::enable_if_t<is_partitioner_v<std::decay_t<P>>, void>* = nullptr
        >
    Task transform(B first1, E last1, O d_first, C c, P part = P());

    /**
  @brief constructs a parallel-transform task

  @tparam B1 beginning input iterator type for the first input range
  @tparam E1 ending input iterator type for the first input range
  @tparam B2 beginning input iterator type for the first second range
  @tparam O output iterator type
  @tparam C callable type
  @tparam P partitioner type (default tf::DefaultPartitioner)

  @param first1 iterator to the beginning of the first input range
  @param last1 iterator to the end of the first input range
  @param first2 iterator to the beginning of the second input range
  @param d_first iterator to the beginning of the output range
  @param c a binary operator to apply to dereferenced input elements
  @param part partitioning algorithm to schedule parallel iterations

  @return a tf::Task handle

  The task spawns asynchronous tasks that applies the callable object to two
  input ranges and stores the result in another output range.
  This method is equivalent to the parallel execution of the following loop:

  @code{.cpp}
  while (first1 != last1) {
    *d_first++ = c(*first1++, *first2++);
  }
  @endcode

  Iterators can be made stateful by using std::reference_wrapper
  The callable needs to take two arguments of dereferenced elements
  from the two input ranges.

  @note
  Please refer to @ref ParallelTransforms for details.
  */
    template <
        typename B1, typename E1, typename B2, typename O, typename C, typename P=DefaultPartitioner,
        std::enable_if_t<!is_partitioner_v<std::decay_t<C>>, void>* = nullptr
        >
    Task transform(B1 first1, E1 last1, B2 first2, O d_first, C c, P part = P());

    // ------------------------------------------------------------------------
    // reduction
    // ------------------------------------------------------------------------

    /**
  @brief constructs an STL-styled parallel-reduction task

  @tparam B beginning iterator type
  @tparam E ending iterator type
  @tparam T result type
  @tparam O binary reducer type
  @tparam P partitioner type (default tf::DefaultPartitioner)

  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  @param init initial value of the reduction and the storage for the reduced result
  @param bop binary operator that will be applied
  @param part partitioning algorithm to schedule parallel iterations

  @return a tf::Task handle

  The task spawns asynchronous tasks to perform parallel reduction over @c init
  and the elements in the range <tt>[first, last)</tt>.
  The reduced result is store in @c init.
  This method is equivalent to the parallel execution of the following loop:

  @code{.cpp}
  for(auto itr=first; itr!=last; itr++) {
    init = bop(init, *itr);
  }
  @endcode

  Iterators can be made stateful by using std::reference_wrapper

  @note
  Please refer to @ref ParallelReduction for details.
  */
    template <typename B, typename E, typename T, typename O, typename P = DefaultPartitioner>
    Task reduce(B first, E last, T& init, O bop, P part = P());

    /**
  @brief constructs an index range-based parallel-reduction task

  @tparam R index range type (tf::IndexRange)
  @tparam T result type
  @tparam L local reducer type
  @tparam G global reducer type
  @tparam P partitioner type (default tf::DefaultPartitioner)

  @param range index range
  @param init initial value of the reduction and the storage for the reduced result
  @param lop binary operator that will be applied locally per worker
  @param gop binary operator that will be applied globally among worker
  @param part partitioning algorithm to schedule parallel iterations

  @return a tf::Task handle

  The task spawns asynchronous tasks to perform parallel reduction over a range with @c init.
  The reduced result is store in @c init.
  Unlike the iterator-based reduction,
  index range-based reduction is particularly useful for applications that benefit from SIMD optimizations
  or other range-based processing strategies.

  @code{.cpp}
  const size_t N = 1000000;
  std::vector<int> data(N);  // uninitialized data vector
  int res = 1;               // res will participate in the reduction

  taskflow.reduce_by_index(
    tf::IndexRange<size_t>(0, N, 1),
    // final result
    res,
    // local reducer
    [&](tf::IndexRange<size_t> subrange, std::optional<int> running_total) -> int {
      int residual = running_total ? *running_total : 0.0;
      for(size_t i=subrange.begin(); i<subrange.end(); i+=subrange.step_size()) {
        data[i] = 1.0;
        residual += data[i];
      }
      printf("partial sum = %lf\n", residual);
      return residual;
    },
    // global reducer
    std::plus<int>()
  );
  executor.run(taskflow).wait();
  assert(res = N + 1);
  @endcode

  Range can be made stateful by using std::reference_wrapper.

  @note
  Please refer to @ref ParallelReduction for details.
  */
    template <typename R, typename T, typename L, typename G, typename P = DefaultPartitioner>
    Task reduce_by_index(R range, T& init, L lop, G gop, P part = P());

    // ------------------------------------------------------------------------
    // transform and reduction
    // ------------------------------------------------------------------------

    /**
  @brief constructs an STL-styled parallel transform-reduce task

  @tparam B beginning iterator type
  @tparam E ending iterator type
  @tparam T result type
  @tparam BOP binary reducer type
  @tparam UOP unary transformation type
  @tparam P partitioner type (default tf::DefaultPartitioner)

  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  @param init initial value of the reduction and the storage for the reduced result
  @param bop binary operator that will be applied in unspecified order to the results of @c uop
  @param uop unary operator that will be applied to transform each element in the range to the result type
  @param part partitioning algorithm to schedule parallel iterations

  @return a tf::Task handle

  The task spawns asynchronous tasks to perform parallel reduction over @c init and
  the transformed elements in the range <tt>[first, last)</tt>.
  The reduced result is store in @c init.
  This method is equivalent to the parallel execution of the following loop:

  @code{.cpp}
  for(auto itr=first; itr!=last; itr++) {
    init = bop(init, uop(*itr));
  }
  @endcode

  Iterators can be made stateful by using std::reference_wrapper

  @note
  Please refer to @ref ParallelReduction for details.
  */
    template <
        typename B, typename E, typename T, typename BOP, typename UOP, typename P = DefaultPartitioner,
        std::enable_if_t<is_partitioner_v<std::decay_t<P>>, void>* = nullptr
        >
    Task transform_reduce(B first, E last, T& init, BOP bop, UOP uop, P part = P());

    /**
  @brief constructs an STL-styled parallel transform-reduce task
  @tparam B1 first beginning iterator type
  @tparam E1 first ending iterator type
  @tparam B2 second beginning iterator type
  @tparam T result type
  @tparam BOP_R binary reducer type
  @tparam BOP_T binary transformation type
  @tparam P partitioner type (default tf::DefaultPartitioner)

  @param first1 iterator to the beginning of the first range (inclusive)
  @param last1 iterator to the end of the first range (exclusive)
  @param first2 iterator to the beginning of the second range
  @param init initial value of the reduction and the storage for the reduced result
  @param bop_r binary operator that will be applied in unspecified order to the results of @c bop_t
  @param bop_t binary operator that will be applied to transform each element in the range to the result type
  @param part partitioning algorithm to schedule parallel iterations

  @return a tf::Task handle

  The task spawns asynchronous tasks to perform parallel reduction over @c init and
  transformed elements in the range <tt>[first, last)</tt>.
  The reduced result is store in @c init.
  This method is equivalent to the parallel execution of the following loop:

  @code{.cpp}
  for(auto itr1=first1, itr2=first2; itr1!=last1; itr1++, itr2++) {
    init = bop_r(init, bop_t(*itr1, *itr2));
  }
  @endcode

  Iterators can be made stateful by using std::reference_wrapper

  @note
  Please refer to @ref ParallelReduction for details.
  */

    template <
        typename B1, typename E1, typename B2, typename T, typename BOP_R, typename BOP_T,
        typename P = DefaultPartitioner,
        std::enable_if_t<!is_partitioner_v<std::decay_t<BOP_T>>, void>* = nullptr
        >
    Task transform_reduce(
        B1 first1, E1 last1, B2 first2, T& init, BOP_R bop_r, BOP_T bop_t, P part = P()
        );

    // ------------------------------------------------------------------------
    // scan
    // ------------------------------------------------------------------------

    /**
  @brief creates an STL-styled parallel inclusive-scan task

  @tparam B beginning iterator type
  @tparam E ending iterator type
  @tparam D destination iterator type
  @tparam BOP summation operator type

  @param first start of input range
  @param last end of input range
  @param d_first start of output range (may be the same as input range)
  @param bop function to perform summation

  Performs the cumulative sum (aka prefix sum, aka scan) of the input range
  and writes the result to the output range.
  Each element of the output range contains the
  running total of all earlier elements using the given binary operator
  for summation.

  This function generates an @em inclusive scan, meaning that the N-th element
  of the output range is the sum of the first N input elements,
  so the N-th input element is included.

  @code{.cpp}
  std::vector<int> input = {1, 2, 3, 4, 5};
  taskflow.inclusive_scan(
    input.begin(), input.end(), input.begin(), std::plus<int>{}
  );
  executor.run(taskflow).wait();

  // input is {1, 3, 6, 10, 15}
  @endcode

  Iterators can be made stateful by using std::reference_wrapper

  @note
  Please refer to @ref ParallelScan for details.
  */
    template <typename B, typename E, typename D, typename BOP>
    Task inclusive_scan(B first, E last, D d_first, BOP bop);

    /**
  @brief creates an STL-styled parallel inclusive-scan task with an initial value

  @tparam B beginning iterator type
  @tparam E ending iterator type
  @tparam D destination iterator type
  @tparam BOP summation operator type
  @tparam T initial value type

  @param first start of input range
  @param last end of input range
  @param d_first start of output range (may be the same as input range)
  @param bop function to perform summation
  @param init initial value

  Performs the cumulative sum (aka prefix sum, aka scan) of the input range
  and writes the result to the output range.
  Each element of the output range contains the
  running total of all earlier elements (and the initial value)
  using the given binary operator for summation.

  This function generates an @em inclusive scan, meaning the N-th element
  of the output range is the sum of the first N input elements,
  so the N-th input element is included.

  @code{.cpp}
  std::vector<int> input = {1, 2, 3, 4, 5};
  taskflow.inclusive_scan(
    input.begin(), input.end(), input.begin(), std::plus<int>{}, -1
  );
  executor.run(taskflow).wait();

  // input is {0, 2, 5, 9, 14}
  @endcode

  Iterators can be made stateful by using std::reference_wrapper

  @note
  Please refer to @ref ParallelScan for details.

  */
    template <typename B, typename E, typename D, typename BOP, typename T>
    Task inclusive_scan(B first, E last, D d_first, BOP bop, T init);

    /**
  @brief creates an STL-styled parallel exclusive-scan task

  @tparam B beginning iterator type
  @tparam E ending iterator type
  @tparam D destination iterator type
  @tparam T initial value type
  @tparam BOP summation operator type

  @param first start of input range
  @param last end of input range
  @param d_first start of output range (may be the same as input range)
  @param init initial value
  @param bop function to perform summation

  Performs the cumulative sum (aka prefix sum, aka scan) of the input range
  and writes the result to the output range.
  Each element of the output range contains the
  running total of all earlier elements (and the initial value)
  using the given binary operator for summation.

  This function generates an @em exclusive scan, meaning the N-th element
  of the output range is the sum of the first N-1 input elements,
  so the N-th input element is not included.

  @code{.cpp}
  std::vector<int> input = {1, 2, 3, 4, 5};
  taskflow.exclusive_scan(
    input.begin(), input.end(), input.begin(), -1, std::plus<int>{}
  );
  executor.run(taskflow).wait();

  // input is {-1, 0, 2, 5, 9}
  @endcode

  Iterators can be made stateful by using std::reference_wrapper

  @note
  Please refer to @ref ParallelScan for details.
  */
    template <typename B, typename E, typename D, typename T, typename BOP>
    Task exclusive_scan(B first, E last, D d_first, T init, BOP bop);

    // ------------------------------------------------------------------------
    // transform scan
    // ------------------------------------------------------------------------

    /**
  @brief creates an STL-styled parallel transform-inclusive scan task

  @tparam B beginning iterator type
  @tparam E ending iterator type
  @tparam D destination iterator type
  @tparam BOP summation operator type
  @tparam UOP transform operator type

  @param first start of input range
  @param last end of input range
  @param d_first start of output range (may be the same as input range)
  @param bop function to perform summation
  @param uop function to transform elements of the input range

  Write the cumulative sum (aka prefix sum, aka scan) of the input range
  to the output range. Each element of the output range contains the
  running total of all earlier elements
  using @c uop to transform the input elements
  and using @c bop for summation.

  This function generates an @em inclusive scan, meaning the Nth element
  of the output range is the sum of the first N input elements,
  so the Nth input element is included.

  @code{.cpp}
  std::vector<int> input = {1, 2, 3, 4, 5};
  taskflow.transform_inclusive_scan(
    input.begin(), input.end(), input.begin(), std::plus<int>{},
    [] (int item) { return -item; }
  );
  executor.run(taskflow).wait();

  // input is {-1, -3, -6, -10, -15}
  @endcode

  Iterators can be made stateful by using std::reference_wrapper

  @note
  Please refer to @ref ParallelScan for details.
  */
    template <typename B, typename E, typename D, typename BOP, typename UOP>
    Task transform_inclusive_scan(B first, E last, D d_first, BOP bop, UOP uop);

    /**
  @brief creates an STL-styled parallel transform-inclusive scan task

  @tparam B beginning iterator type
  @tparam E ending iterator type
  @tparam D destination iterator type
  @tparam BOP summation operator type
  @tparam UOP transform operator type
  @tparam T initial value type

  @param first start of input range
  @param last end of input range
  @param d_first start of output range (may be the same as input range)
  @param bop function to perform summation
  @param uop function to transform elements of the input range
  @param init initial value

  Write the cumulative sum (aka prefix sum, aka scan) of the input range
  to the output range. Each element of the output range contains the
  running total of all earlier elements (including an initial value)
  using @c uop to transform the input elements
  and using @c bop for summation.

  This function generates an @em inclusive scan, meaning the Nth element
  of the output range is the sum of the first N input elements,
  so the Nth input element is included.

  @code{.cpp}
  std::vector<int> input = {1, 2, 3, 4, 5};
  taskflow.transform_inclusive_scan(
    input.begin(), input.end(), input.begin(), std::plus<int>{},
    [] (int item) { return -item; },
    -1
  );
  executor.run(taskflow).wait();

  // input is {-2, -4, -7, -11, -16}
  @endcode

  Iterators can be made stateful by using std::reference_wrapper

  @note
  Please refer to @ref ParallelScan for details.
  */
    template <typename B, typename E, typename D, typename BOP, typename UOP, typename T>
    Task transform_inclusive_scan(B first, E last, D d_first, BOP bop, UOP uop, T init);

    /**
  @brief creates an STL-styled parallel transform-exclusive scan task

  @tparam B beginning iterator type
  @tparam E ending iterator type
  @tparam D destination iterator type
  @tparam BOP summation operator type
  @tparam UOP transform operator type
  @tparam T initial value type

  @param first start of input range
  @param last end of input range
  @param d_first start of output range (may be the same as input range)
  @param bop function to perform summation
  @param uop function to transform elements of the input range
  @param init initial value

  Write the cumulative sum (aka prefix sum, aka scan) of the input range
  to the output range. Each element of the output range contains the
  running total of all earlier elements (including an initial value)
  using @c uop to transform the input elements
  and using @c bop for summation.

  This function generates an @em exclusive scan, meaning the Nth element
  of the output range is the sum of the first N-1 input elements,
  so the Nth input element is not included.

  @code{.cpp}
  std::vector<int> input = {1, 2, 3, 4, 5};
  taskflow.transform_exclusive_scan(
    input.begin(), input.end(), input.begin(), -1, std::plus<int>{},
    [](int item) { return -item; }
  );
  executor.run(taskflow).wait();

  // input is {-1, -2, -4, -7, -11}
  @endcode

  Iterators can be made stateful by using std::reference_wrapper

  @note
  Please refer to @ref ParallelScan for details.
  */
    template <typename B, typename E, typename D, typename T, typename BOP, typename UOP>
    Task transform_exclusive_scan(B first, E last, D d_first, T init, BOP bop, UOP uop);

    // ------------------------------------------------------------------------
    // find
    // ------------------------------------------------------------------------

    /**
  @brief constructs a task to perform STL-styled find-if algorithm

  @tparam B beginning iterator type
  @tparam E ending iterator type
  @tparam T resulting iterator type
  @tparam UOP unary predicate type
  @tparam P partitioner type

  @param first start of the input range
  @param last end of the input range
  @param result resulting iterator to the found element in the input range
  @param predicate unary predicate which returns @c true for the required element
  @param part partitioning algorithm (default tf::DefaultPartitioner)

  Returns an iterator to the first element in the range <tt>[first, last)</tt>
  that satisfies the given criteria (or last if there is no such iterator).
  This method is equivalent to the parallel execution of the following loop:

  @code{.cpp}
  auto find_if(InputIt first, InputIt last, UnaryPredicate p) {
    for (; first != last; ++first) {
      if (predicate(*first)){
        return first;
      }
    }
    return last;
  }
  @endcode

  For example, the code below find the element that satisfies the given
  criteria (value plus one is equal to 23) from an input range of 10 elements:

  @code{.cpp}
  std::vector<int> input = {1, 6, 9, 10, 22, 5, 7, 8, 9, 11};
  std::vector<int>::iterator result;
  taskflow.find_if(
    input.begin(), input.end(), [](int i){ return i+1 = 23; }, result
  );
  executor.run(taskflow).wait();
  assert(*result == 22);
  @endcode

  Iterators can be made stateful by using std::reference_wrapper
  */
    template <typename B, typename E, typename T, typename UOP, typename P = DefaultPartitioner>
    Task find_if(B first, E last, T &result, UOP predicate, P part = P());

    /**
  @brief constructs a task to perform STL-styled find-if-not algorithm

  @tparam B beginning iterator type
  @tparam E ending iterator type
  @tparam T resulting iterator type
  @tparam UOP unary predicate type
  @tparam P partitioner type

  @param first start of the input range
  @param last end of the input range
  @param result resulting iterator to the found element in the input range
  @param predicate unary predicate which returns @c false for the required element
  @param part partitioning algorithm (default tf::DefaultPartitioner)

  Returns an iterator to the first element in the range <tt>[first, last)</tt>
  that satisfies the given criteria (or last if there is no such iterator).
  This method is equivalent to the parallel execution of the following loop:

  @code{.cpp}
  auto find_if(InputIt first, InputIt last, UnaryPredicate p) {
    for (; first != last; ++first) {
      if (!predicate(*first)){
        return first;
      }
    }
    return last;
  }
  @endcode

  For example, the code below find the element that satisfies the given
  criteria (value is not equal to 1) from an input range of 10 elements:

  @code{.cpp}
  std::vector<int> input = {1, 1, 1, 1, 22, 1, 1, 1, 1, 1};
  std::vector<int>::iterator result;
  taskflow.find_if_not(
    input.begin(), input.end(), [](int i){ return i == 1; }, result
  );
  executor.run(taskflow).wait();
  assert(*result == 22);
  @endcode

  Iterators can be made stateful by using std::reference_wrapper
  */
    template <typename B, typename E, typename T, typename UOP, typename P = DefaultPartitioner>
    Task find_if_not(B first, E last, T &result, UOP predicate, P part = P());

    /**
  @brief constructs a task to perform STL-styled min-element algorithm

  @tparam B beginning iterator type
  @tparam E ending iterator type
  @tparam T resulting iterator type
  @tparam C comparator type
  @tparam P partitioner type

  @param first start of the input range
  @param last end of the input range
  @param result resulting iterator to the found element in the input range
  @param comp comparison function object
  @param part partitioning algorithm (default tf::DefaultPartitioner)

  Finds the smallest element in the <tt>[first, last)</tt>
  using the given comparison function object.
  The iterator to that smallest element is stored in @c result.
  This method is equivalent to the parallel execution of the following loop:

  @code{.cpp}
  if (first == last) {
    return last;
  }
  auto smallest = first;
  ++first;
  for (; first != last; ++first) {
    if (comp(*first, *smallest)) {
      smallest = first;
    }
  }
  return smallest;
  @endcode

  For example, the code below find the smallest element from an input
  range of 10 elements.

  @code{.cpp}
  std::vector<int> input = {1, 1, 1, 1, 1, -1, 1, 1, 1, 1};
  std::vector<int>::iterator result;
  taskflow.min_element(
    input.begin(), input.end(), std::less<int>(), result
  );
  executor.run(taskflow).wait();
  assert(*result == -1);
  @endcode

  Iterators can be made stateful by using std::reference_wrapper
  */
    template <typename B, typename E, typename T, typename C, typename P>
    Task min_element(B first, E last, T& result, C comp, P part);

    /**
  @brief constructs a task to perform STL-styled max-element algorithm

  @tparam B beginning iterator type
  @tparam E ending iterator type
  @tparam T resulting iterator type
  @tparam C comparator type
  @tparam P partitioner type

  @param first start of the input range
  @param last end of the input range
  @param result resulting iterator to the found element in the input range
  @param comp comparison function object
  @param part partitioning algorithm (default tf::DefaultPartitioner)

  Finds the largest element in the <tt>[first, last)</tt>
  using the given comparison function object.
  The iterator to that largest element is stored in @c result.
  This method is equivalent to the parallel execution of the following loop:

  @code{.cpp}
  if (first == last){
    return last;
  }
  auto largest = first;
  ++first;
  for (; first != last; ++first) {
    if (comp(*largest, *first)) {
      largest = first;
    }
  }
  return largest;
  @endcode

  For example, the code below find the largest element from an input
  range of 10 elements.

  @code{.cpp}
  std::vector<int> input = {1, 1, 1, 1, 1, 2, 1, 1, 1, 1};
  std::vector<int>::iterator result;
  taskflow.max_element(
    input.begin(), input.end(), std::less<int>(), result
  );
  executor.run(taskflow).wait();
  assert(*result == 2);
  @endcode

  Iterators can be made stateful by using std::reference_wrapper
  */
    template <typename B, typename E, typename T, typename C, typename P>
    Task max_element(B first, E last, T& result, C comp, P part);

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
  @param cmp comparison operator

  The task spawns asynchronous tasks to sort elements in the range
  <tt>[first, last)</tt> in parallel.

  Iterators can be made stateful by using std::reference_wrapper

  @note
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

  The task spawns asynchronous tasks to parallel sort elements in the range
  <tt>[first, last)</tt> using the @c std::less<T> comparator,
  where @c T is the dereferenced iterator type.

  Iterators can be made stateful by using std::reference_wrapper

  @note
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

#ifdef TF_ENABLE_DELEGATE
// Static without value_or_instance
template <auto C>
    requires is_static_task_v<decltype(C)>
Task FlowBuilder::emplace() {
    return Task{_graph._emplace_back(
        NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{},
        nullptr, nullptr, 0,
        Node::Static{WorkHandle<void()>{connect_arg<C>}}
        )};
}

// Static with value_or_instance
template <auto C, typename T>
    requires is_static_task_with_arg_v<decltype(C), T>
Task FlowBuilder::emplace(T&& value_or_instance) {
    return Task{_graph._emplace_back(
        NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{},
        nullptr, nullptr, 0,
        Node::Static{WorkHandle<void()>{connect_arg<C>, std::forward<T>(value_or_instance)}}
        )};
}

// Runtime without value_or_instance
template <auto C>
    requires is_runtime_task_v<decltype(C)>
Task FlowBuilder::emplace() {
    return Task{_graph._emplace_back(
        NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{},
        nullptr, nullptr, 0,
        Node::Runtime{WorkHandle<void(tf::Runtime&)>{connect_arg<C>}}
        )};
}

// Runtime with value_or_instance
template <auto C, typename T>
    requires is_runtime_task_with_arg_v<decltype(C), T>
Task FlowBuilder::emplace(T&& value_or_instance) {
    return Task{_graph._emplace_back(
        NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{},
        nullptr, nullptr, 0,
        Node::Runtime{WorkHandle<void(tf::Runtime&)>{connect_arg<C>, std::forward<T>(value_or_instance)}}
        )};
}

// Subflow without value_or_instance
template <auto C>
    requires is_subflow_task_v<decltype(C)>
Task FlowBuilder::emplace() {
    return Task{_graph._emplace_back(
        NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{},
        nullptr, nullptr, 0,
        Node::Subflow{WorkHandle<void(tf::Subflow&)>{connect_arg<C>}}
        )};
}

// Subflow with value_or_instance
template <auto C, typename T>
    requires is_subflow_task_with_arg_v<decltype(C), T>
Task FlowBuilder::emplace(T&& value_or_instance) {
    return Task{_graph._emplace_back(
        NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{},
        nullptr, nullptr, 0,
        Node::Subflow{WorkHandle<void(tf::Subflow&)>{connect_arg<C>, std::forward<T>(value_or_instance)}}
        )};
}

// Condition without value_or_instance
template <auto C>
    requires is_condition_task_v<decltype(C)>
Task FlowBuilder::emplace() {
    return Task{_graph._emplace_back(
        NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{},
        nullptr, nullptr, 0,
        Node::Condition{WorkHandle<int()>{connect_arg<C>}}
        )};
}

// Condition with value_or_instance
template <auto C, typename T>
    requires is_condition_task_with_arg_v<decltype(C), T>
Task FlowBuilder::emplace(T&& value_or_instance) {
    return Task{_graph._emplace_back(
        NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{},
        nullptr, nullptr, 0,
        Node::Condition{WorkHandle<int()>{connect_arg<C>, std::forward<T>(value_or_instance)}}
        )};
}

// Multi-condition without value_or_instance
template <auto C>
    requires is_multi_condition_task_v<decltype(C)>
Task FlowBuilder::emplace() {
    return Task{_graph._emplace_back(
        NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{},
        nullptr, nullptr, 0,
        Node::MultiCondition{WorkHandle<SmallVector<int>()>{connect_arg<C>}}
        )};
}

// Multi-condition with value_or_instance
template <auto C, typename T>
    requires is_multi_condition_task_with_arg_v<decltype(C), T>
Task FlowBuilder::emplace(T&& value_or_instance) {
    return Task{_graph._emplace_back(
        NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{},
        nullptr, nullptr, 0,
        Node::MultiCondition{WorkHandle<SmallVector<int>()>{connect_arg<C>, std::forward<T>(value_or_instance)}}
        )};
}

template <auto... C>
    requires (sizeof...(C) > 1)
auto FlowBuilder::emplace() {
    return std::make_tuple(emplace<C>()...);
}
template <auto... C, typename... T>
    requires (sizeof...(C) > 1 && sizeof...(C) == sizeof...(T))
auto FlowBuilder::emplace(T&&... ts) {
    return std::make_tuple(emplace<C>(std::forward<T>(ts))...);
}
#else
// Function: emplace
template <typename C, std::enable_if_t<is_static_task_v<C>, void>*>
Task FlowBuilder::emplace(C&& c) {
    return Task(_graph._emplace_back(NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{}, nullptr, nullptr, 0,
                                     std::in_place_type_t<Node::Static>{}, std::forward<C>(c)
                                     ));
}

// Function: emplace
template <typename C, std::enable_if_t<is_runtime_task_v<C>, void>*>
Task FlowBuilder::emplace(C&& c) {
    return Task(_graph._emplace_back(NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{}, nullptr, nullptr, 0,
                                     std::in_place_type_t<Node::Runtime>{}, std::forward<C>(c)
                                     ));
}

// Function: emplace
template <typename C, std::enable_if_t<is_subflow_task_v<C>, void>*>
Task FlowBuilder::emplace(C&& c) {
    return Task(_graph._emplace_back(NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{}, nullptr, nullptr, 0,
                                     std::in_place_type_t<Node::Subflow>{}, std::forward<C>(c)
                                     ));
}

// Function: emplace
template <typename C, std::enable_if_t<is_condition_task_v<C>, void>*>
Task FlowBuilder::emplace(C&& c) {
    return Task(_graph._emplace_back(NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{}, nullptr, nullptr, 0,
                                     std::in_place_type_t<Node::Condition>{}, std::forward<C>(c)
                                     ));
}

// Function: emplace
template <typename C, std::enable_if_t<is_multi_condition_task_v<C>, void>*>
Task FlowBuilder::emplace(C&& c) {
    return Task(_graph._emplace_back(NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{}, nullptr, nullptr, 0,
                                     std::in_place_type_t<Node::MultiCondition>{}, std::forward<C>(c)
                                     ));
}

// Function: emplace
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto FlowBuilder::emplace(C&&... cs) {
    return std::make_tuple(emplace(std::forward<C>(cs))...);
}
#endif
// Function: composed_of
template <typename T>
Task FlowBuilder::composed_of(T& object) {
    auto node = _graph._emplace_back(NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{}, nullptr, nullptr, 0,
                                     std::in_place_type_t<Node::Module>{}, object
                                     );
    return Task(node);
}

// Function: placeholder
inline Task FlowBuilder::placeholder() {
    auto node = _graph._emplace_back(NSTATE::NONE, ESTATE::NONE, DefaultTaskParams{}, nullptr, nullptr, 0,
                                     std::in_place_type_t<Node::Placeholder>{}
                                     );
    return Task(node);
}


// Function: erase
inline void FlowBuilder::erase(Task task) {

    if (!task._node) {
        return;
    }

    // remove task from its successors' predecessor list
    for(size_t i=0; i<task._node->_num_successors; ++i) {
        task._node->_edges[i]->_remove_predecessors(task._node);
    }

    // remove task from its precedessors' successor list
    for(size_t i=task._node->_num_successors; i<task._node->_edges.size(); ++i) {
        task._node->_edges[i]->_remove_successors(task._node);
    }

    _graph._erase(task._node);
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

tf::Subflow is spawned from the execution of a task to dynamically manage a
child graph that may depend on runtime variables.
You can explicitly join a subflow by calling tf::Subflow::join, respectively.
By default, the %Taskflow runtime will implicitly join a subflow it is is joinable.

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
    @brief queries if the subflow is joinable

    This member function queries if the subflow is joinable.
    When a subflow is joined, it becomes not joinable.

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
    @brief acquires the associated executor
    */
    Executor& executor() noexcept;

    /**
    @brief acquires the associated graph
    */
    Graph& graph() { return _graph; }

    /**
    @brief specifies whether to keep the subflow after it is joined

    @param flag `true` to retain the subflow after it is joined; `false` to discard it

    By default, the runtime automatically clears a spawned subflow once it is joined.
    Setting this flag to `true` allows the application to retain the subflow's structure
    for post-execution analysis like visualization.
    */
    void retain(bool flag) noexcept;

    /**
    @brief queries if the subflow will be retained after it is joined
    @return `true` if the subflow will be retained after it is joined; `false` otherwise
    */
    bool retain() const;

private:

    Subflow(Executor&, Worker&, Node*, Graph&);

    Subflow() = delete;
    Subflow(const Subflow&) = delete;
    Subflow(Subflow&&) = delete;

    Executor& _executor;
    Worker& _worker;
    Node* _parent;
};

// Constructor
inline Subflow::Subflow(Executor& executor, Worker& worker, Node* parent, Graph& graph) :
    FlowBuilder {graph},
    _executor   {executor},
    _worker     {worker},
    _parent     {parent} {

    // need to reset since there could have iterative control flow
    _parent->_nstate &= ~(NSTATE::JOINED_SUBFLOW | NSTATE::RETAIN_SUBFLOW);

    // clear the graph
    graph.clear();
}

// Function: joinable
inline bool Subflow::joinable() const noexcept {
    return !(_parent->_nstate & NSTATE::JOINED_SUBFLOW);
}

// Function: executor
inline Executor& Subflow::executor() noexcept {
    return _executor;
}

// Function: retain
inline void Subflow::retain(bool flag) noexcept {
    // default value is not to retain
    if TF_LIKELY(flag == true) {
        _parent->_nstate |= NSTATE::RETAIN_SUBFLOW;
    }
    else {
        _parent->_nstate &= ~NSTATE::RETAIN_SUBFLOW;
    }

    //_parent->_nstate = (_parent->_nstate & ~NSTATE::RETAIN_SUBFLOW) |
    //                   (-static_cast<int>(flag) & NSTATE::RETAIN_SUBFLOW);
}

// Function: retain
inline bool Subflow::retain() const {
    return _parent->_nstate & NSTATE::RETAIN_SUBFLOW;
}

}  // end of namespace tf. ---------------------------------------------------










