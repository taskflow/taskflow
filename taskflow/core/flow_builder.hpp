#pragma once

#include "task.hpp"

/** 
@file flow_builder.hpp
@brief flow builder include file
*/

namespace tf {

/** 
@class FlowBuilder

@brief building methods of a task dependency graph

*/
class FlowBuilder {

  friend class Executor;

  public:
    
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
    @brief creates a module task from a taskflow

    @param taskflow a taskflow object for the module

    @return a tf::Task handle

    Please refer to @ref ComposableTasking for details.
    */
    Task composed_of(Taskflow& taskflow);

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
    @brief creates a %syclFlow task on the given queue

    @tparam C callable type constructible from std::function<void(tf::syclFlow&)>
    @tparam Q queue type

    @return a tf::Task handle
    
    The following example creates a %syclFlow of two kernel tasks, @c task1 and 
    @c task2 on GPU @c 2, where @c task1 runs before @c task2
    
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
    Task emplace_on(C&& callable, Q& queue);

    /**
    @brief adds adjacent dependency links to a linear list of tasks

    @param tasks a vector of tasks
    */
    void linearize(std::vector<Task>& tasks);

    /**
    @brief adds adjacent dependency links to a linear list of tasks

    @param tasks an initializer list of tasks
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

    The task spawns a subflow that applies the callable object to each object obtained by dereferencing every iterator in the range <tt>[first, last)</tt>.
    This method is equivalent to the parallel execution of the following loop:
    
    @code{.cpp}
    for(auto itr=first; itr!=last; itr++) {
      callable(*itr);
    }
    @endcode
    
    Arguments templated to enable stateful passing using std::reference_wrapper. 
    The callable needs to take a single argument of 
    the dereferenced iterator type.

    Please refer to @ref ParallelIterations for details.
    */
    template <typename B, typename E, typename C>
    Task for_each(B&& first, E&& last, C&& callable);
    
    /**
    @brief constructs an index-based parallel-for task 

    @tparam B beginning index type (must be integral)
    @tparam E ending index type (must be integral)
    @tparam S step type (must be integral)
    @tparam C callable type

    @param first index of the beginning (inclusive)
    @param last index of the end (exclusive)
    @param step step size 
    @param callable a callable object to apply to each valid index

    @return a tf::Task handle
    
    The task spawns a subflow that applies the callable object to each index in the range <tt>[first, last)</tt> with the step size.
    
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

    Arguments are templated to enable stateful passing using std::reference_wrapper.
    The callable needs to take a single argument of the integral index type.
    
    Please refer to @ref ParallelIterations for details.
    */
    template <typename B, typename E, typename S, typename C>
    Task for_each_index(B&& first, E&& last, S&& step, C&& callable);
    
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
    
    The task spawns a subflow to perform parallel reduction over @c init and the elements in the range <tt>[first, last)</tt>. The reduced result is store in @c init.
    
    This method is equivalent to the parallel execution of the following loop:
    
    @code{.cpp}
    for(auto itr=first; itr!=last; itr++) {
      init = bop(init, *itr);
    }
    @endcode
    
    Arguments are templated to enable stateful passing using std::reference_wrapper. 

    Please refer to @ref ParallelReduction for details.
    */
    template <typename B, typename E, typename T, typename O>
    Task reduce(B&& first, E&& last, T& init, O&& bop);

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
    
    The task spawns a subflow to perform parallel reduction over @c init and the transformed elements in the range <tt>[first, last)</tt>. 
    The reduced result is store in @c init.
    
    This method is equivalent to the parallel execution of the following loop:
    
    @code{.cpp}
    for(auto itr=first; itr!=last; itr++) {
      init = bop(init, uop(*itr));
    }
    @endcode
    
    Arguments are templated to enable stateful passing using std::reference_wrapper. 
   
    Please refer to @ref ParallelReduction for details. 
    */
    template <typename B, typename E, typename T, typename BOP, typename UOP>
    Task transform_reduce(B&& first, E&& last, T& init, BOP&& bop, UOP&& uop);
    
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
    
    Arguments are templated to enable stateful passing using std::reference_wrapper. 

    Please refer to @ref ParallelSort for details.
    */
    template <typename B, typename E, typename C>
    Task sort(B&& first, E&& last, C&& cmp);
    
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

    Arguments are templated to enable stateful passing using std::reference_wrapper. 
    
    Please refer to @ref ParallelSort for details.
     */
    template <typename B, typename E>
    Task sort(B&& first, E&& last);
    
  protected:
    
    /**
    @brief constructs a flow builder with a graph
    */
    FlowBuilder(Graph& graph);
    
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
  return Task(_graph.emplace_back(
    std::in_place_type_t<Node::Static>{}, std::forward<C>(c)
  ));
}

// Function: emplace
template <typename C, std::enable_if_t<is_dynamic_task_v<C>, void>*>
Task FlowBuilder::emplace(C&& c) {
  return Task(_graph.emplace_back(
    std::in_place_type_t<Node::Dynamic>{}, std::forward<C>(c)
  ));
}

// Function: emplace
template <typename C, std::enable_if_t<is_condition_task_v<C>, void>*>
Task FlowBuilder::emplace(C&& c) {
  return Task(_graph.emplace_back(
    std::in_place_type_t<Node::Condition>{}, std::forward<C>(c)
  ));
}

// Function: emplace
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto FlowBuilder::emplace(C&&... cs) {
  return std::make_tuple(emplace(std::forward<C>(cs))...);
}

// Function: composed_of    
inline Task FlowBuilder::composed_of(Taskflow& taskflow) {
  auto node = _graph.emplace_back(
    std::in_place_type_t<Node::Module>{}, &taskflow
  );
  return Task(node);
}

// Function: placeholder
inline Task FlowBuilder::placeholder() {
  auto node = _graph.emplace_back();
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
// create three regular tasks
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
    */
    void join();

    /**
    @brief enables the subflow to detach from its parent task

    Performs an immediate action to detach the subflow. Once the subflow is detached,
    it is considered finished and you may not modify the subflow anymore.
    */
    void detach();
    
    /**
    @brief queries if the subflow is joinable

    When a subflow is joined or detached, it becomes not joinable.
    */
    bool joinable() const;

    /** 
    @brief runs a given function asynchronously

    @tparam F callable type
    @tparam ArgsT parameter types

    @param f callable object to call
    @param args parameters to pass to the callable
    
    @return a tf::Future that will holds the result of the execution

    This method is thread-safe and can be called by multiple tasks in the 
    subflow at the same time.
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

    You cannot create asynchronous tasks from a detached subflow.
    Doing this results in undefined behavior.
    */
    template <typename F, typename... ArgsT>
    auto async(F&& f, ArgsT&&... args);
    
    /**
    @brief similar to tf::Subflow::async but did not return a future object
     */
    template <typename F, typename... ArgsT>
    void silent_async(F&& f, ArgsT&&... args);

  private:
    
    Subflow(Executor&, Node*, Graph&);

    Executor& _executor;
    Node* _parent;

    bool _joinable {true};
};

// Constructor
inline Subflow::Subflow(Executor& executor, Node* parent, Graph& graph) :
  FlowBuilder {graph},
  _executor   {executor},
  _parent     {parent} {
}

// Function: joined
inline bool Subflow::joinable() const {
  return _joinable;
}

}  // end of namespace tf. ---------------------------------------------------


