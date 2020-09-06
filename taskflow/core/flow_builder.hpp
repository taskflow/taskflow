#pragma once

#include "task.hpp"

namespace tf {

/** 
@class FlowBuilder

@brief building methods of a task dependency graph

*/
class FlowBuilder {

  friend class Executor;

  public:
    
    /**
    @brief creates a static task from a given callable object
    
    @tparam C callable type
    
    @param callable a callable object constructible from std::function<void()>

    @return a Task handle
    */
    template <typename C>
    std::enable_if_t<is_static_task_v<C>, Task> emplace(C&& callable);

    /**
    @brief creates a dynamic task from a given callable object
    
    @tparam C callable type
    
    @param callable a callable object constructible from std::function<void(Subflow&)>

    @return a Task handle
    */
    template <typename C>
    std::enable_if_t<is_dynamic_task_v<C>, Task> emplace(C&& callable);

    /**
    @brief creates a condition task from a given callable object
    
    @tparam C callable type
    
    @param callable a callable object constructible from std::function<int()>

    @return a Task handle
    */
    template <typename C>
    std::enable_if_t<is_condition_task_v<C>, Task> emplace(C&& callable);

#ifdef TF_ENABLE_CUDA
    /**
    @brief creates a cudaflow task from a given callable object
    
    @tparam C callable type
    
    @param callable a callable object constructible from std::function<void(cudaFlow&)>

    @return a Task handle
    */
    template <typename C>
    std::enable_if_t<is_cudaflow_task_v<C>, Task> emplace(C&& callable);
#endif 

    /**
    @brief creates multiple tasks from a list of callable objects
    
    @tparam C... callable types

    @param callables one or multiple callable objects constructible from each task category

    @return a Task handle
    */
    template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
    auto emplace(C&&... callables);

    /**
    @brief creates a module task from a taskflow

    @param taskflow a taskflow object for the module
    @return a Task handle
    */
    Task composed_of(Taskflow& taskflow);
    
    /**
    @brief creates an empty task

    @return a Task handle
    */
    Task placeholder();

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

    @return a Task handle

    The task spawns a subflow that applies the callable object to each object obtained by dereferencing every iterator in the range <tt>[first, last)</tt>. By default, we employ the guided partition algorithm with chunk size equal to one.
    
    This method is equivalent to the parallel execution of the following loop:
    
    @code{.cpp}
    for(auto itr=first; itr!=last; itr++) {
      callable(*itr);
    }
    @endcode
    
    Arguments templated to enable stateful passing using std::reference_wrapper. 
    
    The callable needs to take a single argument of the dereferenced type.
    */
    template <typename B, typename E, typename C>
    Task for_each(B&& first, E&& last, C&& callable);
    
    /**
    @brief constructs a STL-styled parallel-for task using the guided partition algorithm

    @tparam B beginning iterator type
    @tparam E ending iterator type
    @tparam C callable type
    @tparam H chunk size type

    @param beg iterator to the beginning (inclusive)
    @param end iterator to the end (exclusive)
    @param callable a callable object to apply to the dereferenced iterator 
    @param chunk_size chunk size

    @return a Task handle

    The task spawns a subflow that applies the callable object to each object obtained by dereferencing every iterator in the range <tt>[beg, end)</tt>. The runtime partitions the range into chunks of the given chunk size, where each chunk is processed by a worker.
    
    Arguments are templated to enable stateful passing using std::reference_wrapper. 
    
    The callable needs to take a single argument of the dereferenced type.
    */
    template <typename B, typename E, typename C, typename H = size_t>
    Task for_each_guided(B&& beg, E&& end, C&& callable, H&& chunk_size = 1);
    
    /**
    @brief constructs a STL-styled parallel-for task using the dynamic partition algorithm

    @tparam B beginning iterator type
    @tparam E ending iterator type
    @tparam C callable type
    @tparam H chunk size type

    @param beg iterator to the beginning (inclusive)
    @param end iterator to the end (exclusive)
    @param callable a callable object to apply to the dereferenced iterator 
    @param chunk_size chunk size

    @return a Task handle
    
    The task spawns a subflow that applies the callable object to each object obtained by dereferencing every iterator in the range <tt>[beg, end)</tt>. The runtime partitions the range into chunks of the given chunk size, where each chunk is processed by a worker.
    
    Arguments are templated to enable stateful passing using std::reference_wrapper. 
    
    The callable needs to take a single argument of the dereferenced type.
    */
    template <typename B, typename E, typename C, typename H = size_t>
    Task for_each_dynamic(B&& beg, E&& end, C&& callable, H&& chunk_size = 1);
    
    /**
    @brief constructs a STL-styled parallel-for task using the dynamic partition algorithm

    @tparam B beginning iterator type
    @tparam E ending iterator type
    @tparam C callable type
    @tparam H chunk size type

    @param beg iterator to the beginning (inclusive)
    @param end iterator to the end (exclusive)
    @param callable a callable object to apply to the dereferenced iterator 
    @param chunk_size chunk size

    @return a Task handle
    
    The task spawns a subflow that applies the callable object to each object obtained by dereferencing every iterator in the range <tt>[beg, end)</tt>. The runtime partitions the range into chunks of the given chunk size, where each chunk is processed by a worker. When the given chunk size is zero, the runtime distributes the work evenly across workers.
    
    Arguments are templated to enable stateful passing using std::reference_wrapper. 
    
    The callable needs to take a single argument of the dereferenced type.
    */
    template <typename B, typename E, typename C, typename H = size_t>
    Task for_each_static(
      B&& beg, E&& end, C&& callable, H&& chunk_size = 0
    );
    
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

    @return a Task handle
    
    The task spawns a subflow that applies the callable object to each index in the range <tt>[first, last)</tt> with the step size. By default, we employ the guided partition algorithm with chunk size equal to one.
    
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

    The callable needs to take a single argument of the index type.
    
    */
    template <typename B, typename E, typename S, typename C>
    Task for_each_index(B&& first, E&& last, S&& step, C&& callable);
    
    /**
    @brief constructs an index-based parallel-for task using the guided partition algorithm.
    
    @tparam B beginning index type (must be integral)
    @tparam E ending index type (must be integral)
    @tparam S step type (must be integral)
    @tparam C callable type
    @tparam H chunk size type

    @param beg index of the beginning (inclusive)
    @param end index of the end (exclusive)
    @param step step size 
    @param callable a callable object to apply to each valid index
    @param chunk_size chunk size (default 1)

    @return a Task handle
    
    The task spawns a subflow that applies the callable object to each index in the range <tt>[beg, end)</tt> with the step size. The runtime partitions the range into chunks of the given size, where each chunk is processed by a worker.

    Arguments are templated to enable stateful passing using std::reference_wrapper.

    The callable needs to take a single argument of the index type.
    */
    template <typename B, typename E, typename S, typename C, typename H = size_t>
    Task for_each_index_guided(
      B&& beg, E&& end, S&& step, C&& callable, H&& chunk_size = 1
    );
    
    /**
    @brief constructs an index-based parallel-for task using the dynamic partition algorithm.

    @tparam B beginning index type (must be integral)
    @tparam E ending index type (must be integral)
    @tparam S step type (must be integral)
    @tparam C callable type
    @tparam H chunk size type

    @param beg index of the beginning (inclusive)
    @param end index of the end (exclusive)
    @param step step size 
    @param callable a callable object to apply to each valid index
    @param chunk_size chunk size (default 1)

    @return a Task handle
    
    The task spawns a subflow that applies the callable object to each index in the range <tt>[beg, end)</tt> with the step size. The runtime partitions the range into chunks of the given size, where each chunk is processed by a worker.

    Arguments are templated to enable stateful passing using std::reference_wrapper.

    The callable needs to take a single argument of the index type.
    */
    template <typename B, typename E, typename S, typename C, typename H = size_t>
    Task for_each_index_dynamic(
      B&& beg, E&& end, S&& step, C&& callable, H&& chunk_size = 1
    );
    
    /**
    @brief constructs an index-based parallel-for task using the static partition algorithm.
    
    @tparam B beginning index type (must be integral)
    @tparam E ending index type (must be integral)
    @tparam S step type (must be integral)
    @tparam C callable type
    @tparam H chunk size type

    @param beg index of the beginning (inclusive)
    @param end index of the end (exclusive)
    @param step step size 
    @param callable a callable object to apply to each valid index
    @param chunk_size chunk size (default 0)

    @return a Task handle
    
    The task spawns a subflow that applies the callable object to each index in the range <tt>[beg, end)</tt> with the step size. The runtime partitions the range into chunks of the given size, where each chunk is processed by a worker. When the given chunk size is zero, the runtime distributes the work evenly across workers.

    Arguments are templated to enable stateful passing using std::reference_wrapper.

    The callable needs to take a single argument of the index type.
    */
    template <typename B, typename E, typename S, typename C, typename H = size_t>
    Task for_each_index_static(
      B&& beg, E&& end, S&& step, C&& callable, H&& chunk_size = 0
    );

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

    @return a Task handle
    
    The task spawns a subflow to perform parallel reduction over @c init and the elements in the range <tt>[first, last)</tt>. The reduced result is store in @c init. The runtime partitions the range into chunks of the given chunk size, where each chunk is processed by a worker. By default, we employ the guided partition algorithm.
    
    This method is equivalent to the parallel execution of the following loop:
    
    @code{.cpp}
    for(auto itr=first; itr!=last; itr++) {
      init = bop(init, *itr);
    }
    @endcode
    
    Arguments are templated to enable stateful passing using std::reference_wrapper. 
    */
    template <typename B, typename E, typename T, typename O>
    Task reduce(B&& first, E&& last, T& init, O&& bop);

    /**
    @brief constructs a STL-styled parallel-reduce task using the guided partition algorithm

    @tparam B beginning iterator type
    @tparam E ending iterator type
    @tparam T result type 
    @tparam O binary reducer type
    @tparam H chunk size type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param init initial value of the reduction and the storage for the reduced result
    @param bop binary operator that will be applied 
    @param chunk_size chunk size

    The task spawns a subflow to perform parallel reduction over @c init and the elements in the range <tt>[first, last)</tt>. The reduced result is store in @c init. The runtime partitions the range into chunks of size @c chunk_size, where each chunk is processed by a worker. 

    Arguments are templated to enable stateful passing using std::reference_wrapper. 

    @return a Task handle
    */
    template <typename B, typename E, typename T, typename O, typename H = size_t>
    Task reduce_guided(
      B&& first, E&& last, T& init, O&& bop, H&& chunk_size = 1
    );
    
    /**
    @brief constructs a STL-styled parallel-reduce task using the dynamic partition algorithm

    @tparam B beginning iterator type
    @tparam E ending iterator type
    @tparam T result type 
    @tparam O binary reducer type
    @tparam H chunk size type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param init initial value of the reduction and the storage for the reduced result
    @param bop binary operator that will be applied 
    @param chunk_size chunk size

    The task spawns a subflow to perform parallel reduction over @c init and the elements in the range <tt>[first, last)</tt>. The reduced result is store in @c init. The runtime partitions the range into chunks of size @c chunk_size, where each chunk is processed by a worker. 
    
    Arguments are templated to enable stateful passing using std::reference_wrapper. 

    @return a Task handle
    */
    template <typename B, typename E, typename T, typename O, typename H = size_t>
    Task reduce_dynamic(
      B&& first, E&& last, T& init, O&& bop, H&& chunk_size = 1
    );
    
    /**
    @brief constructs a STL-styled parallel-reduce task using the static partition algorithm

    @tparam B beginning iterator type
    @tparam E ending iterator type
    @tparam T result type 
    @tparam O binary reducer type
    @tparam H chunk size type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param init initial value of the reduction and the storage for the reduced result
    @param bop binary operator that will be applied 
    @param chunk_size chunk size

    The task spawns a subflow to perform parallel reduction over @c init and the elements in the range <tt>[first, last)</tt>. The reduced result is store in @c init. The runtime partitions the range into chunks of size @c chunk_size, where each chunk is processed by a worker. 
    
    Arguments are templated to enable stateful passing using std::reference_wrapper. 

    @return a Task handle
    */
    template <typename B, typename E, typename T, typename O, typename H = size_t>
    Task reduce_static(
      B&& first, E&& last, T& init, O&& bop, H&& chunk_size = 0
    );
    
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

    @return a Task handle
    
    The task spawns a subflow to perform parallel reduction over @c init and the transformed elements in the range <tt>[first, last)</tt>. The reduced result is store in @c init. The runtime partitions the range into chunks of the given chunk size, where each chunk is processed by a worker. By default, we employ the guided partition algorithm.
    
    This method is equivalent to the parallel execution of the following loop:
    
    @code{.cpp}
    for(auto itr=first; itr!=last; itr++) {
      init = bop(init, uop(*itr));
    }
    @endcode
    
    Arguments are templated to enable stateful passing using std::reference_wrapper. 
    */
    template <typename B, typename E, typename T, typename BOP, typename UOP>
    Task transform_reduce(B&& first, E&& last, T& init, BOP&& bop, UOP&& uop);
    
    /**
    @brief constructs a STL-styled parallel transform-reduce task using the guided partition algorithm
  
    @tparam B beginning iterator type
    @tparam E ending iterator type
    @tparam T result type 
    @tparam BOP binary reducer type
    @tparam UOP unary transformion type
    @tparam H chunk size type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param init initial value of the reduction and the storage for the reduced result
    @param bop binary operator that will be applied in unspecified order to the results of @c uop
    @param uop unary operator that will be applied to transform each element in the range to the result type
    @param chunk_size chunk size

    @return a Task handle
    
    The task spawns a subflow to perform parallel reduction over @c init and the transformed elements in the range <tt>[first, last)</tt>. The reduced result is store in @c init. The runtime partitions the range into chunks of size @c chunk_size, where each chunk is processed by a worker. 
    
    Arguments are templated to enable stateful passing using std::reference_wrapper. 
    */
    template <typename B, typename E, typename T, typename BOP, typename UOP, typename H = size_t>
    Task transform_reduce_guided(
      B&& first, E&& last, T& init, BOP&& bop, UOP&& uop, H&& chunk_size = 1
    );

    /**
    @brief constructs a STL-styled parallel transform-reduce task using the static partition algorithm
  
    @tparam B beginning iterator type
    @tparam E ending iterator type
    @tparam T result type 
    @tparam BOP binary reducer type
    @tparam UOP unary transformion type
    @tparam H chunk size type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param init initial value of the reduction and the storage for the reduced result
    @param bop binary operator that will be applied in unspecified order to the results of @c uop
    @param uop unary operator that will be applied to transform each element in the range to the result type
    @param chunk_size chunk size

    @return a Task handle
    
    The task spawns a subflow to perform parallel reduction over @c init and the transformed elements in the range <tt>[first, last)</tt>. The reduced result is store in @c init. The runtime partitions the range into chunks of size @c chunk_size, where each chunk is processed by a worker. 
    
    Arguments are templated to enable stateful passing using std::reference_wrapper. 
    */
    template <typename B, typename E, typename T, typename BOP, typename UOP, typename H = size_t>
    Task transform_reduce_static(
      B&& first, E&& last, T& init, BOP&& bop, UOP&& uop, H&& chunk_size = 0
    );

    /**
    @brief constructs a STL-styled parallel transform-reduce task using the dynamic partition algorithm
  
    @tparam B beginning iterator type
    @tparam E ending iterator type
    @tparam T result type 
    @tparam BOP binary reducer type
    @tparam UOP unary transformion type
    @tparam H chunk size type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param init initial value of the reduction and the storage for the reduced result
    @param bop binary operator that will be applied in unspecified order to the results of @c uop
    @param uop unary operator that will be applied to transform each element in the range to the result type
    @param chunk_size chunk size

    @return a Task handle
    
    The task spawns a subflow to perform parallel reduction over @c init and the transformed elements in the range <tt>[first, last)</tt>. The reduced result is store in @c init. The runtime partitions the range into chunks of size @c chunk_size, where each chunk is processed by a worker. 
    
    Arguments are templated to enable stateful passing using std::reference_wrapper. 
    */
    template <typename B, typename E, typename T, typename BOP, typename UOP, typename H = size_t>
    Task transform_reduce_dynamic(
      B&& first, E&& last, T& init, BOP&& bop, UOP&& uop, H&& chunk_size = 1
    );
    
    
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
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto FlowBuilder::emplace(C&&... cs) {
  return std::make_tuple(emplace(std::forward<C>(cs))...);
}

// Function: emplace
// emplaces a static task
template <typename C>
std::enable_if_t<is_static_task_v<C>, Task> FlowBuilder::emplace(C&& c) {
  auto n = _graph.emplace_back(
    nstd::in_place_type_t<Node::StaticWork>{}, std::forward<C>(c)
  );
  return Task(n);
}

// Function: emplace
// emplaces a dynamic task
template <typename C>
std::enable_if_t<is_dynamic_task_v<C>, Task> FlowBuilder::emplace(C&& c) {
  auto n = _graph.emplace_back(
    nstd::in_place_type_t<Node::DynamicWork>{}, std::forward<C>(c)
  );
  return Task(n);
}

// Function: emplace 
// emplaces a condition task
template <typename C>
std::enable_if_t<is_condition_task_v<C>, Task> FlowBuilder::emplace(C&& c) {
  auto n = _graph.emplace_back(
    nstd::in_place_type_t<Node::ConditionWork>{}, std::forward<C>(c)
  );
  return Task(n);
}

#ifdef TF_ENABLE_CUDA
// Function: emplace
// emplaces a cudaflow task
template <typename C>
std::enable_if_t<is_cudaflow_task_v<C>, Task> FlowBuilder::emplace(C&& c) {
  auto n = _graph.emplace_back(
    nstd::in_place_type_t<Node::cudaFlowWork>{}, std::forward<C>(c)
  );
  return Task(n);
}
#endif

// Function: composed_of    
inline Task FlowBuilder::composed_of(Taskflow& taskflow) {
  auto node = _graph.emplace_back(
    nstd::in_place_type_t<Node::ModuleWork>{}, &taskflow
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

@brief building methods of a subflow graph in dynamic tasking

By default, a subflow automatically joins its parent node. You may explicitly
join or detach a subflow by calling Subflow::join or Subflow::detach.

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

// ----------------------------------------------------------------------------
// Legacy code
// ----------------------------------------------------------------------------

using SubflowBuilder = Subflow;

}  // end of namespace tf. ---------------------------------------------------


