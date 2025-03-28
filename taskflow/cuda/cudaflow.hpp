#pragma once

#include "../taskflow.hpp"
#include "cuda_graph_exec.hpp"
//#include "cuda_capturer.hpp"

/**
@file taskflow/cuda/cudaflow.hpp
@brief cudaFlow include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// class definition: cudaFlow
// ----------------------------------------------------------------------------

/**
@class cudaFlow

@brief class to create a %cudaFlow task dependency graph

A %cudaFlow is a high-level interface over CUDA Graph to perform GPU operations
using the task dependency graph model.
The class provides a set of methods for creating and launch different tasks
on one or multiple CUDA devices,
for instance, kernel tasks, data transfer tasks, and memory operation tasks.
The following example creates a %cudaFlow of two kernel tasks, @c task1 and
@c task2, where @c task1 runs before @c task2.

@code{.cpp}
tf::cudaStream stream;
tf::cudaFlow cf;

// create two kernel tasks
tf::cudaTask task1 = cf.kernel(grid1, block1, shm_size1, kernel1, args1);
tf::cudaTask task2 = cf.kernel(grid2, block2, shm_size2, kernel2, args2);

// kernel1 runs before kernel2
task1.precede(task2);

// create an executable graph from the cudaflow
cudaGraphExec exec = cf.instantiate();

// run the executable graph through the given stream
exec.run(stream);
@endcode

Please refer to @ref GPUTaskingcudaFlow for details.
*/
class cudaFlow : public cudaGraph {
  
  public:

  /**
  @brief constructs a %cudaFlow

  A cudaFlow is associated with a tf::cudaGraph that manages a native CUDA graph.
  */
  cudaFlow() = default;

  /**
  @brief destroys the %cudaFlow
   */
  ~cudaFlow() = default;

  // ------------------------------------------------------------------------
  // generic algorithms
  // ------------------------------------------------------------------------

  /**
  @brief runs a callable with only a single kernel thread

  @tparam C callable type

  @param c callable to run by a single kernel thread

  @return a tf::cudaTask handle
  */
  template <typename C>
  cudaTask single_task(C c);

  /**
  @brief updates a single-threaded kernel task

  This method is similar to cudaFlow::single_task but operates
  on an existing task.
  */
  template <typename C>
  void single_task(cudaTask task, C c);

  /**
  @brief applies a callable to each dereferenced element of the data array

  @tparam I iterator type
  @tparam C callable type

  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  @param callable a callable object to apply to the dereferenced iterator

  @return a tf::cudaTask handle

  This method is equivalent to the parallel execution of the following loop on a GPU:

  @code{.cpp}
  for(auto itr = first; itr != last; itr++) {
    callable(*itr);
  }
  @endcode
  */
  template <typename I, typename C>
  cudaTask for_each(I first, I last, C callable);

  /**
  @brief updates parameters of a kernel task created from
         tf::cudaFlow::for_each

  The type of the iterators and the callable must be the same as
  the task created from tf::cudaFlow::for_each.
  */
  template <typename I, typename C>
  void for_each(cudaTask task, I first, I last, C callable);

  /**
  @brief applies a callable to each index in the range with the step size

  @tparam I index type
  @tparam C callable type

  @param first beginning index
  @param last last index
  @param step step size
  @param callable the callable to apply to each element in the data array

  @return a tf::cudaTask handle

  This method is equivalent to the parallel execution of the following loop on a GPU:

  @code{.cpp}
  // step is positive [first, last)
  for(auto i=first; i<last; i+=step) {
    callable(i);
  }

  // step is negative [first, last)
  for(auto i=first; i>last; i+=step) {
    callable(i);
  }
  @endcode
  */
  template <typename I, typename C>
  cudaTask for_each_index(I first, I last, I step, C callable);

  /**
  @brief updates parameters of a kernel task created from
         tf::cudaFlow::for_each_index

  The type of the iterators and the callable must be the same as
  the task created from tf::cudaFlow::for_each_index.
  */
  template <typename I, typename C>
  void for_each_index(
    cudaTask task, I first, I last, I step, C callable
  );

  /**
  @brief applies a callable to a source range and stores the result in a target range

  @tparam I input iterator type
  @tparam O output iterator type
  @tparam C unary operator type

  @param first iterator to the beginning of the input range
  @param last iterator to the end of the input range
  @param output iterator to the beginning of the output range
  @param op the operator to apply to transform each element in the range

  @return a tf::cudaTask handle

  This method is equivalent to the parallel execution of the following loop on a GPU:

  @code{.cpp}
  while (first != last) {
    *output++ = callable(*first++);
  }
  @endcode
  */
  template <typename I, typename O, typename C>
  cudaTask transform(I first, I last, O output, C op);

  /**
  @brief updates parameters of a kernel task created from
         tf::cudaFlow::transform

  The type of the iterators and the callable must be the same as
  the task created from tf::cudaFlow::for_each.
  */
  template <typename I, typename O, typename C>
  void transform(cudaTask task, I first, I last, O output, C c);

  /**
  @brief creates a task to perform parallel transforms over two ranges of items

  @tparam I1 first input iterator type
  @tparam I2 second input iterator type
  @tparam O output iterator type
  @tparam C unary operator type

  @param first1 iterator to the beginning of the input range
  @param last1 iterator to the end of the input range
  @param first2 iterato
  @param output iterator to the beginning of the output range
  @param op binary operator to apply to transform each pair of items in the
            two input ranges

  @return cudaTask handle

  This method is equivalent to the parallel execution of the following loop on a GPU:

  @code{.cpp}
  while (first1 != last1) {
    *output++ = op(*first1++, *first2++);
  }
  @endcode
  */
  template <typename I1, typename I2, typename O, typename C>
  cudaTask transform(I1 first1, I1 last1, I2 first2, O output, C op);

  /**
  @brief updates parameters of a kernel task created from
         tf::cudaFlow::transform

  The type of the iterators and the callable must be the same as
  the task created from tf::cudaFlow::for_each.
  */
  template <typename I1, typename I2, typename O, typename C>
  void transform(
    cudaTask task, I1 first1, I1 last1, I2 first2, O output, C c
  );

  // ------------------------------------------------------------------------
  // subflow
  // ------------------------------------------------------------------------

  /**
  @brief constructs a subflow graph through tf::cudaFlowCapturer

  @tparam C callable type constructible from
            @c std::function<void(tf::cudaFlowCapturer&)>

  @param callable the callable to construct a capture flow

  @return a tf::cudaTask handle

  A captured subflow forms a sub-graph to the %cudaFlow and can be used to
  capture custom (or third-party) kernels that cannot be directly constructed
  from the %cudaFlow.

  Example usage:

  @code{.cpp}
  taskflow.emplace([&](tf::cudaFlow& cf){

    tf::cudaTask my_kernel = cf.kernel(my_arguments);

    // create a flow capturer to capture custom kernels
    tf::cudaTask my_subflow = cf.capture([&](tf::cudaFlowCapturer& capturer){
      capturer.on([&](cudaStream_t stream){
        invoke_custom_kernel_with_stream(stream, custom_arguments);
      });
    });

    my_kernel.precede(my_subflow);
  });
  @endcode
  */
  template <typename C>
  cudaTask capture(C&& callable);

  /**
  @brief updates the captured child graph

  The method is similar to tf::cudaFlow::capture but operates on a task
  of type tf::cudaTaskType::SUBFLOW.
  The new captured graph must be topologically identical to the original
  captured graph.
  */
  template <typename C>
  void capture(cudaTask task, C callable);
    
};

// ------------------------------------------------------------------------
// update methods
// ------------------------------------------------------------------------


//
//// Function: capture
//template <typename C>
//void cudaFlow::capture(cudaTask task, C c) {
//
//  if(task.type() != cudaTaskType::SUBFLOW) {
//    TF_THROW(task, " is not a subflow task");
//  }
//
//  // insert a subflow node
//  // construct a captured flow from the callable
//  auto node_handle = std::get_if<cudaFlowNode::Subflow>(&task._node->_handle);
//  //node_handle->graph.clear();
//
//  cudaFlowCapturer capturer;
//  c(capturer);
//
//  // obtain the optimized captured graph
//  capturer._cfg._native_handle.reset(capturer.capture());
//  node_handle->cfg = std::move(capturer._cfg);
//
//  TF_CHECK_CUDA(
//    cudaGraphExecChildGraphNodeSetParams(
//      _exe, 
//      task._node->_native_handle, 
//      node_handle->cfg._native_handle
//    ),
//    "failed to update a captured child graph"
//  );
//}

// ----------------------------------------------------------------------------
// captured flow
// ----------------------------------------------------------------------------

//// Function: capture
//template <typename C>
//cudaTask cudaFlow::capture(C&& c) {
//
//  // insert a subflow node
//  auto node = _cfg.emplace_back(
//    _cfg, std::in_place_type_t<cudaFlowNode::Subflow>{}
//  );
//
//  // construct a captured flow from the callable
//  auto node_handle = std::get_if<cudaFlowNode::Subflow>(&node->_handle);
//
//  // perform capturing
//  cudaFlowCapturer capturer;
//  c(capturer);
//
//  // obtain the optimized captured graph
//  capturer._cfg._native_handle.reset(capturer.capture());
//
//  // move capturer's cudaFlow graph into node
//  node_handle->cfg = std::move(capturer._cfg);
//
//  TF_CHECK_CUDA(
//    cudaGraphAddChildGraphNode(
//      &node->_native_handle, 
//      _cfg._native_handle, 
//      nullptr, 
//      0, 
//      node_handle->cfg._native_handle
//    ), 
//    "failed to add a cudaFlow capturer task"
//  );
//
//  return cudaTask(node);
//}

// 

}  // end of namespace tf -----------------------------------------------------


