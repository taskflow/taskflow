#pragma once

#include "../taskflow.hpp"
#include "cuda_graph_exec.hpp"
#include "algorithm/single_task.hpp"

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

};

}  // end of namespace tf -----------------------------------------------------


