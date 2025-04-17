#pragma once

/**
@file taskflow/cuda/algorithm/single_task.hpp
@brief cuda single-task algorithms include file
*/

namespace tf {

/** @private */
template <typename C>
__global__ void cuda_single_task(C callable) {
  callable();
}

// Function: single_task
template <typename Creator, typename Deleter>
template <typename C>
cudaTask cudaGraphBase<Creator, Deleter>::single_task(C c) {
  return kernel(1, 1, 0, cuda_single_task<C>, c);
}

// Function: single_task
template <typename Creator, typename Deleter>
template <typename C>
void cudaGraphExecBase<Creator, Deleter>::single_task(cudaTask task, C c) {
  return kernel(task, 1, 1, 0, cuda_single_task<C>, c);
}

}  // end of namespace tf -----------------------------------------------------






