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
template <typename C>
cudaTask cudaGraph::single_task(C c) {
  return kernel(1, 1, 0, cuda_single_task<C>, c);
}

// Function: single_task
template <typename C>
void cudaGraphExec::single_task(cudaTask task, C c) {
  return kernel(task, 1, 1, 0, cuda_single_task<C>, c);
}

}  // end of namespace tf -----------------------------------------------------






