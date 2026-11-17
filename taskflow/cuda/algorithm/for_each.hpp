#pragma once

#include "../cudaflow.hpp"

/**
@file taskflow/cuda/algorithm/for_each.hpp
@brief cuda parallel-iteration algorithms include file
*/

namespace tf {

namespace detail {

/**
@private
*/
template <typename E, typename I, typename C>
__global__ void cuda_for_each_kernel(I first, unsigned count, C c) {
  auto tid  = threadIdx.x;
  auto bid  = blockIdx.x;
  auto tile = cuda_get_tile(bid, E::nv, count);
  cuda_strided_iterate<E::nt, E::vt>(
    [=](auto, auto j) {
      c(*(first + tile.begin + j));
    }, 
    tid, tile.count()
  );
}

/** @private */
template <typename E, typename I, typename C>
__global__ void cuda_for_each_index_kernel(I first, I inc, unsigned count, C c) {
  auto tid = threadIdx.x;
  auto bid = blockIdx.x;
  auto tile = cuda_get_tile(bid, E::nv, count);
  cuda_strided_iterate<E::nt, E::vt>(
    [=]__device__(auto, auto j) {
      c(first + inc*(tile.begin+j));
    }, 
    tid, tile.count()
  );
}

}  // end of namespace detail -------------------------------------------------

// ----------------------------------------------------------------------------
// cudaFlow: for_each, for_each_index
// ----------------------------------------------------------------------------

// Function: for_each
template <typename E, typename I, typename C>
cudaTask cudaGraph::for_each(I first, I last, C c) {

  unsigned count = std::distance(first, last);
  
  return kernel(
    E::num_blocks(count), E::nt, 0, 
    detail::cuda_for_each_kernel<E, I, C>, first, count, c
  );
}

// Function: for_each
template <typename E, typename I, typename C>
void cudaGraphExec::for_each(cudaTask task, I first, I last, C c) {
  
  unsigned count = std::distance(first, last);

  kernel(task, 
    E::num_blocks(count), E::nt, 0, 
    detail::cuda_for_each_kernel<E, I, C>, first, count, c
  );
}

// Function: for_each_index
template <typename E, typename I, typename C>
cudaTask cudaGraph::for_each_index(I first, I last, I inc, C c) {

  unsigned count = distance(first, last, inc);

  return kernel(
    E::num_blocks(count), E::nt, 0, 
    detail::cuda_for_each_index_kernel<E, I, C>, first, inc, count, c
  );
}

// Function: for_each_index
template <typename E, typename I, typename C>
void cudaGraphExec::for_each_index(cudaTask task, I first, I last, I inc, C c) {
  
  unsigned count = distance(first, last, inc);

  return kernel(task,
    E::num_blocks(count), E::nt, 0, 
    detail::cuda_for_each_index_kernel<E, I, C>, first, inc, count, c
  );
}


}  // end of namespace tf -----------------------------------------------------






