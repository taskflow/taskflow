#pragma once

#include "../cudaflow.hpp"

/**
@file taskflow/cuda/algorithm/transform.hpp
@brief cuda parallel-transform algorithms include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// transform
// ----------------------------------------------------------------------------

namespace detail {

/**
@private
*/
template <typename E, typename I, typename O, typename C>
__global__ void cuda_transform_kernel(I first, unsigned count, O output, C op) {
  auto tid = threadIdx.x;
  auto bid = blockIdx.x;
  auto tile = cuda_get_tile(bid, E::nv, count);
  cuda_strided_iterate<E::nt, E::vt>(
    [=]__device__(auto, auto j) {
      auto offset = j + tile.begin;
      *(output + offset) = op(*(first+offset));
    }, 
    tid, 
    tile.count()
  );
}

/**
@private
*/
template <typename E, typename I1, typename I2, typename O, typename C>
__global__ void cuda_transform_kernel(
  I1 first1, I2 first2, unsigned count, O output, C op
) {
  auto tid = threadIdx.x;
  auto bid = blockIdx.x;
  auto tile = cuda_get_tile(bid, E::nv, count);
  cuda_strided_iterate<E::nt, E::vt>(
    [=]__device__(auto, auto j) {
      auto offset = j + tile.begin;
      *(output + offset) = op(*(first1+offset), *(first2+offset));
    }, 
    tid, 
    tile.count()
  );
}

}  // end of namespace detail -------------------------------------------------

// ----------------------------------------------------------------------------
// cudaFlow
// ----------------------------------------------------------------------------

// Function: transform
template <typename E, typename I, typename O, typename C>
cudaTask cudaGraph::transform(I first, I last, O output, C c) {
  
  unsigned count = std::distance(first, last);
  
  return kernel(
    E::num_blocks(count), E::nt, 0,
    detail::cuda_transform_kernel<E, I, O, C>,
    first, count, output, c
  );
}

// Function: transform
template <typename E, typename I1, typename I2, typename O, typename C>
cudaTask cudaGraph::transform(I1 first1, I1 last1, I2 first2, O output, C c) {
  
  unsigned count = std::distance(first1, last1);
  
  return kernel(
    E::num_blocks(count), E::nt, 0,
    detail::cuda_transform_kernel<E, I1, I2, O, C>,
    first1, first2, count, output, c
  );
}


// Function: update transform
template <typename E, typename I, typename O, typename C>
void cudaGraphExec::transform(cudaTask task, I first, I last, O output, C c) {
  
  unsigned count = std::distance(first, last);
  
  kernel(task,
    E::num_blocks(count), E::nt, 0,
    detail::cuda_transform_kernel<E, I, O, C>,
    first, count, output, c
  );
}

// Function: update transform
template <typename E, typename I1, typename I2, typename O, typename C>
void cudaGraphExec::transform(
  cudaTask task, I1 first1, I1 last1, I2 first2, O output, C c
) {
  unsigned count = std::distance(first1, last1);

  kernel(task,
    E::num_blocks(count), E::nt, 0,
    detail::cuda_transform_kernel<E, I1, I2, O, C>,
    first1, first2, count, output, c
  );
}

}  // end of namespace tf -----------------------------------------------------






