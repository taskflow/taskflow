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
template <size_t nt, size_t vt, typename I, typename O, typename C>
__global__ void cuda_transform_kernel(I first, unsigned count, O output, C op) {
  auto tid = threadIdx.x;
  auto bid = blockIdx.x;
  auto tile = cuda_get_tile(bid, nt*vt, count);
  cuda_strided_iterate<nt, vt>(
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
template <size_t nt, size_t vt, typename I1, typename I2, typename O, typename C>
__global__ void cuda_transform_kernel(
  I1 first1, I2 first2, unsigned count, O output, C op
) {
  auto tid = threadIdx.x;
  auto bid = blockIdx.x;
  auto tile = cuda_get_tile(bid, nt*vt, count);
  cuda_strided_iterate<nt, vt>(
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
// CUDA standard algorithms: transform
// ----------------------------------------------------------------------------

/**
@brief performs asynchronous parallel transforms over a range of items

@tparam P execution policy type
@tparam I input iterator type
@tparam O output iterator type
@tparam C unary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param output iterator to the beginning of the output range
@param op unary operator to apply to transform each item

This method is equivalent to the parallel execution of the following loop on a GPU:

@code{.cpp}
while (first != last) {
  *output++ = op(*first++);
}
@endcode

*/
template <typename P, typename I, typename O, typename C>
void cuda_transform(P&& p, I first, I last, O output, C op) {
  
  using E = std::decay_t<P>;

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }

  detail::cuda_transform_kernel<E::nt, E::vt, I, O, C>
    <<<E::num_blocks(count), E::nt, 0, p.stream()>>> (
    first, count, output, op
  );
}

/**
@brief performs asynchronous parallel transforms over two ranges of items

@tparam P execution policy type
@tparam I1 first input iterator type
@tparam I2 second input iterator type
@tparam O output iterator type
@tparam C binary operator type

@param p execution policy
@param first1 iterator to the beginning of the first range
@param last1 iterator to the end of the first range
@param first2 iterator to the beginning of the second range
@param output iterator to the beginning of the output range
@param op binary operator to apply to transform each pair of items

This method is equivalent to the parallel execution of the following loop on a GPU:

@code{.cpp}
while (first1 != last1) {
  *output++ = op(*first1++, *first2++);
}
@endcode
*/
template <typename P, typename I1, typename I2, typename O, typename C>
void cuda_transform(
  P&& p, I1 first1, I1 last1, I2 first2, O output, C op
) {
  
  using E = std::decay_t<P>;

  unsigned count = std::distance(first1, last1);

  if(count == 0) {
    return;
  }

  detail::cuda_transform_kernel<E::nt, E::vt, I1, I2, O, C>
    <<<E::num_blocks(count), E::nt, 0, p.stream()>>> (
    first1, first2, count, output, op
  );
}

// ----------------------------------------------------------------------------
// cudaFlow
// ----------------------------------------------------------------------------

// Function: transform
template <typename I, typename O, typename C>
cudaTask cudaFlow::transform(I first, I last, O output, C c) {
  
  using E = cudaDefaultExecutionPolicy;

  unsigned count = std::distance(first, last);
  
  // TODO:
  //if(count == 0) {
  //  return;
  //}

  return kernel(
    E::num_blocks(count), E::nt, 0,
    detail::cuda_transform_kernel<E::nt, E::vt, I, O, C>,
    first, count, output, c
  );
}

// Function: transform
template <typename I1, typename I2, typename O, typename C>
cudaTask cudaFlow::transform(I1 first1, I1 last1, I2 first2, O output, C c) {
  
  using E = cudaDefaultExecutionPolicy;

  unsigned count = std::distance(first1, last1);
  
  // TODO:
  //if(count == 0) {
  //  return;
  //}

  return kernel(
    E::num_blocks(count), E::nt, 0,
    detail::cuda_transform_kernel<E::nt, E::vt, I1, I2, O, C>,
    first1, first2, count, output, c
  );
}

// Function: update transform
template <typename I, typename O, typename C>
void cudaFlow::transform(cudaTask task, I first, I last, O output, C c) {
  
  using E = cudaDefaultExecutionPolicy;

  unsigned count = std::distance(first, last);
  
  // TODO:
  //if(count == 0) {
  //  return;
  //}

  kernel(task,
    E::num_blocks(count), E::nt, 0,
    detail::cuda_transform_kernel<E::nt, E::vt, I, O, C>,
    first, count, output, c
  );
}

// Function: update transform
template <typename I1, typename I2, typename O, typename C>
void cudaFlow::transform(
  cudaTask task, I1 first1, I1 last1, I2 first2, O output, C c
) {
  using E = cudaDefaultExecutionPolicy;

  unsigned count = std::distance(first1, last1);
  
  // TODO:
  //if(count == 0) {
  //  return;
  //}

  kernel(task,
    E::num_blocks(count), E::nt, 0,
    detail::cuda_transform_kernel<E::nt, E::vt, I1, I2, O, C>,
    first1, first2, count, output, c
  );
}

// ----------------------------------------------------------------------------
// cudaFlowCapturer
// ----------------------------------------------------------------------------

// Function: transform
template <typename I, typename O, typename C>
cudaTask cudaFlowCapturer::transform(I first, I last, O output, C op) {
  return on([=](cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_transform(p, first, last, output, op);
  });
}

// Function: transform
template <typename I1, typename I2, typename O, typename C>
cudaTask cudaFlowCapturer::transform(
  I1 first1, I1 last1, I2 first2, O output, C op
) {
  return on([=](cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_transform(p, first1, last1, first2, output, op);
  });
}

// Function: transform
template <typename I, typename O, typename C>
void cudaFlowCapturer::transform(
  cudaTask task, I first, I last, O output, C op
) {
  on(task, [=] (cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_transform(p, first, last, output, op);
  });
}

// Function: transform
template <typename I1, typename I2, typename O, typename C>
void cudaFlowCapturer::transform(
  cudaTask task, I1 first1, I1 last1, I2 first2, O output, C op
) {
  on(task, [=] (cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_transform(p, first1, last1, first2, output, op);
  });
}

}  // end of namespace tf -----------------------------------------------------






