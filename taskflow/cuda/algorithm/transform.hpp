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

/** @private */
template <typename P, typename I, typename O, typename C>
void cuda_transform_loop(P&& p, I first, unsigned count, O output, C op) {

  using E = std::decay_t<P>;

  unsigned B = (count + E::nv - 1) / E::nv;

  cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=]__device__(auto tid, auto bid) {
    auto tile = cuda_get_tile(bid, E::nv, count);
    cuda_strided_iterate<E::nt, E::vt>([=]__device__(auto, auto j) {
      auto offset = j + tile.begin;
      *(output + offset) = op(*(first+offset));
    }, tid, tile.count());
  });
}

/** @private */
template <typename P, typename I1, typename I2, typename O, typename C>
void cuda_transform_loop(
  P&& p, I1 first1, I2 first2, unsigned count, O output, C op
) {

  using E = std::decay_t<P>;

  unsigned B = (count + E::nv - 1) / E::nv;

  cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=]__device__(auto tid, auto bid) {
    auto tile = cuda_get_tile(bid, E::nv, count);
    cuda_strided_iterate<E::nt, E::vt>([=]__device__(auto, auto j) {
      auto offset = j + tile.begin;
      *(output + offset) = op(*(first1+offset), *(first2+offset));
    }, tid, tile.count());
  });
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

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }

  detail::cuda_transform_loop(p, first, count, output, op);
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

  unsigned count = std::distance(first1, last1);

  if(count == 0) {
    return;
  }

  detail::cuda_transform_loop(p, first1, first2, count, output, op);
}

// ----------------------------------------------------------------------------
// cudaFlow
// ----------------------------------------------------------------------------

// Function: transform
template <typename I, typename O, typename C>
cudaTask cudaFlow::transform(I first, I last, O output, C c) {
  return capture([=](cudaFlowCapturer& cap) mutable {
    cap.make_optimizer<cudaLinearCapturing>();
    cap.transform(first, last, output, c);
  });
}

// Function: transform
template <typename I1, typename I2, typename O, typename C>
cudaTask cudaFlow::transform(I1 first1, I1 last1, I2 first2, O output, C c) {
  return capture([=](cudaFlowCapturer& cap) mutable {
    cap.make_optimizer<cudaLinearCapturing>();
    cap.transform(first1, last1, first2, output, c);
  });
}

// Function: update transform
template <typename I, typename O, typename C>
void cudaFlow::transform(cudaTask task, I first, I last, O output, C c) {
  capture(task, [=](cudaFlowCapturer& cap) mutable {
    cap.make_optimizer<cudaLinearCapturing>();
    cap.transform(first, last, output, c);
  });
}

// Function: update transform
template <typename I1, typename I2, typename O, typename C>
void cudaFlow::transform(
  cudaTask task, I1 first1, I1 last1, I2 first2, O output, C c
) {
  capture(task, [=](cudaFlowCapturer& cap) mutable {
    cap.make_optimizer<cudaLinearCapturing>();
    cap.transform(first1, last1, first2, output, c);
  });
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






