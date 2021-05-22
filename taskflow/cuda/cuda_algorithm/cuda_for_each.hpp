#pragma once

#include "../cuda_flow.hpp"
#include "../cuda_capturer.hpp"
#include "../cuda_meta.hpp"

/** 
@file cuda_for_each.hpp
@brief cuda parallel-iteration algorithms include file
*/

namespace tf {

namespace detail {

/** @private */
template <typename P, typename I, typename C>
void cuda_for_each_loop(P&& p, I first, unsigned count, C c) {

  using E = std::decay_t<P>;

  unsigned B = (count + E::nv - 1) / E::nv;

  cuda_kernel<<<B, E::nt, 0, p.stream()>>>(
  [=] __device__ (auto tid, auto bid) {
    auto tile = cuda_get_tile(bid, E::nv, count);
    cuda_strided_iterate<E::nt, E::vt>([=](auto, auto j) {
      c(*(first + tile.begin + j));
    }, tid, tile.count());
  });
}

/** @private */
template <typename P, typename I, typename C>
void cuda_for_each_index_loop(
  P&& p, I first, I inc, unsigned count, C c
) {

  using E = std::decay_t<P>;

  unsigned B = (count + E::nv - 1) / E::nv;

  cuda_kernel<<<B, E::nt, 0, p.stream()>>>(
  [=]__device__(auto tid, auto bid) {
    auto tile = cuda_get_tile(bid, E::nv, count);
    cuda_strided_iterate<E::nt, E::vt>([=]__device__(auto, auto j) {
      c(first + inc*(tile.begin+j));
    }, tid, tile.count());
  });
}

}  // end of namespace detail -------------------------------------------------

// ----------------------------------------------------------------------------
// cuda standard algorithms: single_task/for_each/for_each_index
// ----------------------------------------------------------------------------

/**
@brief runs a callable asynchronously using one kernel thread

@tparam P execution policy type
@tparam C closure type

@param p execution policy
@param c closure to run by one kernel thread
*/
template <typename P, typename C>
void cuda_single_task_async(P&& p, C c) {
  cuda_kernel<<<1, 1, 0, p.stream()>>>(
    [=]__device__(auto, auto) mutable { c(); }
  );
}

/**
@brief runs a callable using one kernel thread

@tparam P execution policy type
@tparam C closure type

@param p execution policy
@param c closure to run by one kernel thread
*/
template <typename P, typename C>
void cuda_single_task(P&& p, C c) {
  cuda_single_task_async(p, c);
  p.synchronize();
}

/**
@brief performs asynchronous parallel iterations over a range of items

@tparam P execution policy type
@tparam I input iterator type
@tparam C unary operator type

@param p execution policy object
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param c unary operator to apply to each dereferenced iterator

Please refer to @ref CUDASTDForEach for details.
*/
template <typename P, typename I, typename C>
void cuda_for_each_async(P&& p, I first, I last, C c) {
  
  unsigned count = std::distance(first, last);
  
  if(count == 0) {
    return;
  }

  detail::cuda_for_each_loop(p, first, count, c);
}

/**
@brief performs parallel iterations over a range of items

@tparam P execution policy type
@tparam I input iterator type
@tparam C unary operator type

@param p execution policy object
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param c unary operator to apply to each dereferenced iterator

Please refer to @ref CUDASTDForEach for details.
*/
template <typename P, typename I, typename C>
void cuda_for_each(P&& p, I first, I last, C c) {
  cuda_for_each_async(p, first, last, c);
  p.synchronize();
}

/**
@brief performs asynchronous parallel iterations over 
       an index-based range of items

@tparam P execution policy type
@tparam I input index type
@tparam C unary operator type

@param p execution policy object
@param first index to the beginning of the range
@param last  index to the end of the range
@param inc step size between successive iterations
@param c unary operator to apply to each index

Please refer to @ref CUDASTDForEach for details.
*/
template <typename P, typename I, typename C>
void cuda_for_each_index_async(P&& p, I first, I last, I inc, C c) {
  
  if(is_range_invalid(first, last, inc)) {
    TF_THROW("invalid range [", first, ", ", last, ") with inc size ", inc);
  }
  
  unsigned count = distance(first, last, inc);
  
  if(count == 0) {
    return;
  }

  detail::cuda_for_each_index_loop(p, first, inc, count, c);
}

  
/**
@brief performs parallel iterations over an index-based range of items

@tparam P execution policy type
@tparam I input index type
@tparam C unary operator type

@param p execution policy object
@param first index to the beginning of the range
@param last  index to the end of the range
@param inc step size between successive iterations
@param c unary operator to apply to each index

Please refer to @ref CUDASTDForEach for details.
*/
template <typename P, typename I, typename C>
void cuda_for_each_index(P&& p, I first, I last, I inc, C c) {
  cuda_for_each_index_async(p, first, last, inc, c); 
  p.synchronize();
}


// ----------------------------------------------------------------------------
// single_task
// ----------------------------------------------------------------------------

/** @private */
template <typename C>
__global__ void cuda_single_task(C callable) {
  callable();
}

// ----------------------------------------------------------------------------
// cudaFlow
// ----------------------------------------------------------------------------

// Function: single_task
template <typename C>
cudaTask cudaFlow::single_task(C c) {
  return kernel(1, 1, 0, cuda_single_task<C>, c);
}

// Function: single_task
template <typename C>
void cudaFlow::single_task(cudaTask task, C c) {
  return kernel(task, 1, 1, 0, cuda_single_task<C>, c);
}

// Function: for_each
template <typename I, typename C>
cudaTask cudaFlow::for_each(I first, I last, C c) {
  return capture([=](cudaFlowCapturer& cap) mutable {
    cap.make_optimizer<cudaLinearCapturing>();
    cap.for_each(first, last, c);
  });
}

// Function: for_each_index
template <typename I, typename C>
cudaTask cudaFlow::for_each_index(I first, I last, I inc, C c) {
  return capture([=](cudaFlowCapturer& cap) mutable {
    cap.make_optimizer<cudaLinearCapturing>();
    cap.for_each_index(first, last, inc, c);
  });
}

// Function: for_each
template <typename I, typename C>
void cudaFlow::for_each(cudaTask task, I first, I last, C c) {
  capture(task, [=](cudaFlowCapturer& cap) mutable {
    cap.make_optimizer<cudaLinearCapturing>();
    cap.for_each(first, last, c);
  });
}

// Function: for_each_index
template <typename I, typename C>
void cudaFlow::for_each_index(cudaTask task, I first, I last, I inc, C c) {
  capture(task, [=](cudaFlowCapturer& cap) mutable {
    cap.make_optimizer<cudaLinearCapturing>();
    cap.for_each_index(first, last, inc, c);
  });
}

// ----------------------------------------------------------------------------
// cudaFlowCapturer
// ----------------------------------------------------------------------------

// Function: for_each
template <typename I, typename C>
cudaTask cudaFlowCapturer::for_each(I first, I last, C c) {
  return on([=](cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_for_each_async(p, first, last, c);
  });
}

// Function: for_each_index
template <typename I, typename C>
cudaTask cudaFlowCapturer::for_each_index(I beg, I end, I inc, C c) {
  return on([=] (cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_for_each_index_async(p, beg, end, inc, c);
  });
}

// Function: for_each
template <typename I, typename C>
void cudaFlowCapturer::for_each(cudaTask task, I first, I last, C c) {
  on(task, [=](cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_for_each_async(p, first, last, c);
  });
}

// Function: for_each_index
template <typename I, typename C>
void cudaFlowCapturer::for_each_index(
  cudaTask task, I beg, I end, I inc, C c
) {
  on(task, [=] (cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_for_each_index_async(p, beg, end, inc, c);
  });
}

// Function: single_task
template <typename C>
cudaTask cudaFlowCapturer::single_task(C callable) {
  return on([=] (cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_single_task_async(p, callable);
  });
}

// Function: single_task
template <typename C>
void cudaFlowCapturer::single_task(cudaTask task, C callable) {
  on(task, [=] (cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_single_task_async(p, callable);
  });
}

}  // end of namespace tf -----------------------------------------------------






