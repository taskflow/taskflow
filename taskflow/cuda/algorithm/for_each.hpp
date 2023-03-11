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
template <size_t nt, size_t vt, typename I, typename C>
__global__ void cuda_for_each_kernel(I first, unsigned count, C c) {
  auto tid = threadIdx.x;
  auto bid = blockIdx.x;
  auto tile = cuda_get_tile(bid, nt*vt, count);
  cuda_strided_iterate<nt, vt>(
    [=](auto, auto j) {
      c(*(first + tile.begin + j));
    }, 
    tid, tile.count()
  );
}

/** @private */
template <size_t nt, size_t vt, typename I, typename C>
__global__ void cuda_for_each_index_kernel(I first, I inc, unsigned count, C c) {
  auto tid = threadIdx.x;
  auto bid = blockIdx.x;
  auto tile = cuda_get_tile(bid, nt*vt, count);
  cuda_strided_iterate<nt, vt>(
    [=]__device__(auto, auto j) {
      c(first + inc*(tile.begin+j));
    }, 
    tid, tile.count()
  );
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

The function launches a single kernel thread to run the given callable
through the stream in the execution policy object.
*/
template <typename P, typename C>
void cuda_single_task(P&& p, C c) {
  cuda_kernel<<<1, 1, 0, p.stream()>>>(
    [=]__device__(auto, auto) mutable { c(); }
  );
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

This function is equivalent to a parallel execution of the following loop
on a GPU:

@code{.cpp}
for(auto itr = first; itr != last; itr++) {
  c(*itr);
}
@endcode
*/
template <typename P, typename I, typename C>
void cuda_for_each(P&& p, I first, I last, C c) {
  
  using E = std::decay_t<P>;

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }

  detail::cuda_for_each_kernel<E::nt, E::vt, I, C><<<E::num_blocks(count), E::nt, 0, p.stream()>>>(
    first, count, c
  );
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

This function is equivalent to a parallel execution of
the following loop on a GPU:

@code{.cpp}
// step is positive [first, last)
for(auto i=first; i<last; i+=step) {
  c(i);
}

// step is negative [first, last)
for(auto i=first; i>last; i+=step) {
  c(i);
}
@endcode
*/
template <typename P, typename I, typename C>
void cuda_for_each_index(P&& p, I first, I last, I inc, C c) {
  
  using E = std::decay_t<P>;

  unsigned count = distance(first, last, inc);

  if(count == 0) {
    return;
  }

  detail::cuda_for_each_index_kernel<E::nt, E::vt, I, C><<<E::num_blocks(count), E::nt, 0, p.stream()>>>(
    first, inc, count, c
  );
}

// ----------------------------------------------------------------------------
// single_task
// ----------------------------------------------------------------------------

/** @private */
template <typename C>
__global__ void cuda_single_task(C callable) {
  callable();
}

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

// Function: single_task
template <typename C>
cudaTask cudaFlowCapturer::single_task(C callable) {
  return on([=] (cudaStream_t stream) mutable {
    cuda_single_task(cudaDefaultExecutionPolicy(stream), callable);
  });
}

// Function: single_task
template <typename C>
void cudaFlowCapturer::single_task(cudaTask task, C callable) {
  on(task, [=] (cudaStream_t stream) mutable {
    cuda_single_task(cudaDefaultExecutionPolicy(stream), callable);
  });
}

// ----------------------------------------------------------------------------
// cudaFlow: for_each, for_each_index
// ----------------------------------------------------------------------------

// Function: for_each
template <typename I, typename C>
cudaTask cudaFlow::for_each(I first, I last, C c) {

  using E = cudaDefaultExecutionPolicy;
  
  unsigned count = std::distance(first, last);
  
  // TODO:
  //if(count == 0) {
  //  return;
  //}

  return kernel(
    E::num_blocks(count), E::nt, 0, 
    detail::cuda_for_each_kernel<E::nt, E::vt, I, C>, first, count, c
  );
}

// Function: for_each
template <typename I, typename C>
void cudaFlow::for_each(cudaTask task, I first, I last, C c) {

  using E = cudaDefaultExecutionPolicy;
  
  unsigned count = std::distance(first, last);

  // TODO:
  //if(count == 0) {
  //  return;
  //}
  
  kernel(task, 
    E::num_blocks(count), E::nt, 0, 
    detail::cuda_for_each_kernel<E::nt, E::vt, I, C>, first, count, c
  );
}

// Function: for_each_index
template <typename I, typename C>
cudaTask cudaFlow::for_each_index(I first, I last, I inc, C c) {

  using E = cudaDefaultExecutionPolicy;

  unsigned count = distance(first, last, inc);

  // TODO:
  //if(count == 0) {
  //  return;
  //}

  return kernel(
    E::num_blocks(count), E::nt, 0, 
    detail::cuda_for_each_index_kernel<E::nt, E::vt, I, C>, first, inc, count, c
  );
}

// Function: for_each_index
template <typename I, typename C>
void cudaFlow::for_each_index(cudaTask task, I first, I last, I inc, C c) {
  
  using E = cudaDefaultExecutionPolicy;

  unsigned count = distance(first, last, inc);
  
  // TODO:
  //if(count == 0) {
  //  return;
  //}

  return kernel(task,
    E::num_blocks(count), E::nt, 0, 
    detail::cuda_for_each_index_kernel<E::nt, E::vt, I, C>, first, inc, count, c
  );
}

// ----------------------------------------------------------------------------
// cudaFlowCapturer: for_each, for_each_index
// ----------------------------------------------------------------------------

// Function: for_each
template <typename I, typename C>
cudaTask cudaFlowCapturer::for_each(I first, I last, C c) {
  return on([=](cudaStream_t stream) mutable {
    cuda_for_each(cudaDefaultExecutionPolicy(stream), first, last, c);
  });
}

// Function: for_each_index
template <typename I, typename C>
cudaTask cudaFlowCapturer::for_each_index(I beg, I end, I inc, C c) {
  return on([=] (cudaStream_t stream) mutable {
    cuda_for_each_index(cudaDefaultExecutionPolicy(stream), beg, end, inc, c);
  });
}

// Function: for_each
template <typename I, typename C>
void cudaFlowCapturer::for_each(cudaTask task, I first, I last, C c) {
  on(task, [=](cudaStream_t stream) mutable {
    cuda_for_each(cudaDefaultExecutionPolicy(stream), first, last, c);
  });
}

// Function: for_each_index
template <typename I, typename C>
void cudaFlowCapturer::for_each_index(
  cudaTask task, I beg, I end, I inc, C c
) {
  on(task, [=] (cudaStream_t stream) mutable {
    cuda_for_each_index(cudaDefaultExecutionPolicy(stream), beg, end, inc, c);
  });
}



}  // end of namespace tf -----------------------------------------------------






