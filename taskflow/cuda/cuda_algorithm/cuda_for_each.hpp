#pragma once

#include "../cuda_flow.hpp"
#include "../cuda_capturer.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// single_task
// ----------------------------------------------------------------------------

// Kernel: single_task
template <typename C>
__global__ void cuda_single_task(C callable) {
  callable();
}

// ----------------------------------------------------------------------------
// for_each
// ----------------------------------------------------------------------------

// Kernel: for_each
template <typename I, typename F>
__global__ void cuda_for_each(I first, size_t N, F op) {
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    op(*(first+i));
  }
}

// ----------------------------------------------------------------------------
// for_each_index
// ----------------------------------------------------------------------------

// Kernel: for_each_index
template <typename I, typename F>
__global__ void cuda_for_each_index(I beg, I inc, size_t N, F op) {
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    op(static_cast<I>(i)*inc + beg);
  }
}

// ----------------------------------------------------------------------------
// cudaFlow
// ----------------------------------------------------------------------------

// Function: single_task
template <typename C>
cudaTask cudaFlow::single_task(C&& c) {
  return kernel(
    1, 1, 0, cuda_single_task<C>, std::forward<C>(c)
  );
}

// Procedure: update_for_each
template <typename I, typename C>
void cudaFlow::update_for_each(
  cudaTask task, I first, I last, C&& callable
) {
  // TODO: special case when N is 0?
  size_t N = std::distance(first, last);
  size_t B = _default_block_size(N);

  update_kernel(
    task, (N+B-1) / B, B, 0, first, N, std::forward<C>(callable)
  );
}

// Procedure: update_for_each_index
template <typename I, typename C>
void cudaFlow::update_for_each_index(
  cudaTask task, I beg, I end, I inc, C&& c
) {

  if(is_range_invalid(beg, end, inc)) {
    TF_THROW("invalid range [", beg, ", ", end, ") with inc size ", inc);
  }
  
  // TODO: special case when N is 0?
  size_t N = distance(beg, end, inc);
  size_t B = _default_block_size(N);

  update_kernel(
    task, (N+B-1) / B, B, 0, beg, inc, N, std::forward<C>(c)
  );
}

// Function: for_each
template <typename I, typename C>
cudaTask cudaFlow::for_each(I first, I last, C&& c) {
  
  size_t N = std::distance(first, last);
  size_t B = _default_block_size(N);
  
  // TODO: special case when N is 0?

  return kernel(
    (N+B-1) / B, B, 0, cuda_for_each<I, C>, first, N, std::forward<C>(c)
  );
}

// Function: for_each_index
template <typename I, typename C>
cudaTask cudaFlow::for_each_index(I beg, I end, I inc, C&& c) {
      
  if(is_range_invalid(beg, end, inc)) {
    TF_THROW("invalid range [", beg, ", ", end, ") with inc size ", inc);
  }
  
  // TODO: special case when N is 0?

  size_t N = distance(beg, end, inc);
  size_t B = _default_block_size(N);

  return kernel(
    (N+B-1) / B, B, 0, cuda_for_each_index<I, C>, beg, inc, N, std::forward<C>(c)
  );
}

// ----------------------------------------------------------------------------
// cudaFlowCapturer
// ----------------------------------------------------------------------------

// Function: for_each
template <typename I, typename C>
cudaTask cudaFlowCapturer::for_each(I first, I last, C&& c) {
  
  // TODO: special case for N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_block_size(N);

  return on([=, c=std::forward<C>(c)](cudaStream_t stream) mutable {
    cuda_for_each<I, C><<<(N+B-1)/B, B, 0, stream>>>(first, N, c);
  });
}

// Function: for_each_index
template <typename I, typename C>
cudaTask cudaFlowCapturer::for_each_index(I beg, I end, I inc, C&& c) {
      
  if(is_range_invalid(beg, end, inc)) {
    TF_THROW("invalid range [", beg, ", ", end, ") with inc size ", inc);
  }
    
  // TODO: special case when N is 0?
  size_t N = distance(beg, end, inc);
  size_t B = _default_block_size(N);
  
  return on([=, c=std::forward<C>(c)] (cudaStream_t stream) mutable {
    cuda_for_each_index<I, C><<<(N+B-1)/B, B, 0, stream>>>(beg, inc, N, c);
  });
}

// Function: rebind_for_each
template <typename I, typename C>
void cudaFlowCapturer::rebind_for_each(cudaTask task, I first, I last, C&& c) {
  
  // TODO: special case for N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_block_size(N);

  rebind_on(task, [=, c=std::forward<C>(c)](cudaStream_t stream) mutable {
    cuda_for_each<I, C><<<(N+B-1)/B, B, 0, stream>>>(first, N, c);
  });
}

// Function: rebind_for_each_index
template <typename I, typename C>
void cudaFlowCapturer::rebind_for_each_index(
  cudaTask task, I beg, I end, I inc, C&& c
) {
      
  if(is_range_invalid(beg, end, inc)) {
    TF_THROW("invalid range [", beg, ", ", end, ") with inc size ", inc);
  }
    
  // TODO: special case when N is 0?
  size_t N = distance(beg, end, inc);
  size_t B = _default_block_size(N);
  
  rebind_on(task, [=, c=std::forward<C>(c)] (cudaStream_t stream) mutable {
    cuda_for_each_index<I, C><<<(N+B-1)/B, B, 0, stream>>>(beg, inc, N, c);
  });
}

// Function: single_task
template <typename C>
cudaTask cudaFlowCapturer::single_task(C&& callable) {
  return on([c=std::forward<C>(callable)] (cudaStream_t stream) mutable {
    cuda_single_task<C><<<1, 1, 0, stream>>>(c);
  });
}

// Function: rebind_single_task
template <typename C>
void cudaFlowCapturer::rebind_single_task(cudaTask task, C&& callable) {
  rebind_on(task, [c=std::forward<C>(callable)] (cudaStream_t stream) mutable {
    cuda_single_task<C><<<1, 1, 0, stream>>>(c);
  });
}

}  // end of namespace tf -----------------------------------------------------






