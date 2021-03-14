#pragma once

#include "../cuda_flow.hpp"
#include "../cuda_capturer.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// transform
// ----------------------------------------------------------------------------

// Kernel: for_each
template <typename I, typename F, typename... S>
__global__ void cuda_transform(I first, size_t N, F op, S... srcs) {
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    //data[i] = op(src[i]...);
    *(first + i) = op((*(srcs+i))...);
  }
}

// ----------------------------------------------------------------------------
// cudaFlow
// ----------------------------------------------------------------------------

// Function: transform
template <typename I, typename C, typename... S>
cudaTask cudaFlow::transform(I first, I last, C&& c, S... srcs) {
  
  // TODO: special case when N is 0?
  
  size_t N = std::distance(first, last);
  size_t B = _default_block_size(N);

  return kernel(
    (N+B-1) / B, B, 0, cuda_transform<I, C, S...>, 
    first, N, std::forward<C>(c), srcs...
  );
}

// Procedure: update_transform
template <typename I, typename C, typename... S>
void cudaFlow::update_transform(
  cudaTask task, I first, I last, C&& c, S... srcs
) {
  
  // TODO: special case when N is 0?
  size_t N = std::distance(first, last);
  size_t B = _default_block_size(N);

  update_kernel(
    task, (N+B-1) / B, B, 0, first, N, std::forward<C>(c), srcs...
  );
}

// ----------------------------------------------------------------------------
// cudaFlowCapturer
// ----------------------------------------------------------------------------

// Function: transform
template <typename I, typename C, typename... S>
cudaTask cudaFlowCapturer::transform(I first, I last, C&& c, S... srcs) {
  
  // TODO: special case when N is 0?
  size_t N = std::distance(first, last);
  size_t B = _default_block_size(N);

  return on([=, c=std::forward<C>(c)] 
  (cudaStream_t stream) mutable {
    cuda_transform<I, C, S...><<<(N+B-1)/B, B, 0, stream>>>(first, N, c, srcs...);
  });
}

// Function: rebind_transform
template <typename I, typename C, typename... S>
void cudaFlowCapturer::rebind_transform(
  cudaTask task, I first, I last, C&& c, S... srcs
) {
  
  // TODO: special case when N is 0?
  size_t N = std::distance(first, last);
  size_t B = _default_block_size(N);

  rebind_on(task, [=, c=std::forward<C>(c)] 
  (cudaStream_t stream) mutable {
    cuda_transform<I, C, S...><<<(N+B-1)/B, B, 0, stream>>>(first, N, c, srcs...);
  });
}

}  // end of namespace tf -----------------------------------------------------






