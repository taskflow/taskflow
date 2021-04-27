#pragma once

#include "../cuda_flow.hpp"
#include "../cuda_capturer.hpp"

namespace tf {

template <typename T, typename C>
__device__ void cuda_warp_reduce(
  volatile T* shm, size_t N, size_t tid, C op
) {
  if(tid + 32 < N) shm[tid] = op(shm[tid], shm[tid+32]);
  if(tid + 16 < N) shm[tid] = op(shm[tid], shm[tid+16]);
  if(tid +  8 < N) shm[tid] = op(shm[tid], shm[tid+8]);
  if(tid +  4 < N) shm[tid] = op(shm[tid], shm[tid+4]);
  if(tid +  2 < N) shm[tid] = op(shm[tid], shm[tid+2]);
  if(tid +  1 < N) shm[tid] = op(shm[tid], shm[tid+1]);
}

// Kernel: cuda_reduce
// This reduction kernel assums only one block to avoid extra output memory.
template <typename I, typename T, typename C, bool uninitialized>
__global__ void cuda_reduce(I first, size_t N, T* res, C op) {

  size_t tid = threadIdx.x;

  if(tid >= N) {
    return;
  }
  
  cudaSharedMemory<T> shared_memory;
  T* shm = shared_memory.get();

  shm[tid] = *(first+tid);
  
  for(size_t i=tid+blockDim.x; i<N; i+=blockDim.x) {
    shm[tid] = op(shm[tid], *(first+i));
  }

  __syncthreads();

  for(size_t s = blockDim.x / 2; s > 32; s >>= 1) {
    if(tid < s && tid + s < N) {
      shm[tid] = op(shm[tid], shm[tid+s]);
    }
    __syncthreads();
  }

  if(tid < 32) {
    cuda_warp_reduce(shm, N, tid, op);
  }
  
  if(tid == 0) {
    if constexpr (uninitialized) {
      *res = shm[0];
    }
    else {
      *res = op(*res, shm[0]);
    }
  }
}

// ----------------------------------------------------------------------------
// cudaFlow 
// ----------------------------------------------------------------------------

// Function: reduce
template <typename I, typename T, typename C>
cudaTask cudaFlow::reduce(I first, I last, T* result, C&& op) {
  
  // TODO: special case N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_block_size(N);
  
  return kernel(
    1, B, B*sizeof(T), cuda_reduce<I, T, C, false>, 
    first, N, result, std::forward<C>(op)
  );
}

// Function: uninitialized_reduce
template <typename I, typename T, typename C>
cudaTask cudaFlow::uninitialized_reduce(I first, I last, T* result, C&& op) {
  // TODO: special case N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_block_size(N);
  
  return kernel(
    1, B, B*sizeof(T), cuda_reduce<I, T, C, true>, 
    first, N, result, std::forward<C>(op)
  );
}

// Procedure: update_reduce
template <typename I, typename T, typename C>
void cudaFlow::update_reduce(
  cudaTask task, I first, I last, T* result, C&& op
) {

  // TODO: special case N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_block_size(N);
  
  update_kernel(
    task, 1, B, B*sizeof(T), first, N, result, std::forward<C>(op)
  );
}

// Procedure: update_uninitialized_reduce
template <typename I, typename T, typename C>
void cudaFlow::update_uninitialized_reduce(
  cudaTask task, I first, I last, T* result, C&& op
) {
  // TODO: special case N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_block_size(N);
  
  update_kernel(
    task, 1, B, B*sizeof(T), first, N, result, std::forward<C>(op)
  );
}

// ----------------------------------------------------------------------------
// cudaFlowCapturer
// ----------------------------------------------------------------------------

// Function: reduce
template <typename I, typename T, typename C>
cudaTask cudaFlowCapturer::reduce(I first, I last, T* result, C&& c) {
    
  // TODO: special case N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_block_size(N);
  
  return on([=, c=std::forward<C>(c)] 
  (cudaStream_t stream) mutable {
    cuda_reduce<I, T, C, false><<<1, B, B*sizeof(T), stream>>>(
      first, N, result, c
    );
  });
}

// Function: uninitialized_reduce
template <typename I, typename T, typename C>
cudaTask cudaFlowCapturer::uninitialized_reduce(
  I first, I last, T* result, C&& c
) {
    
  // TODO: special case N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_block_size(N);
  
  return on([=, c=std::forward<C>(c)] 
  (cudaStream_t stream) mutable {
    cuda_reduce<I, T, C, true><<<1, B, B*sizeof(T), stream>>>(
      first, N, result, c
    );
  });
}

// Function: rebind_reduce
template <typename I, typename T, typename C>
void cudaFlowCapturer::rebind_reduce(
  cudaTask task, I first, I last, T* result, C&& c
) {
    
  // TODO: special case N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_block_size(N);
  
  rebind_on(task, [=, c=std::forward<C>(c)] 
  (cudaStream_t stream) mutable {
    cuda_reduce<I, T, C, false><<<1, B, B*sizeof(T), stream>>>(
      first, N, result, c
    );
  });
}

// Function: rebind_uninitialized_reduce
template <typename I, typename T, typename C>
void cudaFlowCapturer::rebind_uninitialized_reduce(
  cudaTask task, I first, I last, T* result, C&& c
) {
    
  // TODO: special case N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_block_size(N);
  
  rebind_on(task, [=, c=std::forward<C>(c)] 
  (cudaStream_t stream) mutable {
    cuda_reduce<I, T, C, true><<<1, B, B*sizeof(T), stream>>>(
      first, N, result, c
    );
  });
}

}  // end of namespace tf -----------------------------------------------------

