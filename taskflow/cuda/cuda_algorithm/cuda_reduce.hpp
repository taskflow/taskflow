#pragma once

#include "../cuda_flow.hpp"
#include "../cuda_capturer.hpp"
#include "../cuda_meta.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// reduction helper functions
// ----------------------------------------------------------------------------

namespace detail {

/** @private */
template<unsigned nt, typename T>
struct cudaBlockReduce {

  static const unsigned group_size = std::min(nt, CUDA_WARP_SIZE);
  static const unsigned num_passes = log2(group_size);
  static const unsigned num_items = nt / group_size;

  static_assert(
    nt && (0 == nt % CUDA_WARP_SIZE), 
    "cudaBlockReduce requires num threads to be a multiple of warp_size (32)"
  );

  struct Storage {
    T data[std::max(nt, 2 * group_size)];
  };

  template<typename op_t>
  __device__ T operator()(unsigned, T, Storage&, unsigned, op_t, bool = true) const;
};

// function: reduce to be called from a block
template<unsigned nt, typename T>
template<typename op_t>
__device__ T cudaBlockReduce<nt, T>::operator ()(
  unsigned tid, T x, Storage& storage, unsigned count, op_t op, bool ret
) const {

  // Store your data into shared memory.
  storage.data[tid] = x;
  __syncthreads();

  if(tid < group_size) {
    // Each thread scans within its lane.
    cuda_strided_iterate<group_size, num_items>([&](auto i, auto j) {
      if(i > 0) {
        x = op(x, storage.data[j]);
      }
    }, tid, count);
    storage.data[tid] = x;
  }
  __syncthreads();

  auto count2 = count < group_size ? count : group_size; 
  auto first = (1 & num_passes) ? group_size : 0;
  if(tid < group_size) {
    storage.data[first + tid] = x;
  }
  __syncthreads();

  cuda_iterate<num_passes>([&](auto pass) {
    if(tid < group_size) {
      if(auto offset = 1 << pass; tid + offset < count2) {
        x = op(x, storage.data[first + offset + tid]);
      }
      first = group_size - first;
      storage.data[first + tid] = x;
    }
    __syncthreads();
  });

  if(ret) {
    x = storage.data[0];
    __syncthreads();
  }
  return x;
}

/** @private */
template <typename P, typename I, typename T, typename O>
void cuda_reduce_loop(
  P&&, I input, unsigned count, T* res, O op, bool incl, cudaStream_t S, T* buf
) {

  using E = std::decay_t<P>;
  
  auto B = (count + E::nv - 1) / E::nv;
  
  do {
    cuda_kernel<<<B, E::nt, 0, S>>>([=] __device__ (auto tid, auto bid) {
      __shared__ typename cudaBlockReduce<E::nt, T>::Storage shm;
      auto tile = cuda_get_tile(bid, E::nv, count);
      auto x = cuda_mem_to_reg_strided<E::nt, E::vt>(
        input + tile.begin, tid, tile.count()
      );
      // Reduce the multiple values per thread into a scalar.
      T s;
      cuda_strided_iterate<E::nt, E::vt>(
        [&] (auto i, auto) { s = i ? op(s, x[i]) : x[0]; }, tid, tile.count()
      );
      // reduce to a scalar per block.
      s = cudaBlockReduce<E::nt, T>()(
        tid, s, shm, (tile.count() < E::nt ? tile.count() : E::nt), op, false
      );
      if(!tid) {
        (1 == B) ? *res = (incl ? op(*res, s) : s) : buf[bid] = s;
      }
    });

    if(B == 1) {
      break;
    }
    
    // move on to the next block
    input = buf;
    buf += B;
    count = B;
    B = (B + E::nv - 1) / E::nv;

  } while(true);

}

/** @private */
template <typename P, typename I, typename T, typename O, typename U>
void cuda_transform_reduce_loop(
  P&& p, I input, unsigned count, T* res, O bop, U uop, bool incl, cudaStream_t S, T* buf
) {

  using E = std::decay_t<P>;
  
  auto B = (count + E::nv - 1) / E::nv;

  // first round will take care of transform
  cuda_kernel<<<B, E::nt, 0, S>>>([=] __device__ (auto tid, auto bid) {
    __shared__ typename cudaBlockReduce<E::nt, T>::Storage shm;
    auto tile = cuda_get_tile(bid, E::nv, count);
    auto x = cuda_transform_mem_to_reg_strided<E::nt, E::vt>(
      input + tile.begin, tid, tile.count(), uop
    );
    // Reduce the multiple values per thread into a scalar.
    T s;
    cuda_strided_iterate<E::nt, E::vt>(
      [&] (auto i, auto) { s = i ? bop(s, x[i]) : x[0]; }, tid, tile.count()
    );
    // reduce to a scalar per block.
    s = cudaBlockReduce<E::nt, T>()(
      tid, s, shm, (tile.count() < E::nt ? tile.count() : E::nt), bop, false
    );
    if(!tid) {
      (1 == B) ? *res = (incl ? bop(*res, s) : s) : buf[bid] = s;
    }
  });

  if(B == 1) return;
    
  cuda_reduce_loop(p, buf, B, res, bop, incl, S, buf+B);
}

}  // namespace tf::detail ----------------------------------------------------

template<typename T>
struct cuda_plus : public std::binary_function<T, T, T> {
  __device__ T operator()(T a, T b) const { return a + b; }
};

template<typename T>
struct cuda_minus : public std::binary_function<T, T, T> {
  __device__ T operator()(T a, T b) const { return a - b; }
};

template<typename T>
struct cuda_multiplies : public std::binary_function<T, T, T> {
  __device__ T operator()(T a, T b) const { return a * b; }
};

// cuda_reduce_buffer_size
template <typename P>
unsigned cuda_reduce_buffer_size(P&&, unsigned count) {
  using E = std::decay_t<P>;
  unsigned B = (count + E::nv - 1) / E::nv;
  unsigned buffer_sz = 0;
  for(auto b=B; b>1; buffer_sz += (b=(b+E::nv-1)/E::nv));
  return buffer_sz;
}

// cuda_reduce
template<typename P, typename I, typename T, typename O>
void cuda_reduce(P&& p, I first, I last, T* res, O op) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }
  
  // allocate temporary buffer
  cudaDeviceMemory<T> temp(cuda_reduce_buffer_size(p, count));
  auto buf = temp.data();
  
  // reduction loop
  detail::cuda_reduce_loop(p, first, count, res, op, true, 0, buf);
  
  // synchronize the execution
  TF_CHECK_CUDA(cudaStreamSynchronize(0), "failed to sync reduce on stream 0");
}

// cuda_reduce
template <typename I, typename T, typename O>
void cuda_reduce(I first, I last, T* res, O op) {
  cuda_reduce(cudaDefaultExecutionPolicy{}, first, last, res, op);
}

// cuda_reduce_async
template <typename P, typename I, typename T, typename O>
void cuda_reduce_async(
  P&& p, I first, I last, T* res, O op, cudaStream_t s, T* buf
) {
  unsigned count = std::distance(first, last);
  if(count == 0) {
    return;
  }
  detail::cuda_reduce_loop<P>(p, first, count, res, op, true, s, buf);
}

// cuda_reduce_async
template <typename I, typename T, typename O>
void cuda_reduce_async(
  I first, I last, T* res, O op, cudaStream_t s, T* buf
) {
  cuda_reduce_async(
    cudaDefaultExecutionPolicy{}, first, last, res, op, s, buf
  );
}

// cuda_uninitialized_reduce
template<typename P, typename I, typename T, typename O>
void cuda_uninitialized_reduce(P&& p, I first, I last, T* res, O op) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }
  
  // allocate temporary buffer
  cudaDeviceMemory<T> temp(cuda_reduce_buffer_size(p, count));
  auto buf = temp.data();
  
  // reduction loop
  detail::cuda_reduce_loop(p, first, count, res, op, false, 0, buf);
  
  // synchronize the execution
  TF_CHECK_CUDA(cudaStreamSynchronize(0), "uninitialized_reduce failed to sync stream 0");
}

// cuda_uninitialized_reduce
template <typename I, typename T, typename O>
void cuda_uninitialized_reduce(I first, I last, T* res, O op) {
  cuda_uninitialized_reduce(cudaDefaultExecutionPolicy{}, first, last, res, op);
}

// cuda_uninitialized_reduce_async
template <typename P, typename I, typename T, typename O>
void cuda_uninitialized_reduce_async(
  P&& p, I first, I last, T* res, O op, cudaStream_t s, T* buf
) {
  unsigned count = std::distance(first, last);
  if(count == 0) {
    return;
  }
  detail::cuda_reduce_loop(p, first, count, res, op, false, s, buf);
}

// cuda_uninitialized_reduce_async
template <typename I, typename T, typename O>
void cuda_uninitialized_reduce_async(
  I first, I last, T* res, O op, cudaStream_t s, T* buf
) {
  cuda_uninitialized_reduce_async(
    cudaDefaultExecutionPolicy{}, first, last, res, op, s, buf
  );
}

// ----------------------------------------------------------------------------
// transform_reduce
// ----------------------------------------------------------------------------

// cuda_transform_reduce
template<typename P, typename I, typename T, typename O, typename U>
void cuda_transform_reduce(P&& p, I first, I last, T* res, O bop, U uop) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }
  
  // allocate temporary buffer
  cudaDeviceMemory<T> temp(cuda_reduce_buffer_size(p, count));
  auto buf = temp.data();
  
  // reduction loop
  detail::cuda_transform_reduce_loop(p, first, count, res, bop, uop, true, 0, buf);
  
  // synchronize the execution
  TF_CHECK_CUDA(cudaStreamSynchronize(0), "failed to sync reduce on stream 0");
}

// cuda_reduce
template <typename I, typename T, typename O, typename U>
void cuda_transform_reduce(I first, I last, T* res, O bop, U uop) {
  cuda_transform_reduce(
    cudaDefaultExecutionPolicy{}, first, last, res, bop, uop
  );
}

// ----------------------------------------------------------------------------

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

