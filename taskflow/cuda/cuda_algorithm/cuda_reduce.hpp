#pragma once

#include "../cuda_flow.hpp"
#include "../cuda_capturer.hpp"
#include "../cuda_meta.hpp"

/** 
@file cuda_reduce.hpp
@brief cuda reduce algorithms include file
*/

namespace tf::detail {

// ----------------------------------------------------------------------------
// reduction helper functions
// ----------------------------------------------------------------------------

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
  
  /** @private */  
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
  P&& p, I input, unsigned count, T* res, O op, bool incl, void* ptr
) {

  using E = std::decay_t<P>;

  auto buf = static_cast<T*>(ptr);
  auto B = (count + E::nv - 1) / E::nv;
  
  cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
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

  if(B > 1) {
    cuda_reduce_loop(p, buf, B, res, op, incl, buf+B);
  }
}

}  // namespace tf::detail ----------------------------------------------------

namespace tf {

/**
@brief queries the buffer size in bytes needed to call reduce kernels

@tparam P execution policy type
@tparam T value type

@param count number of elements to reduce

The function is used to allocate a buffer for calling asynchronous reduce.
Please refer to @ref CUDASTDReduce for details.
*/
template <typename P, typename T>
unsigned cuda_reduce_buffer_size(unsigned count) {
  using E = std::decay_t<P>;
  unsigned B = (count + E::nv - 1) / E::nv;
  unsigned n = 0;
  for(auto b=B; b>1; n += (b=(b+E::nv-1)/E::nv));
  return n*sizeof(T);
}

// ----------------------------------------------------------------------------
// cuda_reduce
// ----------------------------------------------------------------------------

/**
@brief performs parallel reduction over a range of items

@tparam P execution policy type
@tparam I input iterator type
@tparam T value type
@tparam O binary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param res pointer to the result
@param op binary operator to apply to reduce elements

Please refer to @ref CUDASTDReduce for details.
 */
template<typename P, typename I, typename T, typename O>
void cuda_reduce(P&& p, I first, I last, T* res, O op) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }
  
  // allocate temporary buffer
  cudaDeviceMemory<std::byte> temp(cuda_reduce_buffer_size<P, T>(count));
  
  // reduction loop
  detail::cuda_reduce_loop(p, first, count, res, op, true, temp.data());
  
  // synchronize the execution
  p.synchronize();
}

/**
@brief performs asynchronous parallel reduction over a range of items

@tparam P execution policy type
@tparam I input iterator type
@tparam T value type
@tparam O binary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param res pointer to the result
@param op binary operator to apply to reduce elements
@param buf pointer to the temporary buffer

Please refer to @ref CUDASTDReduce for details.
 */
template <typename P, typename I, typename T, typename O>
void cuda_reduce_async(
  P&& p, I first, I last, T* res, O op, void* buf
) {
  unsigned count = std::distance(first, last);
  if(count == 0) {
    return;
  }
  detail::cuda_reduce_loop(p, first, count, res, op, true, buf);
}

// ----------------------------------------------------------------------------
// cuda_uninitialized_reduce
// ----------------------------------------------------------------------------

/**
@brief performs parallel reduction over a range of items without an initial value

@tparam P execution policy type
@tparam I input iterator type
@tparam T value type
@tparam O binary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param res pointer to the result
@param op binary operator to apply to reduce elements

Similar to tf::cuda_reduce but does not assume any initial value in @c res 
to reduce. Please refer to @ref CUDASTDReduce for more details.
*/
template<typename P, typename I, typename T, typename O>
void cuda_uninitialized_reduce(P&& p, I first, I last, T* res, O op) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }
  
  // allocate temporary buffer
  cudaDeviceMemory<std::byte> temp(cuda_reduce_buffer_size<P, T>(count));
  auto buf = temp.data();
  
  // reduction loop
  detail::cuda_reduce_loop(p, first, count, res, op, false, buf);
  
  // synchronize the execution
  p.synchronize();
}

/**
@brief performs asynchronous parallel reduction over a range of items without
       an initial value

@tparam P execution policy type
@tparam I input iterator type
@tparam T value type
@tparam O binary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param res pointer to the result
@param op binary operator to apply to reduce elements
@param buf pointer to the temporary buffer

Asynchronous version of tf::cuda_uninitialized_reduce. 
Please refer to @ref CUDASTDReduce for more details.
*/
template <typename P, typename I, typename T, typename O>
void cuda_uninitialized_reduce_async(
  P&& p, I first, I last, T* res, O op, void* buf
) {
  unsigned count = std::distance(first, last);
  if(count == 0) {
    return;
  }
  detail::cuda_reduce_loop(p, first, count, res, op, false, buf);
}

// ----------------------------------------------------------------------------
// transform_reduce
// ----------------------------------------------------------------------------

/**
@brief performs parallel reduction over a range of transformed items without
       an initial value

@tparam P execution policy type
@tparam I input iterator type
@tparam T value type
@tparam O binary operator type
@tparam U unary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param res pointer to the result
@param bop binary operator to apply to reduce elements
@param uop unary operator to apply to transform elements

Transforms each element in the range using the unary operator @c uop 
and then reduce these transformed elements to @c res using the binary
operator @c bop. Please refer to @ref CUDASTDReduce for more details.
*/
template<typename P, typename I, typename T, typename O, typename U>
void cuda_transform_reduce(P&& p, I first, I last, T* res, O bop, U uop) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }
  
  // allocate temporary buffer
  cudaDeviceMemory<std::byte> temp(cuda_reduce_buffer_size<P, T>(count));
  auto buf = temp.data();
  
  // reduction loop
  //detail::cuda_transform_reduce_loop(p, first, count, res, bop, uop, true, 0, buf);
  detail::cuda_reduce_loop(p, 
    cuda_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }), 
    count, res, bop, true, buf
  );
  
  // synchronize the execution
  p.synchronize();
}

/**
@brief performs asynchronous parallel reduction over a range of transformed items 
       without an initial value

@tparam P execution policy type
@tparam I input iterator type
@tparam T value type
@tparam O binary operator type
@tparam U unary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param res pointer to the result
@param bop binary operator to apply to reduce elements
@param uop unary operator to apply to transform elements
@param buf pointer to the temporary buffer

Asynchronous version of tf::cuda_transform_reduce.
Please refer to @ref CUDASTDReduce for more details.
*/
template<typename P, typename I, typename T, typename O, typename U>
void cuda_transform_reduce_async(
  P&& p, I first, I last, T* res, O bop, U uop, void* buf
) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }

  // reduction loop
  //detail::cuda_transform_reduce_loop(p, first, count, res, bop, uop, true, s, buf);
  detail::cuda_reduce_loop(p, 
    cuda_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }), 
    count, res, bop, true, buf
  );
}

// ----------------------------------------------------------------------------
// transform_uninitialized_reduce
// ----------------------------------------------------------------------------

/**
@brief performs parallel reduction over a range of transformed items 
       with an initial value

@tparam P execution policy type
@tparam I input iterator type
@tparam T value type
@tparam O binary operator type
@tparam U unary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param res pointer to the result
@param bop binary operator to apply to reduce elements
@param uop unary operator to apply to transform elements

Similar to tf::cuda_transform_reduce but does not assum any initial value
to reduce. Please refer to @ref CUDASTDReduce for more details.
*/
template<typename P, typename I, typename T, typename O, typename U>
void cuda_transform_uninitialized_reduce(
  P&& p, I first, I last, T* res, O bop, U uop
) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }
  
  // allocate temporary buffer
  cudaDeviceMemory<std::byte> temp(cuda_reduce_buffer_size<P, T>(count));
  auto buf = temp.data();
  
  // reduction loop
  //detail::cuda_transform_reduce_loop(p, first, count, res, bop, uop, false, 0, buf);
  detail::cuda_reduce_loop(p,
    cuda_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }), 
    count, res, bop, false, buf
  );
  
  // synchronize the execution
  p.synchronize();
}

/**
@brief performs asynchronous parallel reduction over a range of transformed items 
       with an initial value

@tparam P execution policy type
@tparam I input iterator type
@tparam T value type
@tparam O binary operator type
@tparam U unary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param res pointer to the result
@param bop binary operator to apply to reduce elements
@param uop unary operator to apply to transform elements
@param buf pointer to the temporary buffer

Asynchronous version of tf::cuda_transform_uninitialized_reduce.
Please refer to @ref CUDASTDReduce for more details.
*/
template<typename P, typename I, typename T, typename O, typename U>
void cuda_transform_uninitialized_reduce_async(
  P&& p, I first, I last, T* res, O bop, U uop, void* buf
) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }
  
  // reduction loop
  //detail::cuda_transform_reduce_loop(
  //  p, first, count, res, bop, uop, false, s, buf
  //);
  detail::cuda_reduce_loop(p, 
    cuda_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }), 
    count, res, bop, false, buf
  );
}

// ----------------------------------------------------------------------------

//template <typename T, typename C>
//__device__ void cuda_warp_reduce(
//  volatile T* shm, size_t N, size_t tid, C op
//) {
//  if(tid + 32 < N) shm[tid] = op(shm[tid], shm[tid+32]);
//  if(tid + 16 < N) shm[tid] = op(shm[tid], shm[tid+16]);
//  if(tid +  8 < N) shm[tid] = op(shm[tid], shm[tid+8]);
//  if(tid +  4 < N) shm[tid] = op(shm[tid], shm[tid+4]);
//  if(tid +  2 < N) shm[tid] = op(shm[tid], shm[tid+2]);
//  if(tid +  1 < N) shm[tid] = op(shm[tid], shm[tid+1]);
//}
//
//template <typename I, typename T, typename C, bool uninitialized>
//__global__ void cuda_reduce(I first, size_t N, T* res, C op) {
//
//  size_t tid = threadIdx.x;
//
//  if(tid >= N) {
//    return;
//  }
//  
//  cudaSharedMemory<T> shared_memory;
//  T* shm = shared_memory.get();
//
//  shm[tid] = *(first+tid);
//  
//  for(size_t i=tid+blockDim.x; i<N; i+=blockDim.x) {
//    shm[tid] = op(shm[tid], *(first+i));
//  }
//
//  __syncthreads();
//
//  for(size_t s = blockDim.x / 2; s > 32; s >>= 1) {
//    if(tid < s && tid + s < N) {
//      shm[tid] = op(shm[tid], shm[tid+s]);
//    }
//    __syncthreads();
//  }
//
//  if(tid < 32) {
//    cuda_warp_reduce(shm, N, tid, op);
//  }
//  
//  if(tid == 0) {
//    if constexpr (uninitialized) {
//      *res = shm[0];
//    }
//    else {
//      *res = op(*res, shm[0]);
//    }
//  }
//}

// ----------------------------------------------------------------------------
// cudaFlow 
// ----------------------------------------------------------------------------

// Function: reduce
template <typename I, typename T, typename B>
cudaTask cudaFlow::reduce(I first, I last, T* result, B bop) {
  return capture([=](cudaFlowCapturer& cap){
    cap.make_optimizer<cudaLinearCapturing>();
    cap.reduce(first, last, result, bop);
  });
}

// Function: uninitialized_reduce
template <typename I, typename T, typename B>
cudaTask cudaFlow::uninitialized_reduce(I first, I last, T* result, B bop) {
  return capture([=](cudaFlowCapturer& cap){
    cap.make_optimizer<cudaLinearCapturing>();
    cap.uninitialized_reduce(first, last, result, bop);
  });
}

// Function: transform_reduce
template <typename I, typename T, typename B, typename U>
cudaTask cudaFlow::transform_reduce(I first, I last, T* result, B bop, U uop) {
  return capture([=](cudaFlowCapturer& cap){
    cap.make_optimizer<cudaLinearCapturing>();
    cap.transform_reduce(first, last, result, bop, uop);
  });
}

// Function: transform_uninitialized_reduce
template <typename I, typename T, typename B, typename U>
cudaTask cudaFlow::transform_uninitialized_reduce(
  I first, I last, T* result, B bop, U uop
) {
  return capture([=](cudaFlowCapturer& cap){
    cap.make_optimizer<cudaLinearCapturing>();
    cap.transform_uninitialized_reduce(first, last, result, bop, uop);
  });
}

// Function: reduce
template <typename I, typename T, typename C>
void cudaFlow::reduce(cudaTask task, I first, I last, T* result, C op) {
  capture(task, [=](cudaFlowCapturer& cap){
    cap.make_optimizer<cudaLinearCapturing>();
    cap.reduce(first, last, result, op);
  });
}

// Function: uninitialized_reduce
template <typename I, typename T, typename C>
void cudaFlow::uninitialized_reduce(
  cudaTask task, I first, I last, T* result, C op
) {
  capture(task, [=](cudaFlowCapturer& cap){
    cap.make_optimizer<cudaLinearCapturing>();
    cap.uninitialized_reduce(first, last, result, op);
  });
}

// Function: transform_reduce
template <typename I, typename T, typename B, typename U>
void cudaFlow::transform_reduce(
  cudaTask task, I first, I last, T* result, B bop, U uop
) {
  capture(task, [=](cudaFlowCapturer& cap){
    cap.make_optimizer<cudaLinearCapturing>();
    cap.transform_reduce(first, last, result, bop, uop);
  });
}

// Function: transform_uninitialized_reduce
template <typename I, typename T, typename B, typename U>
void cudaFlow::transform_uninitialized_reduce(
  cudaTask task, I first, I last, T* result, B bop, U uop
) {
  capture(task, [=](cudaFlowCapturer& cap){
    cap.make_optimizer<cudaLinearCapturing>();
    cap.transform_uninitialized_reduce(first, last, result, bop, uop);
  });
}

// ----------------------------------------------------------------------------
// cudaFlowCapturer
// ----------------------------------------------------------------------------

// Function: reduce
template <typename I, typename T, typename C>
cudaTask cudaFlowCapturer::reduce(I first, I last, T* result, C c) {
  
  // TODO
  auto bufsz = cuda_reduce_buffer_size<cudaDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  return on([=, buf=MoC{cudaDeviceMemory<std::byte>(bufsz)}] 
  (cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_reduce_async(p, first, last, result, c, buf.get().data());
  });
}

// Function: uninitialized_reduce
template <typename I, typename T, typename C>
cudaTask cudaFlowCapturer::uninitialized_reduce(I first, I last, T* result, C c) {
  
  // TODO
  auto bufsz = cuda_reduce_buffer_size<cudaDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  return on([=, buf=MoC{cudaDeviceMemory<std::byte>(bufsz)}] 
  (cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_uninitialized_reduce_async(p, first, last, result, c, buf.get().data());
  });
}

// Function: transform_reduce
template <typename I, typename T, typename C, typename U>
cudaTask cudaFlowCapturer::transform_reduce(
  I first, I last, T* result, C bop, U uop
) {
  
  // TODO
  auto bufsz = cuda_reduce_buffer_size<cudaDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  return on([=, buf=MoC{cudaDeviceMemory<std::byte>(bufsz)}] 
  (cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_transform_reduce_async(
      p, first, last, result, bop, uop, buf.get().data()
    );
  });
}

// Function: transform_uninitialized_reduce
template <typename I, typename T, typename C, typename U>
cudaTask cudaFlowCapturer::transform_uninitialized_reduce(
  I first, I last, T* result, C bop, U uop) {
  
  // TODO
  auto bufsz = cuda_reduce_buffer_size<cudaDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  return on([=, buf=MoC{cudaDeviceMemory<std::byte>(bufsz)}] 
  (cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_transform_uninitialized_reduce_async(
      p, first, last, result, bop, uop, buf.get().data()
    );
  });
}

// Function: reduce
template <typename I, typename T, typename C>
void cudaFlowCapturer::reduce(
  cudaTask task, I first, I last, T* result, C c
) {
  
  // TODO
  auto bufsz = cuda_reduce_buffer_size<cudaDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  on(task, [=, buf=MoC{cudaDeviceMemory<std::byte>(bufsz)}] 
  (cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_reduce_async(p, first, last, result, c, buf.get().data());
  });
}

// Function: uninitialized_reduce
template <typename I, typename T, typename C>
void cudaFlowCapturer::uninitialized_reduce(
  cudaTask task, I first, I last, T* result, C c
) {
  // TODO  
  auto bufsz = cuda_reduce_buffer_size<cudaDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  on(task, [=, buf=MoC{cudaDeviceMemory<std::byte>(bufsz)}] 
  (cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_uninitialized_reduce_async(p, first, last, result, c, buf.get().data());
  });
}

// Function: transform_reduce
template <typename I, typename T, typename C, typename U>
void cudaFlowCapturer::transform_reduce(
  cudaTask task, I first, I last, T* result, C bop, U uop
) {
  
  // TODO
  auto bufsz = cuda_reduce_buffer_size<cudaDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  on(task, [=, buf=MoC{cudaDeviceMemory<std::byte>(bufsz)}] 
  (cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_transform_reduce_async(
      p, first, last, result, bop, uop, buf.get().data()
    );
  });
}

// Function: transform_uninitialized_reduce
template <typename I, typename T, typename C, typename U>
void cudaFlowCapturer::transform_uninitialized_reduce(
  cudaTask task, I first, I last, T* result, C bop, U uop
) {

  // TODO
  auto bufsz = cuda_reduce_buffer_size<cudaDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  on(task, [=, buf=MoC{cudaDeviceMemory<std::byte>(bufsz)}] 
  (cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_transform_uninitialized_reduce_async(
      p, first, last, result, bop, uop, buf.get().data()
    );
  });
}

}  // end of namespace tf -----------------------------------------------------

