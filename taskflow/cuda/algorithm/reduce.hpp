#pragma once

#include "../cudaflow.hpp"

/**
@file taskflow/cuda/algorithm/reduce.hpp
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

// ----------------------------------------------------------------------------
// cuda_reduce
// ----------------------------------------------------------------------------

/**
@private 
*/
template <size_t nt, size_t vt, typename I, typename T, typename O>
__global__ void cuda_reduce_kernel(
  I input, unsigned count, T* res, O op, void* ptr
) {
  
  using U = typename std::iterator_traits<I>::value_type;

  __shared__ typename cudaBlockReduce<nt, U>::Storage shm;
  
  auto tid = threadIdx.x;
  auto bid = blockIdx.x;
  auto tile = cuda_get_tile(bid, nt*vt, count);
  auto x = cuda_mem_to_reg_strided<nt, vt>(
    input + tile.begin, tid, tile.count()
  );

  // reduce multiple values per thread into a scalar.
  U s;
  cuda_strided_iterate<nt, vt>(
    [&] (auto i, auto) { s = i ? op(s, x[i]) : x[0]; }, tid, tile.count()
  );
  // reduce to a scalar per block.
  s = cudaBlockReduce<nt, U>()(
    tid, s, shm, (tile.count() < nt ? tile.count() : nt), op, false
  );

  if(!tid) {
    auto buf = static_cast<U*>(ptr);
    (count <= nt*vt) ? *res = op(*res, s) : buf[bid] = s;
  }
}

/** @private */
template <typename P, typename I, typename T, typename O>
void cuda_reduce_loop(
  P&& p, I input, unsigned count, T* res, O op, void* ptr
) {

  using U = typename std::iterator_traits<I>::value_type;
  using E = std::decay_t<P>;

  auto buf = static_cast<U*>(ptr);
  auto B = E::num_blocks(count);

  cuda_reduce_kernel<E::nt, E::vt><<<B, E::nt, 0, p.stream()>>>(
    input, count, res, op, ptr
  );

  if(B > 1) {
    cuda_reduce_loop(p, buf, B, res, op, buf+B);
  }
}

// ----------------------------------------------------------------------------
// cuda_uninitialized_reduce
// ----------------------------------------------------------------------------

/**
@private
*/
template <size_t nt, size_t vt, typename I, typename T, typename O>
__global__ void cuda_uninitialized_reduce_kernel(
  I input, unsigned count, T* res, O op, void* ptr
) {

  using U = typename std::iterator_traits<I>::value_type;

  __shared__ typename cudaBlockReduce<nt, U>::Storage shm;

  auto tid = threadIdx.x;
  auto bid = blockIdx.x;
  auto tile = cuda_get_tile(bid, nt*vt, count);
  auto x = cuda_mem_to_reg_strided<nt, vt>(
    input + tile.begin, tid, tile.count()
  );

  // reduce multiple values per thread into a scalar.
  U s;
  cuda_strided_iterate<nt, vt>(
    [&] (auto i, auto) { s = i ? op(s, x[i]) : x[0]; }, tid, tile.count()
  );

  // reduce to a scalar per block.
  s = cudaBlockReduce<nt, U>()(
    tid, s, shm, (tile.count() < nt ? tile.count() : nt), op, false
  );

  if(!tid) {
    auto buf = static_cast<U*>(ptr);
    (count <= nt*vt) ? *res = s : buf[bid] = s;
  }
}

/** 
@private 
*/
template <typename P, typename I, typename T, typename O>
void cuda_uninitialized_reduce_loop(
  P&& p, I input, unsigned count, T* res, O op, void* ptr
) {

  using U = typename std::iterator_traits<I>::value_type;
  using E = std::decay_t<P>;

  auto buf = static_cast<U*>(ptr);
  auto B = (count + E::nv - 1) / E::nv;

  cuda_uninitialized_reduce_kernel<E::nt, E:: vt><<<B, E::nt, 0, p.stream()>>>(
    input, count, res, op, buf
  );

  if(B > 1) {
    cuda_uninitialized_reduce_loop(p, buf, B, res, op, buf+B);
  }
}

}  // namespace tf::detail ----------------------------------------------------

namespace tf {

// Function: reduce_bufsz
template <unsigned NT, unsigned VT>  
template <typename T>
unsigned cudaExecutionPolicy<NT, VT>::reduce_bufsz(unsigned count) {
  unsigned B = num_blocks(count);
  unsigned n = 0;
  while(B > 1) {
    n += B;
    B = num_blocks(B);
  }
  return n*sizeof(T);
}

// ----------------------------------------------------------------------------
// cuda_reduce
// ----------------------------------------------------------------------------

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

This method is equivalent to the parallel execution of the following loop on a GPU:

@code{.cpp}
while (first != last) {
  *result = op(*result, *first++);
}
@endcode
 */
template <typename P, typename I, typename T, typename O>
void cuda_reduce(
  P&& p, I first, I last, T* res, O op, void* buf
) {
  unsigned count = std::distance(first, last);
  if(count == 0) {
    return;
  }
  detail::cuda_reduce_loop(p, first, count, res, op, buf);
}

// ----------------------------------------------------------------------------
// cuda_uninitialized_reduce
// ----------------------------------------------------------------------------

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

This method is equivalent to the parallel execution of the following loop
on a GPU:

@code{.cpp}
*result = *first++;  // no initial values partitipcate in the loop
while (first != last) {
  *result = op(*result, *first++);
}
@endcode
*/
template <typename P, typename I, typename T, typename O>
void cuda_uninitialized_reduce(
  P&& p, I first, I last, T* res, O op, void* buf
) {
  unsigned count = std::distance(first, last);
  if(count == 0) {
    return;
  }
  detail::cuda_uninitialized_reduce_loop(p, first, count, res, op, buf);
}

// ----------------------------------------------------------------------------
// transform_reduce
// ----------------------------------------------------------------------------

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

This method is equivalent to the parallel execution of the following loop on a GPU:

@code{.cpp}
while (first != last) {
  *result = bop(*result, uop(*first++));
}
@endcode
*/
template<typename P, typename I, typename T, typename O, typename U>
void cuda_transform_reduce(
  P&& p, I first, I last, T* res, O bop, U uop, void* buf
) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }

  // reduction loop
  detail::cuda_reduce_loop(p,
    cuda_make_load_iterator<T>([=]__device__(auto i){
      return uop(*(first+i));
    }),
    count, res, bop, buf
  );
}

// ----------------------------------------------------------------------------
// transform_uninitialized_reduce
// ----------------------------------------------------------------------------

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

This method is equivalent to the parallel execution of the following loop
on a GPU:

@code{.cpp}
*result = uop(*first++);  // no initial values partitipcate in the loop
while (first != last) {
  *result = bop(*result, uop(*first++));
}
@endcode
*/
template<typename P, typename I, typename T, typename O, typename U>
void cuda_uninitialized_transform_reduce(
  P&& p, I first, I last, T* res, O bop, U uop, void* buf
) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }

  detail::cuda_uninitialized_reduce_loop(p,
    cuda_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }),
    count, res, bop, buf
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


}  // end of namespace tf -----------------------------------------------------

