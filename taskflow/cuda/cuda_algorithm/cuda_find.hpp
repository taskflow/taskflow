#pragma once

#include "../cuda_flow.hpp"
#include "../cuda_capturer.hpp"
#include "../cuda_meta.hpp"

/** 
@file cuda_find.hpp
@brief cuda find algorithms include file
*/

namespace tf::detail {

/** @private */
template <typename P, typename I, typename U>
void cuda_find_if_loop(P&& p, I input, unsigned count, unsigned* idx, U pred) {

  if(count == 0) {
    cuda_single_task(p, [=] __device__ () { *idx = 0; });
    return;
  }
  
  using E = std::decay_t<P>;
  
  auto B = (count + E::nv - 1) / E::nv;
  
  // set the index to the maximum
  cuda_single_task(p, [=] __device__ () { *idx = count; });
  
  // launch the kernel to atomic-find the minimum
  cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
    
    __shared__ unsigned shm_id;

    if(!tid) {
      shm_id = count;
    }

    __syncthreads();
    
    auto tile = cuda_get_tile(bid, E::nv, count);
    
    auto x = cuda_mem_to_reg_strided<E::nt, E::vt>(
      input + tile.begin, tid, tile.count()
    );

    auto id = count;

    for(unsigned i=0; i<E::vt; i++) {
      auto j = E::nt*i + tid;
      if(j < tile.count() && pred(x[i])) {
        id = j + tile.begin;
        break;
      }
    }
    
    // Note: the reduce version is not faster though
    // reduce to a scalar per block.
    //__shared__ typename cudaBlockReduce<E::nt, unsigned>::Storage shm;

    //id = cudaBlockReduce<E::nt, unsigned>()(
    //  tid, 
    //  id, 
    //  shm, 
    //  (tile.count() < E::nt ? tile.count() : E::nt), 
    //  cuda_minimum<unsigned>{},
    //  false
    //);
    
    // only need the minimum id
    atomicMin(&shm_id, id);
    __syncthreads();
    
    // reduce all to the global memory
    if(!tid) {
      atomicMin(idx, shm_id);
      //atomicMin(idx, id);
    }
  });
}

template <typename T>
struct cudaFindPair {
  T key;
  unsigned index;

  __device__ operator unsigned () const { return index; }
};

/** @private */
template <typename P, typename I, typename O>
void cuda_min_element_loop(
  P&& p, I input, unsigned count, unsigned* res, O op, void* ptr
) {

  if(count == 0) {
    cuda_single_task(p, [=] __device__ () { *res = 0; });
    return;
  }

  using T = typename std::iterator_traits<I>::value_type;

  auto buf = static_cast<cudaFindPair<T>*>(ptr);

  // transform to key-index pair
  cuda_for_each_index_loop(p, 0, 1, count, [=]__device__(auto i){
    buf[i].key = input[i];
    buf[i].index = i;
  });
  
  // perform reduction
  cuda_uninitialized_reduce_loop(p, buf, count, res, 
    [=] __device__ (const auto& a, const auto& b) {
      return op(a.key, b.key) ? a : b;
    },
    buf+count
  );
} 

}  // end of namespace tf::detail ---------------------------------------------

namespace tf {


// ----------------------------------------------------------------------------
// cuda_find_if
// ----------------------------------------------------------------------------

/**
@brief finds the index of the first element that satisfies the given criteria

@tparam P execution policy type
@tparam I input iterator type
@tparam U unary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param idx pointer to the index of the found element
@param op unary operator which returns @c true for the required element

The asynchronous function launches kernels to find the index @c idx of the
first element in the range <tt>[first, last)</tt> 
such that <tt>op(*(first+idx))</tt> is true.
This is equivalent to the parallel execution of the following loop:

@code{.cpp}
unsigned idx = 0;
for(; first != last; ++first, ++idx) {
  if (p(*first)) {
    return idx;
  }
}
return idx;
@endcode
*/
template <typename P, typename I, typename U>
void cuda_find_if(
  P&& p, I first, I last, unsigned* idx, U op
) {
  detail::cuda_find_if_loop(p, first, std::distance(first, last), idx, op);
}

// ----------------------------------------------------------------------------
// cudaFlow
// ----------------------------------------------------------------------------

// Function: find_if
template <typename I, typename U>
cudaTask cudaFlow::find_if(I first, I last, unsigned* idx, U op) {
  return capture([=](cudaFlowCapturer& cap){
    cap.make_optimizer<cudaLinearCapturing>();
    cap.find_if(first, last, idx, op);
  });
}

// Function: find_if
template <typename I, typename U>
void cudaFlow::find_if(cudaTask task, I first, I last, unsigned* idx, U op) {
  capture(task, [=](cudaFlowCapturer& cap){
    cap.make_optimizer<cudaLinearCapturing>();
    cap.find_if(first, last, idx, op);
  });
}

// ----------------------------------------------------------------------------
// cudaFlowCapturer
// ----------------------------------------------------------------------------

// Function: find_if
template <typename I, typename U>
cudaTask cudaFlowCapturer::find_if(I first, I last, unsigned* idx, U op) {
  return on([=](cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_find_if(p, first, last, idx, op);
  });
}

// Function: find_if
template <typename I, typename U>
void cudaFlowCapturer::find_if(
  cudaTask task, I first, I last, unsigned* idx, U op
) {
  on(task, [=](cudaStream_t stream) mutable {
    cudaDefaultExecutionPolicy p(stream);
    cuda_find_if(p, first, last, idx, op);
  });
}

// ----------------------------------------------------------------------------
// cuda_min_element
// ----------------------------------------------------------------------------

/**
@brief queries the buffer size in bytes needed to call tf::cuda_min_element

@tparam P execution policy type
@tparam T value type

@param count number of elements to search

The function is used to decide the buffer size in bytes for calling
tf::cuda_min_element.
*/
template <typename P, typename T>
unsigned cuda_min_element_buffer_size(unsigned count) {
  using E = std::decay_t<P>;
  unsigned B = (count + E::nv - 1) / E::nv;
  unsigned n = 0;
  for(auto b=B; b>1; n += (b=(b+E::nv-1)/E::nv));
  return (count + n)*sizeof(detail::cudaFindPair<T>);
}

/**
@brief finds the index of the minimum element in a range

@tparam P execution policy type
@tparam I input iterator type
@tparam O comparator type

@param p execution policy object
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param idx solution index of the minimum element
@param op comparison function object
@param buf pointer to the buffer

The function finds the smallest element in the range <tt>[first, last)</tt>
using the given comparator @c op. 
It is equivalent to a parallel execution of the following loop:

@code{.cpp}
if(first == last) {
  return 0;
}
auto smallest = first;
for (++first; first != last; ++first) {
  if (op(*first, *smallest)) {
    smallest = first;
  }
}
return std::distance(first, smallest);
@endcode
*/
template <typename P, typename I, typename O>
void cuda_min_element(P&& p, I first, I last, unsigned* idx, O op, void* buf) {
  detail::cuda_min_element_loop(
    p, first, std::distance(first, last), idx, op, buf
  );
}


}  // end of namespace tf -----------------------------------------------------


