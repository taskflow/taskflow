#pragma once

#include "../cuda_flow.hpp"
#include "../cuda_capturer.hpp"
#include "../cuda_meta.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// scan
// ----------------------------------------------------------------------------

namespace detail {

inline constexpr unsigned cudaScanRecursionThreshold = 8; 

enum class cudaScanType {
  EXCLUSIVE,
  INCLUSIVE
};

template<typename T, unsigned vt = 0, bool is_array = (vt > 0)>
struct cudaScanResult {
  T scan;
  T reduction;
};

template<typename T, unsigned vt>
struct cudaScanResult<T, vt, true> {
  cudaArray<T, vt> scan;
  T reduction;
};

//-----------------------------------------------------------------------------

/** @private */
template<unsigned nt, typename T>
struct cudaBlockScan {

  const static unsigned num_warps  = nt / CUDA_WARP_SIZE;
  const static unsigned num_passes = log2(nt);
  const static unsigned capacity   = nt + num_warps; 

  union storage_t {
    T data[2 * nt];
    struct { T threads[nt], warps[num_warps]; };
  };

  // standard scan
  template<typename op_t>
  __device__ cudaScanResult<T> operator ()(
    unsigned tid, 
    T x, 
    storage_t& storage, 
    unsigned count = nt, 
    op_t op = op_t(), 
    T init = T(), 
    cudaScanType type = cudaScanType::EXCLUSIVE
  ) const;

  // vectorized scan. accepts multiple values per thread and adds in
  // optional global carry-in
  template<unsigned vt, typename op_t>
  __device__ cudaScanResult<T, vt> operator()(
    unsigned tid, 
    cudaArray<T, vt> x, 
    storage_t& storage, 
    T carry_in = T(), 
    bool use_carry_in = false, 
    unsigned count = nt, 
    op_t op = op_t(), 
    T init = T(),
    cudaScanType type = cudaScanType::EXCLUSIVE
  ) const;
};
  
// standard scan
template <unsigned nt, typename T>
template<typename op_t>
__device__ cudaScanResult<T> cudaBlockScan<nt, T>::operator () (
  unsigned tid, T x, storage_t& storage, unsigned count, op_t op, 
  T init, cudaScanType type
) const {

  unsigned first = 0;
  storage.data[first + tid] = x;
  __syncthreads();

  cuda_iterate<num_passes>([&](auto pass) {
    if(auto offset = 1<<pass; tid >= offset) {
      x = op(storage.data[first + tid - offset], x);
    }
    first = nt - first;
    storage.data[first + tid] = x;
    __syncthreads();
  });

  cudaScanResult<T> result;
  result.reduction = storage.data[first + count - 1];
  result.scan = (tid < count) ? 
    (cudaScanType::INCLUSIVE == type ? x :
      (tid ? storage.data[first + tid - 1] : init)) :
    result.reduction;
  __syncthreads();

  return result;
}

// vectorized scan block
template <unsigned nt, typename T>
template<unsigned vt, typename op_t>
__device__ cudaScanResult<T, vt> cudaBlockScan<nt, T>::operator()(
  unsigned tid, 
  cudaArray<T, vt> x, 
  storage_t& storage, 
  T carry_in, 
  bool use_carry_in, 
  unsigned count, op_t op, 
  T init,
  cudaScanType type
) const {

  // Start with an inclusive scan of the in-range elements.
  if(count >= nt * vt) {
    cuda_iterate<vt>([&](auto i) {
      x[i] = i ? op(x[i], x[i - 1]) : x[i];
    });
  } else {
    cuda_iterate<vt>([&](auto i) {
      auto index = vt * tid + i;
      x[i] = i ? 
        ((index < count) ? op(x[i], x[i - 1]) : x[i - 1]) :
        (x[i] = (index < count) ? x[i] : init);
    });
  }

  // Scan the thread-local reductions for a carry-in for each thread.
  auto result = operator()(
    tid, x[vt - 1], storage, 
    (count + vt - 1) / vt, op, init, cudaScanType::EXCLUSIVE
  );

  // Perform the scan downsweep and add both the global carry-in and the
  // thread carry-in to the values.
  if(use_carry_in) {
    result.reduction = op(carry_in, result.reduction);
    result.scan = tid ? op(carry_in, result.scan) : carry_in;
  } else {
    use_carry_in = tid > 0;
  }

  cudaArray<T, vt> y;
  cuda_iterate<vt>([&](auto i) {
    if(cudaScanType::EXCLUSIVE == type) {
      y[i] = i ? x[i - 1] : result.scan;
      if(use_carry_in && i > 0) y[i] = op(result.scan, y[i]);
    } else {
      y[i] = use_carry_in ? op(x[i], result.scan) : x[i];
    }
  });

  return cudaScanResult<T, vt> { y, result.reduction };
}

/** 
@private 

@brief main scan loop
*/
template<typename P, typename I, typename O, typename C, typename T>
void cuda_scan_loop(
  P&&p, 
  cudaScanType scan_type,
  I input, 
  unsigned count, 
  O output, 
  C op, 
  //reduction_it reduction, 
  cudaStream_t S,
  T* buffer
) {

  using E = std::decay_t<P>;

  //launch_t::cta_dim(context).B(count);
  unsigned B = (count + E::nv - 1) / E::nv;

  if(B > cudaScanRecursionThreshold) {

    //cudaDeviceMemory<T> partials(B);
    //auto buffer = partials.data();

    // upsweep phase
    cuda_kernel<<<B, E::nt, 0, S>>>([=] __device__ (auto tid, auto bid) {

      __shared__ typename cudaBlockReduce<E::nt, T>::Storage shm;

      // Load the tile's data into register.
      auto tile = cuda_get_tile(bid, E::nv, count);
      auto x = cuda_mem_to_reg_strided<E::nt, E::vt>(
        input + tile.begin, tid, tile.count()
      );

      // Reduce the thread's values into a scalar.
      T scalar;
      cuda_strided_iterate<E::nt, E::vt>(
        [&](auto i, auto j) { scalar = i ? op(scalar, x[i]) : x[0]; }, 
        tid, tile.count()
      );

      // Reduce across all threads.
      auto all_reduce = cudaBlockReduce<E::nt, T>()(
        tid, scalar, shm, tile.count(), op
      );

      // Store the final reduction to the partials.
      if(!tid) {
        buffer[bid] = all_reduce;
      }
    });

    // recursively call scan
    //cuda_scan_loop(p, cudaScanType::EXCLUSIVE, buffer, B, buffer, op, S);
    cuda_scan_loop(
      p, cudaScanType::EXCLUSIVE, buffer, B, buffer, op, S, buffer+B
    );

    // downsweep: perform an intra-tile scan and add the scan of the partials
    // as carry-in
    cuda_kernel<<<B, E::nt, 0, S>>>([=] __device__ (auto tid, auto bid) {

      using scan_t = cudaBlockScan<E::nt, T>;

      __shared__ union {
        typename scan_t::storage_t scan;
        T values[E::nv];
      } shared;

      // Load a tile to register in thread order.
      auto tile = cuda_get_tile(bid, E::nv, count);
      auto x = cuda_mem_to_reg_thread<E::nt, E::vt>(
        input + tile.begin, tid, tile.count(), shared.values
      );

      // Scan the array with carry-in from the partials.
      auto y = scan_t()(tid, x, shared.scan, 
        buffer[bid], bid > 0, tile.count(), op, T(), 
        scan_type).scan;

      // Store the scanned values to the output.
      cuda_reg_to_mem_thread<E::nt, E::vt>(
        y, tid, tile.count(), output + tile.begin, shared.values
      );
    });
  
  } else {

    // Small input specialization. This is the non-recursive branch.
    cuda_kernel<<<1, E::nt, 0, S>>>([=] __device__ (auto tid, auto bid) {
     
      using scan_t = cudaBlockScan<E::nt, T>;

      __shared__ union {
        typename scan_t::storage_t scan;
        T values[E::nv];
      } shared;

      auto carry_in = T();
      for(unsigned cur = 0; cur < count; cur += E::nv) {
        // Cooperatively load values into register.
        auto count2 = min(count - cur, E::nv);
        
        auto x = cuda_mem_to_reg_thread<E::nt, E::vt>(input + cur, 
          tid, count2, shared.values);
        
        auto result = scan_t()(tid, x, shared.scan,
          carry_in, cur > 0, count2, op, T(), scan_type);
        
        // Store the scanned values back to global memory.
        cuda_reg_to_mem_thread<E::nt, E::vt>(result.scan, tid, count2, 
          output + cur, shared.values);
        
        // Roll the reduction into carry_in.
        carry_in = result.reduction;
      }

      // Store the carry-out to the reduction pointer. This may be a
      // discard_iterator_t if no reduction is wanted.
      //if(!tid) *reduction = carry_in;
    });
  }
}

}  // namespace tf::detail ----------------------------------------------------

/** 
@function cuda_scan_buffer_size
*/
template <typename P>
unsigned cuda_scan_buffer_size(P&&, unsigned count) {
  using E = std::decay_t<P>;
  unsigned B = (count + E::nv - 1) / E::nv;
  unsigned n = 0;
  for(auto b=B; b>detail::cudaScanRecursionThreshold; b=(b+E::nv-1)/E::nv) {
    n += b;
  }
  return n;
}

/**
@function cuda_scan_buffer_size
*/
template <typename I>
unsigned cuda_scan_buffer_size(I first, I last) {
  return cuda_scan_buffer_size(
    cudaDefaultExecutionPolicy, std::distance(first, last)
  );
}

// ----------------------------------------------------------------------------
// inclusive scan
// ----------------------------------------------------------------------------

/**
@function cuda_inclusive_scan
*/
template<typename P, typename I, typename O, typename C>
void cuda_inclusive_scan(P&& p, I first, I last, O output, C op) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }
  
  using T = typename std::iterator_traits<O>::value_type;
  
  // allocate temporary buffer
  cudaDeviceMemory<T> temp(cuda_scan_buffer_size(p, count));
  auto buf = temp.data();
  
  // launch the scan loop
  detail::cuda_scan_loop(
    p, detail::cudaScanType::INCLUSIVE, first, count, output, op, 0, buf
  );
  
  // synchronize the execution
  TF_CHECK_CUDA(
    cudaStreamSynchronize(0), "inclusive_scan failed to sync stream 0"
  );
}

/**
@function cuda_inclusive_scan
*/
template <typename I, typename O, typename C>
void cuda_inclusive_scan(I first, I last, O output, C op) {
  cuda_inclusive_scan(cudaDefaultExecutionPolicy, first, last, output, op);
}

/**
@function cuda_inclusive_scan
*/
template<typename P, typename I, typename O, typename C, typename T>
void cuda_inclusive_scan_async(
  P&& p, I first, I last, O output, C op, cudaStream_t s, T* buf
) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }
  
  // launch the scan loop
  detail::cuda_scan_loop(
    p, detail::cudaScanType::INCLUSIVE, first, count, output, op, s, buf
  );
}

/**
@function cuda_inclusive_scan
*/
template<typename I, typename O, typename C, typename T>
void cuda_inclusive_scan_async(
  I first, I last, O output, C op, cudaStream_t s, T* buf
) {
  cuda_inclusive_scan_async(
    cudaDefaultExecutionPolicy, first, last, output, op, s, buf
  );
}

// ----------------------------------------------------------------------------
// exclusive scan
// ----------------------------------------------------------------------------

/**
@function cuda_exclusive_scan
*/
template<typename P, typename I, typename O, typename C>
void cuda_exclusive_scan(P&& p, I first, I last, O output, C op) {
  
  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }
  
  using T = typename std::iterator_traits<O>::value_type;
  
  // allocate temporary buffer
  cudaDeviceMemory<T> temp(cuda_scan_buffer_size(p, count));
  auto buf = temp.data();
  
  // launch the scan loop
  detail::cuda_scan_loop(
    p, detail::cudaScanType::EXCLUSIVE, first, count, output, op, 0, buf
  );
  
  // synchronize the execution
  TF_CHECK_CUDA(
    cudaStreamSynchronize(0), "exclusive_scan failed to sync stream 0"
  );
}

/**
@function cuda_exclusive_scan
*/
template <typename I, typename O, typename C>
void cuda_exclusive_scan(I first, I last, O output, C op) {
  cuda_exclusive_scan(cudaDefaultExecutionPolicy, first, last, output, op);
}

/**
@function cuda_exclusive_scan
*/
template<typename P, typename I, typename O, typename C, typename T>
void cuda_exclusive_scan_async(
  P&& p, I first, I last, O output, C op, cudaStream_t s, T* buf
) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }
  
  // launch the scan loop
  detail::cuda_scan_loop(
    p, detail::cudaScanType::EXCLUSIVE, first, count, output, op, s, buf
  );
}

/**
@function cuda_exclusive_scan
*/
template<typename I, typename O, typename C, typename T>
void cuda_exclusive_scan_async(
  I first, I last, O output, C op, cudaStream_t s, T* buf
) {
  cuda_exclusive_scan_async(
    cudaDefaultExecutionPolicy, first, last, output, op, s, buf
  );
}

// ----------------------------------------------------------------------------
// cudaFlowCapturer
// ----------------------------------------------------------------------------

// Function: inclusive_scan
template <typename I, typename O, typename C>
cudaTask cudaFlowCapturer::inclusive_scan(I first, I last, O output, C op) {
  
  using T = typename std::iterator_traits<O>::value_type;
  
  auto bufsz = cuda_scan_buffer_size(first, last);

  return on([=, buf=MoC{cudaDeviceMemory<T>(bufsz)}] 
  (cudaStream_t stream) mutable {
    cuda_inclusive_scan_async(first, last, output, op, stream, buf.get().data());
  });
}

// Function: exclusive_scan
template <typename I, typename O, typename C>
cudaTask cudaFlowCapturer::exclusive_scan(I first, I last, O output, C op) {
  
  using T = typename std::iterator_traits<O>::value_type;
  
  auto bufsz = cuda_scan_buffer_size(first, last);

  return on([=, buf=MoC{cudaDeviceMemory<T>(bufsz)}] 
  (cudaStream_t stream) mutable {
    cuda_exclusive_scan_async(first, last, output, op, stream, buf.get().data());
  });
}


}  // end of namespace tf -----------------------------------------------------



