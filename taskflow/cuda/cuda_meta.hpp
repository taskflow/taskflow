#pragma once

#include "cuda_error.hpp"

#include <typeinfo>
#include <type_traits>
#include <iterator>
#include <cassert>
#include <cfloat>
#include <cstdint>

namespace tf {

constexpr unsigned CUDA_WARP_SIZE = 32;

struct cudaEmpty { };

// Template unrolled looping construct.
template<unsigned i, unsigned count, bool valid = (i < count)>
struct cudaIterate {
  template<typename F>
  __host__ __device__ static void eval(F f) {
    f(i);
    cudaIterate<i + 1, count>::eval(f);
  }
};

template<unsigned i, unsigned count>
struct cudaIterate<i, count, false> {
  template<typename F>
  __host__ __device__ static void eval(F f) { }
};

template<unsigned begin, unsigned end, typename F>
__host__ __device__ void cuda_iterate(F f) {
  cudaIterate<begin, end>::eval(f);
}

template<unsigned count, typename F>
__host__ __device__ void cuda_iterate(F f) {
  cuda_iterate<0, count>(f);
}

template<unsigned count, typename T>
__host__ __device__ T reduce(const T(&x)[count]) {
  T y;
  cuda_iterate<count>([&](auto i) { y = i ? x[i] + y : x[i]; });
  return y;
}

template<unsigned count, typename T>
__host__ __device__ void fill(T(&x)[count], T val) {
  cuda_iterate<count>([&](auto i) { x[i] = val; });
}

// Invoke unconditionally.
template<unsigned nt, unsigned vt, typename F>
__device__ void cuda_strided_iterate(F f, unsigned tid) {
  cuda_iterate<vt>([=](auto i) { f(i, nt * i + tid); });
}

// Check range.
template<unsigned nt, unsigned vt, unsigned vt0 = vt, typename F>
__device__ void cuda_strided_iterate(F f, unsigned tid, unsigned count) {
  // Unroll the first vt0 elements of each thread.
  if(vt0 > 1 && count >= nt * vt0) {
    cuda_strided_iterate<nt, vt0>(f, tid);    // No checking
  } else {
    cuda_iterate<vt0>([=](auto i) {
      auto j = nt * i + tid;
      if(j < count) f(i, j);
    });
  }

  cuda_iterate<vt0, vt>([=](auto i) {
    auto j = nt * i + tid;
    if(j < count) f(i, j);
  });
}

template<unsigned vt, typename F>
__device__ void cuda_thread_iterate(F f, unsigned tid) {
  cuda_iterate<vt>([=](auto i) { f(i, vt * tid + i); });
}


// ----------------------------------------------------------------------------
// cudaRange 
// ----------------------------------------------------------------------------

// range
struct cudaRange {
  unsigned begin, end;
  __host__ __device__ unsigned size() const { return end - begin; }
  __host__ __device__ unsigned count() const { return size(); }
  __host__ __device__ bool valid() const { return end > begin; }
};

__host__ __device__ cudaRange cuda_get_tile(unsigned b, unsigned nv, unsigned count) {
  return cudaRange { nv * b, min(count, nv * (b + 1)) };
}

// ----------------------------------------------------------------------------
// cudaArray
// ----------------------------------------------------------------------------

template<typename T, unsigned size>
struct cudaArray {
  T data[size];

  __host__ __device__ T operator[](unsigned i) const { return data[i]; }
  __host__ __device__ T& operator[](unsigned i) { return data[i]; }

  cudaArray() = default;
  cudaArray(const cudaArray&) = default;
  cudaArray& operator=(const cudaArray&) = default;

  // Fill the array with x.
  __host__ __device__ cudaArray(T x) { 
    cuda_iterate<size>([&](unsigned i) { data[i] = x; });  
  }
};

template<typename T>
struct cudaArray<T, 0> { 
  __host__ __device__ T operator[](unsigned i) const { return T(); }
  __host__ __device__ T& operator[](unsigned i) { return *(T*)nullptr; }
};

// ----------------------------------------------------------------------------
// reg <-> mem 
// ----------------------------------------------------------------------------


template<unsigned nt, unsigned vt, unsigned vt0 = vt, typename I>
//__device__ cudaArray<typename std::iterator_traits<I>::value_type, vt> 
__device__ auto cuda_mem_to_reg_strided(I mem, unsigned tid, unsigned count) {
  using T = typename std::iterator_traits<I>::value_type;
  cudaArray<T, vt> x;
  cuda_strided_iterate<nt, vt, vt0>(
    [&](auto i, auto j) { x[i] = mem[j]; }, tid, count
  );
  return x;
}

template<unsigned nt, unsigned vt, unsigned vt0 = vt, typename I, typename O>
__device__ auto cuda_transform_mem_to_reg_strided(
  I mem, unsigned tid, unsigned count, O op
) {
  using T = std::invoke_result_t<O, typename std::iterator_traits<I>::value_type>;
  cudaArray<T, vt> x;
  cuda_strided_iterate<nt, vt, vt0>(
    [&](auto i, auto j) { x[i] = op(mem[j]); }, tid, count
  );
  return x;
}

// ----------------------------------------------------------------------------
// launch kernel
// ----------------------------------------------------------------------------

template<typename F, typename... args_t>
__global__ void cuda_kernel(F f, args_t... args) {
  f(threadIdx.x, blockIdx.x, args...);
}

// ----------------------------------------------------------------------------
// Launch parameters
// ----------------------------------------------------------------------------

template<unsigned NT, unsigned VT>
struct cudaExecutionPolicy {
  const static unsigned nt = NT;     
  const static unsigned vt = VT;
  const static unsigned nv = NT*VT;
};

using cudaDefaultExecutionPolicy = cudaExecutionPolicy<512, 7>;


}  // end of namespace tf -----------------------------------------------------



