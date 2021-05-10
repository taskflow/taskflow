#pragma once

#include "cuda_error.hpp"

#include <typeinfo>
#include <type_traits>
#include <iterator>
#include <cassert>
#include <cfloat>
#include <cstdint>

namespace tf {

inline constexpr unsigned CUDA_WARP_SIZE = 32;

// empty type
struct cudaEmpty { };

// ----------------------------------------------------------------------------
// iterator unrolling
// ----------------------------------------------------------------------------

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
// thread reg <-> global mem 
// ----------------------------------------------------------------------------

template<unsigned nt, unsigned vt, unsigned vt0 = vt, typename I>
__device__ auto cuda_mem_to_reg_strided(I mem, unsigned tid, unsigned count) {
  using T = typename std::iterator_traits<I>::value_type;
  cudaArray<T, vt> x;
  cuda_strided_iterate<nt, vt, vt0>(
    [&](auto i, auto j) { x[i] = mem[j]; }, tid, count
  );
  return x;
}

template<unsigned nt, unsigned vt, unsigned vt0 = vt, typename T, typename it_t>
__device__ void cuda_reg_to_mem_strided(
  cudaArray<T, vt> x, unsigned tid, unsigned count, it_t mem) {

  cuda_strided_iterate<nt, vt, vt0>(
    [=](auto i, auto j) { mem[j] = x[i]; }, tid, count
  );
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
// thread reg <-> shared
// ----------------------------------------------------------------------------

template<unsigned nt, unsigned vt, typename T, unsigned shared_size>
__device__ void cuda_reg_to_shared_thread(
  cudaArray<T, vt> x, unsigned tid, T (&shared)[shared_size], bool sync = true
) {

  static_assert(shared_size >= nt * vt,
    "reg_to_shared_thread must have at least nt * vt storage");

  cuda_thread_iterate<vt>([&](auto i, auto j) { shared[j] = x[i]; }, tid);

  if(sync) __syncthreads();
}

template<unsigned nt, unsigned vt, typename T, unsigned shared_size>
__device__ auto cuda_shared_to_reg_thread(
  const T (&shared)[shared_size], unsigned tid, bool sync = true
) {

  static_assert(shared_size >= nt * vt,
    "reg_to_shared_thread must have at least nt * vt storage");

  cudaArray<T, vt> x;
  cuda_thread_iterate<vt>([&](auto i, auto j) { 
    x[i] = shared[j];
  }, tid);

  if(sync) __syncthreads();

  return x;
}

template<unsigned nt, unsigned vt, typename T, unsigned shared_size>
__device__ void cuda_reg_to_shared_strided(
  cudaArray<T, vt> x, unsigned tid, T (&shared)[shared_size], bool sync = true
) {

  static_assert(shared_size >= nt * vt,
    "reg_to_shared_strided must have at least nt * vt storage");

  cuda_strided_iterate<nt, vt>(
    [&](auto i, auto j) { shared[j] = x[i]; }, tid
  );

  if(sync) __syncthreads();
}

template<unsigned nt, unsigned vt, typename T, unsigned shared_size>
__device__ auto cuda_shared_to_reg_strided(
  const T (&shared)[shared_size], int tid, bool sync = true
) {

  static_assert(shared_size >= nt * vt,
    "shared_to_reg_strided must have at least nt * vt storage");

  cudaArray<T, vt> x;
  cuda_strided_iterate<nt, vt>([&](auto i, auto j) { x[i] = shared[j]; }, tid);
  if(sync) __syncthreads();

  return x;
}

template<
  unsigned nt, unsigned vt, unsigned vt0 = vt, typename T, typename it_t, 
  unsigned shared_size
>
__device__ auto cuda_reg_to_mem_thread(
  cudaArray<T, vt> x, unsigned tid,
  unsigned count, it_t mem, T (&shared)[shared_size]
) {
  cuda_reg_to_shared_thread<nt>(x, tid, shared);
  auto y = cuda_shared_to_reg_strided<nt, vt>(shared, tid);
  cuda_reg_to_mem_strided<nt, vt, vt0>(y, tid, count, mem);
}

template<
  unsigned nt, unsigned vt, unsigned vt0 = vt, typename T, typename it_t, 
  unsigned shared_size>
__device__ auto cuda_mem_to_reg_thread(
  it_t mem, unsigned tid, unsigned count, T (&shared)[shared_size]
) {

  auto x = cuda_mem_to_reg_strided<nt, vt, vt0>(mem, tid, count);
  cuda_reg_to_shared_strided<nt, vt>(x, tid, shared);
  auto y = cuda_shared_to_reg_thread<nt, vt>(shared, tid);
  return y;
}

// ----------------------------------------------------------------------------
// cudaLoadStoreIterator
// ----------------------------------------------------------------------------

template<typename L, typename S, typename T, typename I>
struct cudaLoadStoreIterator : std::iterator_traits<const T*> {

  L load;
  S store;
  I base;

  cudaLoadStoreIterator(L load_, S store_, I base_) :
    load(load_), store(store_), base(base_) { }

  struct assign_t {
    L load;
    S store;
    I index;

    __device__ assign_t& operator=(T rhs) {
      static_assert(!std::is_same<S, cudaEmpty>::value, 
        "load_iterator is being stored to.");
      store(rhs, index);
      return *this;
    }
    __device__ operator T() const {
      static_assert(!std::is_same<L, cudaEmpty>::value,
        "store_iterator is being loaded from.");
      return load(index);
    }
  };

  __device__ assign_t operator[](I index) const {
    return assign_t { load, store, base + index };
  } 
  __device__ assign_t operator*() const {
    return assign_t { load, store, base };
  }

  __host__ __device__ cudaLoadStoreIterator operator+(I offset) const {
    cudaLoadStoreIterator cp = *this;
    cp += offset;
    return cp;
  }

  __host__ __device__ cudaLoadStoreIterator& operator+=(I offset) {
    base += offset;
    return *this;
  }

  __host__ __device__ cudaLoadStoreIterator operator-(I offset) const {
    cudaLoadStoreIterator cp = *this;
    cp -= offset;
    return cp;
  }

  __host__ __device__ cudaLoadStoreIterator& operator-=(I offset) {
    base -= offset;
    return *this;
  }
};

//template<typename T>
//struct trivial_load_functor {
//  template<typename I>
//  __host__ __device__ T operator()(I index) const {
//    return T();
//  }
//};

//template<typename T>
//struct trivial_store_functor {
//  template<typename I>
//  __host__ __device__ void operator()(T v, I index) const { }
//};

template <typename T, typename I = int, typename L, typename S>
auto cuda_make_load_store_iterator(L load, S store, I base = 0) {
  return cudaLoadStoreIterator<L, S, T, I>(load, store, base);
}

template <typename T, typename I = int, typename L>
auto cuda_make_load_iterator(L load, I base = 0) {
  return cuda_make_load_store_iterator<T>(load, cudaEmpty(), base);
}

template <typename T, typename I = int, typename S>
auto cuda_make_store_iterator(S store, I base = 0) {
  return cuda_make_load_store_iterator<T>(cudaEmpty(), store, base);
}

// ----------------------------------------------------------------------------
// launch kernel
// ----------------------------------------------------------------------------

template<typename F, typename... args_t>
__global__ void cuda_kernel(F f, args_t... args) {
  f(threadIdx.x, blockIdx.x, args...);
}

// ----------------------------------------------------------------------------
// operators
// ----------------------------------------------------------------------------

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

template<typename T>
struct cuda_maximum  : public std::binary_function<T, T, T> {
  __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};

template<typename T>
struct cuda_minimum  : public std::binary_function<T, T, T> {
  __device__ T operator()(T a, T b) const { return a < b ? a : b; }
};

// ----------------------------------------------------------------------------
// Launch parameters
// ----------------------------------------------------------------------------

template<unsigned NT, unsigned VT>
struct cudaExecutionPolicy {
  const static unsigned nt = NT;     
  const static unsigned vt = VT;
  const static unsigned nv = NT*VT;
};

inline constexpr cudaExecutionPolicy<512, 7> cudaDefaultExecutionPolicy{};



}  // end of namespace tf -----------------------------------------------------



