#pragma once

#include "cuda_device.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// memory
// ----------------------------------------------------------------------------

/** 
@brief queries the free memory (expensive call) 
*/
inline size_t cuda_get_free_mem(int d) {
  cudaScopedDevice ctx(d);
  size_t free, total;
  TF_CHECK_CUDA(
    cudaMemGetInfo(&free, &total), "failed to get mem info on device ", d
  );
  return free;
}

/** 
@brief queries the total available memory (expensive call) 
*/
inline size_t cuda_get_total_mem(int d) {
  cudaScopedDevice ctx(d);
  size_t free, total;
  TF_CHECK_CUDA(
    cudaMemGetInfo(&free, &total), "failed to get mem info on device ", d
  );
  return total;
}

/**
@brief allocates memory on the given device for holding @c N elements of type @c T

The function calls @c cudaMalloc to allocate <tt>N*sizeof(T)</tt> bytes of memory
on the given device @c d and returns a pointer to the starting address of 
the device memory.
*/
template <typename T>
T* cuda_malloc_device(size_t N, int d) {
  cudaScopedDevice ctx(d);
  T* ptr {nullptr};
  TF_CHECK_CUDA(
    cudaMalloc(&ptr, N*sizeof(T)), 
    "failed to allocate memory (", N*sizeof(T), "bytes) on device ", d
  )
  return ptr;
}

/**
@brief allocates memory on the current device associated with the caller

The function calls cuda_malloc_device from the current device associated
with the caller.
*/
template <typename T>
T* cuda_malloc_device(size_t N) {
  return cuda_malloc_device<T>(N, cuda_get_device());
}

/**
@brief allocates shared memory for holding @c N elements of type @c T

The function calls @c cudaMallocManaged to allocate <tt>N*sizeof(T)</tt> bytes
of memory and returns a pointer to the starting address of the shared memory.
*/
template <typename T>
T* cuda_malloc_shared(size_t N) {
  T* ptr {nullptr};
  TF_CHECK_CUDA(
    cudaMallocManaged(&ptr, N*sizeof(T)), 
    "failed to allocate shared memory (", N*sizeof(T), "bytes)"
  )
  return ptr;
}

/**
@brief frees cuda memory on the given device
*/
template <typename T>
void cuda_free(T* ptr, int d) {
  cudaScopedDevice ctx(d);
  TF_CHECK_CUDA(cudaFree(ptr), "failed to free memory ", ptr);
}

/**
@brief frees cuda memory on the current device associated with the caller
*/
template <typename T>
void cuda_free(T* ptr) {
  cuda_free(ptr, cuda_get_device());
}

}  // end of namespace tf -----------------------------------------------------






