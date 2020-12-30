#pragma once

#include "cuda_device.hpp"

/**
@file cuda_memory.hpp
@brief CUDA memory utilities include file
*/

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
@brief frees memory on the GPU device

@tparam T pointer type
@param ptr device pointer to memory to free
@param d device context identifier

This methods call @c cudaFree to free the memory space pointed to by @c ptr
using the given device context.
*/
template <typename T>
void cuda_free(T* ptr, int d) {
  cudaScopedDevice ctx(d);
  TF_CHECK_CUDA(cudaFree(ptr), "failed to free memory ", ptr, " on GPU ", d);
}

/**
@brief frees memory on the GPU device

@tparam T pointer type
@param ptr device pointer to memory to free

This methods call @c cudaFree to free the memory space pointed to by @c ptr
using the current device context of the caller.
*/
template <typename T>
void cuda_free(T* ptr) {
  cuda_free(ptr, cuda_get_device());
}

/**
@brief copies data between host and device asynchronously through a stream

@param stream stream identifier
@param dst destination memory address
@param src source memory address
@param count size in bytes to copy

The method calls @c cudaMemcpyAsync with the given @c stream
using @c cudaMemcpyDefault to infer the memory space of the source and 
the destination pointers. The memory areas may not overlap. 
*/
inline void cuda_memcpy_async(
  cudaStream_t stream, void* dst, const void* src, size_t count
) {
  TF_CHECK_CUDA(
    cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream),
    "failed to perform cudaMemcpyAsync"
  );
}

/**
@brief initializes or sets GPU memory to the given value byte by byte

@param stream stream identifier
@param devPtr pointer to GPU mempry
@param value value to set for each byte of the specified memory
@param count size in bytes to set

The method calls @c cudaMemsetAsync with the given @c stream
to fill the first @c count bytes of the memory area pointed to by @c devPtr 
with the constant byte value @c value.
*/
inline void cuda_memset_async(
  cudaStream_t stream, void* devPtr, int value, size_t count
){
  TF_CHECK_CUDA(
    cudaMemsetAsync(devPtr, value, count, stream),
    "failed to perform cudaMemsetAsync"
  );
}

// ----------------------------------------------------------------------------
// Shared Memory
// ----------------------------------------------------------------------------
//
// Because dynamically sized shared memory arrays are declared "extern",
// we can't templatize them directly.  To get around this, we declare a
// simple wrapper struct that will declare the extern array with a different
// name depending on the type.  This avoids compiler errors about duplicate
// definitions.
//
// To use dynamically allocated shared memory in a templatized __global__ or
// __device__ function, just replace code like this:
//
//  template<class T>
//  __global__ void
//  foo( T* g_idata, T* g_odata)
//  {
//      // Shared mem size is determined by the host app at run time
//      extern __shared__  T sdata[];
//      ...
//      doStuff(sdata);
//      ...
//   }
//
//  With this:
//
//  template<class T>
//  __global__ void
//  foo( T* g_idata, T* g_odata)
//  {
//      // Shared mem size is determined by the host app at run time
//      cudaSharedMemory<T> smem;
//      T* sdata = smem.get();
//      ...
//      doStuff(sdata);
//      ...
//   }
// ----------------------------------------------------------------------------

// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
/**
@private
*/
template <typename T>
struct cudaSharedMemory
{
  // Ensure that we won't compile any un-specialized types
  __device__ T *get()
  {
    extern __device__ void error(void);
    error();
    return NULL;
  }
};

// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double
// One could also specialize it for user-defined types.

/**
@private
*/
template <>
struct cudaSharedMemory <int>
{
  __device__ int *get()
  {
    extern __shared__ int s_int[];
    return s_int;
  }
};

/**
@private
*/
template <>
struct cudaSharedMemory <unsigned int>
{
  __device__ unsigned int *get()
  {
    extern __shared__ unsigned int s_uint[];
    return s_uint;
  }
};

/**
@private
*/
template <>
struct cudaSharedMemory <char>
{
  __device__ char *get()
  {
    extern __shared__ char s_char[];
    return s_char;
  }
};

/**
@private
*/
template <>
struct cudaSharedMemory <unsigned char>
{
  __device__ unsigned char *get()
  {
    extern __shared__ unsigned char s_uchar[];
    return s_uchar;
  }
};

/**
@private
*/
template <>
struct cudaSharedMemory <short>
{
  __device__ short *get()
  {
    extern __shared__ short s_short[];
    return s_short;
  }
};

/**
@private
*/
template <>
struct cudaSharedMemory <unsigned short>
{
  __device__ unsigned short *get()
  {
    extern __shared__ unsigned short s_ushort[];
    return s_ushort;
  }
};

/**
@private
*/
template <>
struct cudaSharedMemory <long>
{
  __device__ long *get()
  {
    extern __shared__ long s_long[];
    return s_long;
  }
};

/**
@private
*/
template <>
struct cudaSharedMemory <unsigned long>
{
  __device__ unsigned long *get()
  {
    extern __shared__ unsigned long s_ulong[];
    return s_ulong;
  }
};

//template <>
//struct cudaSharedMemory <size_t>
//{
//  __device__ size_t *get()
//  {
//    extern __shared__ size_t s_sizet[];
//    return s_sizet;
//  }
//};

/**
@private
*/
template <>
struct cudaSharedMemory <bool>
{
  __device__ bool *get()
  {
    extern __shared__ bool s_bool[];
    return s_bool;
  }
};

/**
@private
*/
template <>
struct cudaSharedMemory <float>
{
  __device__ float *get()
  {
    extern __shared__ float s_float[];
    return s_float;
  }
};

/**
@private
*/
template <>
struct cudaSharedMemory <double>
{
  __device__ double *get()
  {
    extern __shared__ double s_double[];
    return s_double;
  }
};

}  // end of namespace tf -----------------------------------------------------






