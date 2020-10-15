#pragma once

#include "cuda_statgrab.hpp"

namespace tf {

/** @class cudaScopedDevice

@brief RAII-styled device context switcher

*/
class cudaScopedDevice {

  public:
    
    /**
    @brief constructs a RAII-styled device switcher
    */
    cudaScopedDevice(int d);

    /**
    @brief destructs the guard and returns back to the original device
    */
    ~cudaScopedDevice();

  private:

    int _p;
};

// Constructor
inline cudaScopedDevice::cudaScopedDevice(int dev) { 
  TF_CHECK_CUDA(cudaGetDevice(&_p), "failed to get current device scope");
  if(_p == dev) {
    _p = -1;
  }
  else {
    TF_CHECK_CUDA(cudaSetDevice(dev), "failed to scope on device ", dev);
  }
}

// Destructor
inline cudaScopedDevice::~cudaScopedDevice() { 
  if(_p != -1) {
    cudaSetDevice(_p);
    //TF_CHECK_CUDA(cudaSetDevice(_p), "failed to scope back to device ", _p);
  }
}

// ----------------------------------------------------------------------------
// memory
// ----------------------------------------------------------------------------

// get the free memory (expensive call)
inline size_t cuda_get_free_mem(int d) {
  cudaScopedDevice ctx(d);
  size_t free, total;
  TF_CHECK_CUDA(
    cudaMemGetInfo(&free, &total), "failed to get mem info on device ", d
  );
  return free;
}

// get the free memory (expensive call)
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
inline T* cuda_malloc_device(size_t N, int d) {
  cudaScopedDevice ctx(d);
  T* ptr {nullptr};
  TF_CHECK_CUDA(
    cudaMalloc(&ptr, N*sizeof(T)), 
    "failed to allocate memory (", N*sizeof(T), "bytes) on device ", d
  )
  return ptr;
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
@brief frees cuda memory 
*/
template <typename T>
void cuda_free(T* ptr) {
  TF_CHECK_CUDA(cudaFree(ptr), "failed to free memory ", ptr);
}

}  // end of namespace tf -----------------------------------------------------






