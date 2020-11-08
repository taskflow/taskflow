#pragma once

#include "cuda_error.hpp"

/**
@file cuda_device.hpp
@brief CUDA device utilities include file
*/

namespace tf { 

/**
@brief queries the number of available devices
*/
inline size_t cuda_get_num_devices() {
	int N = 0;
  TF_CHECK_CUDA(cudaGetDeviceCount(&N), "failed to get device count");
	return static_cast<size_t>(N);
}

/**
@brief gets the current device associated with the caller thread
*/
inline int cuda_get_device() {
  int id;
  TF_CHECK_CUDA(cudaGetDevice(&id), "failed to get current device id");
	return id;
}

/**
@brief switches to a given device context
*/
inline void cuda_set_device(int id) {
  TF_CHECK_CUDA(cudaSetDevice(id), "failed to switch to device ", id);
}
    
/**    
@brief obtains the device property 
*/
inline void cuda_get_device_property(int i, cudaDeviceProp& p) {
  TF_CHECK_CUDA(
    cudaGetDeviceProperties(&p, i), "failed to get property of device ", i
  );
}

/**    
@brief obtains the device property 
*/
inline cudaDeviceProp cuda_get_device_property(int i) {
  cudaDeviceProp p;
  TF_CHECK_CUDA(
    cudaGetDeviceProperties(&p, i), "failed to get property of device ", i
  );
  return p;
}

/**
@brief dumps the device property
*/
inline void cuda_dump_device_property(std::ostream& os, const cudaDeviceProp& p) {

  os << "Major revision number:         " << p.major << '\n'
     << "Minor revision number:         " << p.minor << '\n'
     << "Name:                          " << p.name  << '\n'
     << "Total global memory:           " << p.totalGlobalMem << '\n'
     << "Total shared memory per block: " << p.sharedMemPerBlock << '\n'
     << "Total registers per block:     " << p.regsPerBlock << '\n'
     << "Warp size:                     " << p.warpSize << '\n'
     << "Maximum memory pitch:          " << p.memPitch << '\n'
     << "Maximum threads per block:     " << p.maxThreadsPerBlock << '\n';

  os << "Maximum dimension of block:    ";
  for (int i = 0; i < 3; ++i) {
    if(i) os << 'x';
    os << p.maxThreadsDim[i];
  }
  os << '\n';

  os << "Maximum dimenstion of grid:    ";
  for (int i = 0; i < 3; ++i) {
    if(i) os << 'x';
    os << p.maxGridSize[i];;
  }
  os << '\n';

  os << "Clock rate:                    " << p.clockRate << '\n'
     << "Total constant memory:         " << p.totalConstMem << '\n'
     << "Texture alignment:             " << p.textureAlignment << '\n'
     << "Concurrent copy and execution: " << p.deviceOverlap << '\n'
     << "Number of multiprocessors:     " << p.multiProcessorCount << '\n'
     << "Kernel execution timeout:      " << p.kernelExecTimeoutEnabled << '\n'
     << "GPU sharing Host Memory:       " << p.integrated << '\n'
     << "Host page-locked mem mapping:  " << p.canMapHostMemory << '\n'
     << "Alignment for Surfaces:        " << p.surfaceAlignment << '\n'
     << "Device has ECC support:        " << p.ECCEnabled << '\n'
     << "Unified Addressing (UVA):      " << p.unifiedAddressing << '\n';
}

/**
@brief queries the maximum threads per block on a device
*/
inline size_t cuda_get_device_max_threads_per_block(int d) {
  int threads = 0;
  TF_CHECK_CUDA(
    cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerBlock, d),
    "failed to query the maximum threads per block on device ", d
  )
  return threads;
}

/**
@brief queries the maximum x-dimension per block on a device
*/
inline size_t cuda_get_device_max_x_dim_per_block(int d) {
  int dim = 0;
  TF_CHECK_CUDA(
    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxBlockDimX, d),
    "failed to query the maximum x-dimension per block on device ", d
  )
  return dim;
}

/**
@brief queries the maximum y-dimension per block on a device
*/
inline size_t cuda_get_device_max_y_dim_per_block(int d) {
  int dim = 0;
  TF_CHECK_CUDA(
    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxBlockDimY, d),
    "failed to query the maximum y-dimension per block on device ", d
  )
  return dim;
}

/**
@brief queries the maximum z-dimension per block on a device
*/
inline size_t cuda_get_device_max_z_dim_per_block(int d) {
  int dim = 0;
  TF_CHECK_CUDA(
    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxBlockDimZ, d),
    "failed to query the maximum z-dimension per block on device ", d
  )
  return dim;
}

/**
@brief queries the maximum x-dimension per grid on a device
*/
inline size_t cuda_get_device_max_x_dim_per_grid(int d) {
  int dim = 0;
  TF_CHECK_CUDA(
    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimX, d),
    "failed to query the maximum x-dimension per grid on device ", d
  )
  return dim;
}

/**
@brief queries the maximum y-dimension per grid on a device
*/
inline size_t cuda_get_device_max_y_dim_per_grid(int d) {
  int dim = 0;
  TF_CHECK_CUDA(
    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimY, d),
    "failed to query the maximum y-dimension per grid on device ", d
  )
  return dim;
}

/**
@brief queries the maximum z-dimension per grid on a device
*/
inline size_t cuda_get_device_max_z_dim_per_grid(int d) {
  int dim = 0;
  TF_CHECK_CUDA(
    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimZ, d),
    "failed to query the maximum z-dimension per grid on device ", d
  )
  return dim;
}

/**
@brief queries the maximum shared memory size in bytes per block on a device
*/
inline size_t cuda_get_device_max_shm_per_block(int d) {
  int num = 0;
  TF_CHECK_CUDA(
    cudaDeviceGetAttribute(&num, cudaDevAttrMaxSharedMemoryPerBlock, d),
    "failed to query the maximum shared memory per block on device ", d
  )
  return num;
}

/**
@brief queries the warp size on a device
*/
inline size_t cuda_get_device_warp_size(int d) {
  int num = 0;
  TF_CHECK_CUDA(
    cudaDeviceGetAttribute(&num, cudaDevAttrWarpSize, d),
    "failed to query the warp size per block on device ", d
  )
  return num;
}

/**
@brief queries the major number of compute capability of a device
*/
inline int cuda_get_device_compute_capability_major(int d) {
  int num = 0;
  TF_CHECK_CUDA(
    cudaDeviceGetAttribute(&num, cudaDevAttrComputeCapabilityMajor, d),
    "failed to query the major number of compute capability of device ", d
  )
  return num;
}

/**
@brief queries the minor number of compute capability of a device
*/
inline int cuda_get_device_compute_capability_minor(int d) {
  int num = 0;
  TF_CHECK_CUDA(
    cudaDeviceGetAttribute(&num, cudaDevAttrComputeCapabilityMinor, d),
    "failed to query the minor number of compute capability of device ", d
  )
  return num;
}

/**
@brief queries if the device supports unified addressing
*/
inline bool cuda_get_device_unified_addressing(int d) {
  int num = 0;
  TF_CHECK_CUDA(
    cudaDeviceGetAttribute(&num, cudaDevAttrUnifiedAddressing, d),
    "failed to query unified addressing status on device ", d
  )
  return num;
}

// ----------------------------------------------------------------------------
// CUDA Version
// ----------------------------------------------------------------------------

/**
@brief queries the latest CUDA version (1000 * major + 10 * minor) supported by the driver 
*/
inline int cuda_get_driver_version() {
  int num = 0;
  TF_CHECK_CUDA(
    cudaDriverGetVersion(&num), 
    "failed to query the latest cuda version supported by the driver"
  );
  return num;
}

/**
@brief queries the CUDA Runtime version (1000 * major + 10 * minor)
*/
inline int cuda_get_runtime_version() {
  int num = 0;
  TF_CHECK_CUDA(
    cudaRuntimeGetVersion(&num), "failed to query cuda runtime version"
  );
  return num;
}

// ----------------------------------------------------------------------------
// cudaScopedDevice
// ----------------------------------------------------------------------------

/** @class cudaScopedDevice

@brief RAII-styled device context switch

Sample usage:
    
@code{.cpp}
{
  tf::cudaScopedDevice device(1);  // switch to the device context 1

  // create a stream under device context 1
  cudaStream_t stream;
  cudaStreamCreate(&stream);

}  // leaving the scope and goes back to the previous device context
@endcode

%cudaScopedDevice is neither movable nor copyable.
*/
class cudaScopedDevice {

  public:
    
    /**
    @brief constructs a RAII-styled device switcher

    @param device device context to scope in the guard
    */
    explicit cudaScopedDevice(int device);

    /**
    @brief destructs the guard and switches back to the previous device context
    */
    ~cudaScopedDevice();

  private:
    
    cudaScopedDevice() = delete;
    cudaScopedDevice(const cudaScopedDevice&) = delete;
    cudaScopedDevice(cudaScopedDevice&&) = delete;

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

}  // end of namespace cuda ---------------------------------------------------





