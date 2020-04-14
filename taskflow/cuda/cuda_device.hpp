#pragma once

#include "cuda_error.hpp"

namespace tf { 

/**
@brief queries the number of available devices
*/
inline size_t cuda_num_devices() {
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
@brief cuda_dump_device_property
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

// ----------------------------------------------------------------------------
// Class definitions
// ----------------------------------------------------------------------------

/** @class cudaScopedDevice

@brief RAII-style device context switch

*/
class cudaScopedDevice {

  public:
    
    cudaScopedDevice(int);
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

}  // end of namespace cuda ---------------------------------------------------


