#pragma once

#include "cuda_error.hpp"

namespace tf { 

/**
@brief queries the number of available devices
*/
inline unsigned cuda_num_devices() {
	int N = 0;
  TF_CHECK_CUDA(cudaGetDeviceCount(&N), "failed to get device count");
	return N;
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


