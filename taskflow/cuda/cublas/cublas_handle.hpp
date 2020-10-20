#pragma once

#include "cublas_error.hpp"
#include "../cuda_handle.hpp"

namespace tf {

// Function object class to create a cublas handle
struct cublasHandleCreator {
  cublasHandle_t operator () () const {
    cublasHandle_t handle;
    TF_CHECK_CUBLAS(
      cublasCreate(&handle), "failed to create a cublas handle"
    );
    TF_CHECK_CUBLAS(
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE),
      "failed to set cublas pointer mode on device"
    );
    std::cout << "create cublas handle " << handle << '\n';
    return handle;
  }
};

// Function object class to delete a cublas handle.
struct cublasHandleDeleter {
  void operator () (cublasHandle_t ptr) const {
    std::cout << "destroy cublas handle " << ptr << '\n';
    cublasDestroy(ptr);
  }
};

/**
@brief per thread cuBLAS handle pool
*/
inline thread_local cudaPerThreadHandlePool<
  cublasHandle_t, cublasHandleCreator, cublasHandleDeleter
> cublas_per_thread_handle_pool;



}  // end of namespace tf -----------------------------------------------------


