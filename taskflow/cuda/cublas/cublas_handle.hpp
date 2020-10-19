#pragma once

#include "cublas_error.hpp"
#include "../cuda_handle.hpp"

namespace tf {

// Function object class to create a cublas handle
struct cublasHandleCreator {
  cublasHandle_t operator () () const {
    cublasHandle_t handle;
    auto stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cublas initialization failed\n");
    }
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


