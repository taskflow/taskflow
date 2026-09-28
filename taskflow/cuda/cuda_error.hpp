#pragma once

#include <cuda.h>
#include <iostream>
#include <sstream>
#include <exception>

#include "../utility/stream.hpp"

#define TF_CUDA_EXPAND( x ) x
#define TF_CUDA_REMOVE_FIRST_HELPER(N, ...) __VA_ARGS__
#define TF_CUDA_REMOVE_FIRST(...) TF_CUDA_EXPAND(TF_CUDA_REMOVE_FIRST_HELPER(__VA_ARGS__))
#define TF_CUDA_GET_FIRST_HELPER(N, ...) N
#define TF_CUDA_GET_FIRST(...) TF_CUDA_EXPAND(TF_CUDA_GET_FIRST_HELPER(__VA_ARGS__))

#define TF_CHECK_CUDA(...)                                       \
if(TF_CUDA_GET_FIRST(__VA_ARGS__) != cudaSuccess) {              \
  std::ostringstream oss;                                        \
  auto __ev__ = TF_CUDA_GET_FIRST(__VA_ARGS__);                  \
  oss << "[" << __FILE__ << ":" << __LINE__ << "] "              \
      << (cudaGetErrorString(__ev__)) << " ("                    \
      << (cudaGetErrorName(__ev__)) << ") - ";                   \
  tf::ostreamize(oss, TF_CUDA_REMOVE_FIRST(__VA_ARGS__));        \
  throw std::runtime_error(oss.str());                           \
}

#if __CUDACC_VER_MAJOR__ >= 13
#define TF_CUDA_POST13(X) X
#define TF_CUDA_PRE13(X)
#else
#define TF_CUDA_PRE13(X) X
#define TF_CUDA_POST13(X)
#endif

namespace tf {

/**
@struct cudaProxyDeleter

@brief deleter that forwards a CUDA handle to the destroy function @c F

@tparam F CUDA runtime function that destroys a handle (e.g., @c cudaEventDestroy)

This stateless deleter lets `std::unique_ptr` manage a CUDA handle without
storing a function pointer, for example,
`std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, cudaProxyDeleter<cudaEventDestroy>>`.
*/
template <auto F>
struct cudaProxyDeleter {

  /**
  @brief destroys the given CUDA handle by calling @c F
  */
  void operator()(auto handle) const noexcept {
    F(handle);
  }
};

}  // end of namespace tf -----------------------------------------------------
