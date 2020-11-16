#pragma once

#include "cublas_handle.hpp"

namespace tf {

// ---------------------------------------------------------------------------- 
// global utility functions
// ---------------------------------------------------------------------------- 
// find the tranposed op
template <typename T, std::enable_if<
  std::is_same_v<T, float> && std::is_same_v<T, double>, void>* = nullptr
>
constexpr cublasOperation_t cublas_rtran(cublasOperation_t op) {
  if(op != CUBLAS_OP_N && op != CUBLAS_OP_T) {
    TF_THROW("invalid transposition op for floating data types"); 
  }
  return (op == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N;
}

// find the transposed fill
constexpr cublasFillMode_t cublas_rfill(cublasFillMode_t uplo) {
  switch(uplo) {
    case CUBLAS_FILL_MODE_LOWER: return CUBLAS_FILL_MODE_UPPER;
    case CUBLAS_FILL_MODE_UPPER: return CUBLAS_FILL_MODE_LOWER;
    default: return uplo;
  }
}

// find the transposed side
constexpr cublasSideMode_t cublas_rside(cublasSideMode_t side) {
  switch(side) {
    case CUBLAS_SIDE_LEFT : return CUBLAS_SIDE_RIGHT;
    case CUBLAS_SIDE_RIGHT: return CUBLAS_SIDE_LEFT;
    default: return side;
  }
}

// ---------------------------------------------------------------------------- 
// cublasFlowCapturer helper functions
// ---------------------------------------------------------------------------- 

// Function: vset
template <typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>*
>
cudaTask cublasFlowCapturer::vset(
  size_t n, const T* h, int inch, T* d, int incd
) {
  return on([n, h, inch, d, incd] (cudaStream_t stream) mutable {
    TF_CHECK_CUBLAS(
      cublasSetVectorAsync(n, sizeof(T), h, inch, d, incd, stream),
      "failed to run vset_async"
    );
  });
}

// Function: vget 
template <typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>*
>
cudaTask cublasFlowCapturer::vget(size_t n, const T* d, int incd, T* h, int inch) {
  return on([n, d, incd, h, inch] (cudaStream_t stream) mutable {
    TF_CHECK_CUBLAS(
      cublasGetVectorAsync(n, sizeof(T), d, incd, h, inch, stream),
      "failed to run vget_async"
    );
  });
}

}  // end of namespace tf -----------------------------------------------------


