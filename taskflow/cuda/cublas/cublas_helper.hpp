#pragma once

#include "cublas_handle.hpp"

namespace tf {

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


