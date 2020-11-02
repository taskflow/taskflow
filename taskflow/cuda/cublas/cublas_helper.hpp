#pragma once

#include "cublas_handle.hpp"

/** 
@file cublas_helper.hpp
*/

namespace tf {

/**
@brief copies vector data from host to device

This method copies @c n elements from a vector @c h in host memory space 
to a vector @c d in GPU memory space. 
The storage spacing between consecutive elements is given by @c inch for 
the source vector @c h and by @c incd for the destination vector @c d.

@tparam T data type
@param stream stream to associate with this copy operation
@param n number of elements
@param d target device pointer
@param incd spacing between consecutive elements in @c d
@param h source host pointer
@param inch spacing between consecutive elements in @c h

*/
template <typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
>
void cublas_vset_async(
  cudaStream_t stream, size_t n, const T* h, int inch, T* d, int incd
) {
  TF_CHECK_CUBLAS(
    cublasSetVectorAsync(n, sizeof(T), h, inch, d, incd, stream),
    "failed to run vset_async"
  );
}

/**
@brief copies vector data from device to host

This method copies @c n elements from a vector @c d in GPU memory space 
to a vector @c h in host memory space. 
The storage spacing between consecutive elements is given by @c inch for 
the target vector @c h and by @c incd for the source vector @c d.

@tparam T data type
@param stream stream to associate with this copy operation
@param n number of elements
@param h target host pointer
@param inch spacing between consecutive elements in @c h
@param d source device pointer
@param incd spacing between consecutive elements in @c d

*/
template <typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
>
void cublas_vget_async(
  cudaStream_t stream, size_t n, const T* d, int incd, T* h, int inch
) {
  TF_CHECK_CUBLAS(
    cublasGetVectorAsync(n, sizeof(T), d, incd, h, inch, stream),
    "failed to run vget_async"
  );
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


