#pragma once

#include "cublas_handle.hpp"

/** 
@file cublas_level2.hpp
*/

namespace tf {

// ----------------------------------------------------------------------------
// gemv
// ----------------------------------------------------------------------------

// Function: gemv
template <typename T>
void cublas_gemv(
  cublasHandle_t handle,
  cublasOperation_t trans,
  int m, int n,
  const T *alpha,
  const T *A, int lda,
  const T *x, int incx,
  const T *beta,
  T *y, int incy
) {
  cublasStatus_t stat;
  
  if constexpr(std::is_same_v<T, float>) {
    stat = cublasSgemv(handle,
      trans, m, n, alpha, A, lda, x, incx, beta, y, incy
    );
  }
  else if constexpr(std::is_same_v<T, double>) {
    stat = cublasDgemv(handle,
      trans, m, n, alpha, A, lda, x, incx, beta, y, incy
    );
  }
  else {
    static_assert(dependent_false_v<T>, "unknown cublas data type");
  }

  TF_CHECK_CUBLAS(stat, "failed to capture gemv");

}

// ----------------------------------------------------------------------------
// c_gemv
// ----------------------------------------------------------------------------

// Function: c_gemv
template <typename T>
void cublas_c_gemv(
  cublasHandle_t handle,
  cublasOperation_t trans,
  int m, int n,
  const T *alpha,
  const T *A, int lda,
  const T *x, int incx,
  const T *beta,
  T *y, int incy
) {
  cublasStatus_t stat;

  (trans == CUBLAS_OP_N) ? trans = CUBLAS_OP_T : trans = CUBLAS_OP_N;
  
  if constexpr(std::is_same_v<T, float>) {
    stat = cublasSgemv(handle,
      trans, n, m, alpha, A, lda, x, incx, beta, y, incy
    );
  }
  else if constexpr(std::is_same_v<T, double>) {
    stat = cublasDgemv(handle,
      trans, n, m, alpha, A, lda, x, incx, beta, y, incy
    );
  }
  else {
    static_assert(dependent_false_v<T>, "unknown cublas data type");
  }

  TF_CHECK_CUBLAS(stat, "failed to capture gemv");
}

// ---------------------------------------------------------------------------- 
// cublasFlowCapturere level-2 functions
// ---------------------------------------------------------------------------- 

template <typename T>
cudaTask cublasFlowCapturer::gemv(
  cublasOperation_t trans,
  int m, int n,
  const T *alpha,
  const T *A, int lda,
  const T *x, int incx,
  const T *beta,
  T *y, int incy
) {
  return on([this, trans, m, n, alpha, A, lda, x, incx, beta, y, incy] (cudaStream_t stream) mutable {
    _stream(stream);

    cublasStatus_t stat;
    
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSgemv(_handle,
        trans, m, n, alpha, A, lda, x, incx, beta, y, incy
      );
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDgemv(_handle,
        trans, m, n, alpha, A, lda, x, incx, beta, y, incy
      );
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }

    TF_CHECK_CUBLAS(stat, "failed to capture gemv");
  });
}

template <typename T>
cudaTask cublasFlowCapturer::c_gemv(
  cublasOperation_t trans,
  int m, int n,
  const T *alpha,
  const T *A, int lda,
  const T *x, int incx,
  const T *beta,
  T *y, int incy
) {
  return on([this, trans, m, n, alpha, A, lda, x, incx, beta, y, incy] (cudaStream_t stream) mutable {
    _stream(stream);

    cublasStatus_t stat;

    (trans == CUBLAS_OP_N) ? trans = CUBLAS_OP_T : trans = CUBLAS_OP_N;
    
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSgemv(_handle,
        trans, n, m, alpha, A, lda, x, incx, beta, y, incy
      );
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDgemv(_handle,
        trans, n, m, alpha, A, lda, x, incx, beta, y, incy
      );
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }

    TF_CHECK_CUBLAS(stat, "failed to capture gemv");
  });
}

}  // end of namespace tf -----------------------------------------------------
