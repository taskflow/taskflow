#pragma once

#include "cublas_handle.hpp"

namespace tf {

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
  return on([this, trans, m, n, alpha, A, lda, x, incx, beta, y, incy] 
  (cudaStream_t stream) mutable {
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
  return on([this, trans, m, n, alpha, A, lda, x, incx, beta, y, incy] 
  (cudaStream_t stream) mutable {
    _stream(stream);

    cublasStatus_t stat;

    trans = cublas_transpose_tran<T>(trans); 
    
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

    TF_CHECK_CUBLAS(stat, "failed to capture c_gemv");
  });
}

// trsv
template <typename T>
cudaTask cublasFlowCapturer::trsv(
  cublasFillMode_t uplo,
  cublasOperation_t tran, cublasDiagType_t diag,
  int n, const T* A, int lda,
  T *x, int incx
) {
  return on([this, uplo, tran, diag, n, A, lda, x, incx] 
  (cudaStream_t stream) mutable {

    _stream(stream);

    cublasStatus_t stat;

    if constexpr(std::is_same_v<T, float>) {
      stat = cublasStrsv(_handle, uplo, tran, diag, n, A, lda, x, incx);
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDtrsv(_handle, uplo, tran, diag, n, A, lda, x, incx);
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }

    TF_CHECK_CUBLAS(stat, "failed to capture trsv");
  });
}

// c_trsv
template <typename T>
cudaTask cublasFlowCapturer::c_trsv(
  cublasFillMode_t uplo,
  cublasOperation_t tran, cublasDiagType_t diag,
  int n, const T* A, int lda,
  T *x, int incx
) {
  return on([this, uplo, tran, diag, n, A, lda, x, incx] 
  (cudaStream_t stream) mutable {

    _stream(stream);

    cublasStatus_t stat;

    tran = cublas_transpose_tran<T>(tran);
    uplo = cublas_transpose_fill(uplo);
    
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasStrsv(_handle, uplo, tran, diag, n, A, lda, x, incx);
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDtrsv(_handle, uplo, tran, diag, n, A, lda, x, incx);
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }

    TF_CHECK_CUBLAS(stat, "failed to capture c_trsv");
  });
}

}  // end of namespace tf -----------------------------------------------------

