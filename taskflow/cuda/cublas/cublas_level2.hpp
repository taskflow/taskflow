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

// gemv
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
  return gemv(
    cublas_rtran<T>(trans), n, m, alpha, A, lda, x, incx, beta, y, incy
  );
}

// trmv
template <typename T>
cudaTask cublasFlowCapturer::trmv(
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
      stat = cublasStrmv(_handle, uplo, tran, diag, n, A, lda, x, incx);
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDtrmv(_handle, uplo, tran, diag, n, A, lda, x, incx);
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }

    TF_CHECK_CUBLAS(stat, "failed to capture trmv");
  });
}

// c_trmv
template <typename T>
cudaTask cublasFlowCapturer::c_trmv(
  cublasFillMode_t uplo,
  cublasOperation_t tran, cublasDiagType_t diag,
  int n, const T* A, int lda,
  T *x, int incx
) {
  return trmv(
    cublas_rfill(uplo), cublas_rtran<T>(tran), diag, n, A, lda, x, incx
  );
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
  return trsv(
    cublas_rfill(uplo), cublas_rtran<T>(tran), diag, n, A, lda, x, incx
  );
}

// symv
template <typename T>
cudaTask cublasFlowCapturer::symv(
  cublasFillMode_t uplo,
  int n,
  const T *alpha,
  const T *A, int lda,
  const T *x, int incx,
  const T *beta,
  T *y, int incy
) {
  return on([this, uplo, n, alpha, A, lda, x, incx, beta, y, incy] 
  (cudaStream_t stream) mutable {

    _stream(stream);

    cublasStatus_t stat;

    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSsymv(_handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDsymv(_handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }

    TF_CHECK_CUBLAS(stat, "failed to capture symv");
  });
}

// c_symv
template <typename T>
cudaTask cublasFlowCapturer::c_symv(
  cublasFillMode_t uplo,
  int n,
  const T *alpha,
  const T *A, int lda,
  const T *x, int incx,
  const T *beta,
  T *y, int incy
) {
  return symv(
    cublas_rfill(uplo), n, alpha, A, lda, x, incx, beta, y, incy
  );
}
    
// syr
template <typename T>
cudaTask cublasFlowCapturer::syr(
  cublasFillMode_t uplo,
  int n,
  const T *alpha,
  const T *x, int incx,
  T *A, int lda
) {

  return on([this, uplo, n, alpha, x, incx, A, lda] 
  (cudaStream_t stream) mutable {

    _stream(stream);

    cublasStatus_t stat;

    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSsyr(_handle, uplo, n, alpha, x, incx, A, lda);
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDsyr(_handle, uplo, n, alpha, x, incx, A, lda);
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }

    TF_CHECK_CUBLAS(stat, "failed to capture syr");
  });
}

// c_syr
template <typename T>
cudaTask cublasFlowCapturer::c_syr(
  cublasFillMode_t uplo,
  int n,
  const T *alpha,
  const T *x, int incx,
  T *A, int lda
) {
  return syr(
    cublas_rfill(uplo), n, alpha, x, incx, A, lda
  );
}

// syr2
template <typename T>
cudaTask cublasFlowCapturer::syr2(
  cublasFillMode_t uplo,
  int n,
  const T *alpha,
  const T *x, int incx,
  const T *y, int incy,
  T *A, int lda
) {

  return on([this, uplo, n, alpha, x, incx, y, incy, A, lda] 
  (cudaStream_t stream) mutable {

    _stream(stream);

    cublasStatus_t stat;

    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSsyr2(_handle, uplo, n, alpha, x, incx, y, incy, A, lda);
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDsyr2(_handle, uplo, n, alpha, x, incx, y, incy, A, lda);
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }

    TF_CHECK_CUBLAS(stat, "failed to capture syr2");
  });
}

// c_syr2
template <typename T>
cudaTask cublasFlowCapturer::c_syr2(
  cublasFillMode_t uplo,
  int n,
  const T *alpha,
  const T *x, int incx,
  const T *y, int incy,
  T *A, int lda
) {
  return syr2(
    cublas_rfill(uplo), n, alpha, x, incx, y, incy, A, lda
  );
}

}  // end of namespace tf -----------------------------------------------------

