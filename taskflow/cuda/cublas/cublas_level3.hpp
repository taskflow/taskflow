#pragma once

#include "cublas_handle.hpp"

namespace tf {

// ---------------------------------------------------------------------------- 
// cublasFlowCapturere level-3 functions
// ---------------------------------------------------------------------------- 

// Function: geam
template <typename T>
cudaTask cublasFlowCapturer::geam(
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n,
  const T *alpha,
  const T *A, int lda,
  const T *beta,
  const T *B, int ldb,
  T *C, int ldc
) {
  return on([this, ta, tb, m, n, alpha, A, lda, beta, B, ldb, C, ldc] 
  (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSgeam(_handle,
        ta, tb, m, n, alpha, A, lda, beta, B, ldb, C, ldc
      );
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDgeam(_handle,
        ta, tb, m, n, alpha, A, lda, beta, B, ldb, C, ldc
      );
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }
    TF_CHECK_CUBLAS(stat, "failed to run geam");
  });
}

// Function: c_geam
template <typename T>
cudaTask cublasFlowCapturer::c_geam(
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n,
  const T *alpha,
  const T *A, int lda,
  const T *beta,
  const T *B, int ldb,
  T *C, int ldc
) {
  return on([this, ta, tb, m, n, alpha, A, lda, beta, B, ldb, C, ldc] 
  (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSgeam(_handle,
        ta, tb, n, m, alpha, A, lda, beta, B, ldb, C, ldc
      );
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDgeam(_handle,
        ta, tb, n, m, alpha, A, lda, beta, B, ldb, C, ldc
      );
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }
    TF_CHECK_CUBLAS(stat, "failed to run c_geam");
  });
}

// Function: gemm
template <typename T>
cudaTask cublasFlowCapturer::gemm(
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A, int lda,
  const T *B, int ldb,
  const T *beta,
  T *C, int ldc
) {
  return on([this, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc] 
  (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSgemm(_handle,
        ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
      );
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDgemm(_handle,
        ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
      );
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }
    TF_CHECK_CUBLAS(stat, "failed to run gemm");
  });
}
    
template <typename T>
cudaTask cublasFlowCapturer::c_gemm(
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A, int lda,
  const T *B, int ldb,
  const T *beta,
  T *C, int ldc
) {
  return on([this, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc] 
  (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSgemm(_handle,
        tb, ta, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc
      );
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDgemm(_handle,
        tb, ta, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc
      );
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }
    TF_CHECK_CUBLAS(stat, "failed to run c_gemm");
  });
}
    
// Function: gemm_batched
template <typename T>
cudaTask cublasFlowCapturer::gemm_batched(
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A[], int lda,
  const T *B[], int ldb,
  const T *beta,
  T *C[], int ldc,
  int bc
) {
  return on([this, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bc] 
  (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSgemmBatched(_handle,
        ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bc
      );
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDgemmBatched(_handle,
        ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bc
      );
    }
    else static_assert(dependent_false_v<T>, "unknown cublas data type");
    TF_CHECK_CUBLAS(stat, "failed to run gemm_batched");
  });
}

// Function: c_gemm_batched
template <typename T>
cudaTask cublasFlowCapturer::c_gemm_batched(
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A[], int lda,
  const T *B[], int ldb,
  const T *beta,
  T *C[], int ldc,
  int bc
) {
  return on([this, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bc] 
  (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSgemmBatched(_handle,
        tb, ta, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, bc
      );
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDgemmBatched(_handle,
        tb, ta, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, bc
      );
    }
    else static_assert(dependent_false_v<T>, "unknown cublas data type");
    TF_CHECK_CUBLAS(stat, "failed to run c_gemm_batched");
  });
}

// Function: gemm_sbatched (strided)    
template <typename T>
cudaTask cublasFlowCapturer::gemm_sbatched(
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A, int lda, long long int sA,
  const T *B, int ldb, long long int sB,
  const T *beta,
  T *C, int ldc, long long int sC,
  int bc
) {
  return on([this, ta, tb, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, bc] 
  (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSgemmStridedBatched(_handle,
        ta, tb, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, bc
      );
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDgemmStridedBatched(_handle,
        ta, tb, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, bc
      );
    }
    else static_assert(dependent_false_v<T>, "unknown cublas data type");
    TF_CHECK_CUBLAS(stat, "failed to run gemm_sbatched");
  });
}

// Function: c_gemm_sbatched (strided)    
template <typename T>
cudaTask cublasFlowCapturer::c_gemm_sbatched(
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A, int lda, long long int sA,
  const T *B, int ldb, long long int sB,
  const T *beta,
  T *C, int ldc, long long int sC,
  int bc
){
  return on(
  [this, ta, tb, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, bc] 
  (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSgemmStridedBatched(_handle,
        tb, ta, n, m, k, alpha, B, ldb, sB, A, lda, sA, beta, C, ldc, sC, bc
      );
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDgemmStridedBatched(_handle,
        tb, ta, n, m, k, alpha, B, ldb, sB, A, lda, sA, beta, C, ldc, sC, bc
      );
    }
    else static_assert(dependent_false_v<T>, "unknown cublas data type");
    TF_CHECK_CUBLAS(stat, "failed to run c_gemm_sbatched");
  });
}

}  // end of namespace tf -----------------------------------------------------

