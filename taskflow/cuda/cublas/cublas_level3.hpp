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
  return geam(
    ta, tb, n, m, alpha, A, lda, beta, B, ldb, C, ldc
  );
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
  return gemm(
    tb, ta, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc
  );
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
  return gemm_batched(
    tb, ta, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, bc
  );
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
  return gemm_sbatched(
    tb, ta, n, m, k, alpha, B, ldb, sB, A, lda, sA, beta, C, ldc, sC, bc
  );
}

// symm    
template <typename T>
cudaTask cublasFlowCapturer::symm(
  cublasSideMode_t side, cublasFillMode_t uplo,
  int m, int n,
  const T *alpha,
  const T *A, int lda,
  const T *B, int ldb,
  const T *beta,
  T *C, int ldc
) {
  return on(
  [this, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc] 
  (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSsymm(_handle,
        side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc
      );
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDsymm(_handle,
        side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc
      );
    }
    else static_assert(dependent_false_v<T>, "unknown cublas data type");
    TF_CHECK_CUBLAS(stat, "failed to run symm");
  });
}

// c_symm    
template <typename T>
cudaTask cublasFlowCapturer::c_symm(
  cublasSideMode_t side, cublasFillMode_t uplo,
  int m, int n,
  const T *alpha,
  const T *A, int lda,
  const T *B, int ldb,
  const T *beta,
  T *C, int ldc
) {
  return symm(
    cublas_rside(side), cublas_rfill(uplo),
    n, m, alpha, A, lda, B, ldb, beta, C, ldc
  );
}
    
// syrk
template <typename T>
cudaTask cublasFlowCapturer::syrk(
  cublasFillMode_t uplo, cublasOperation_t tran,
  int n, int k,
  const T *alpha,
  const T *A, int lda,
  const T *beta,
  T *C, int ldc
) {
  return on(
  [this, uplo, tran, n, k, alpha, A, lda, beta, C, ldc] 
  (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSsyrk(_handle,
        uplo, tran, n, k, alpha, A, lda, beta, C, ldc
      );
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDsyrk(_handle,
        uplo, tran, n, k, alpha, A, lda, beta, C, ldc
      );
    }
    else static_assert(dependent_false_v<T>, "unknown cublas data type");
    TF_CHECK_CUBLAS(stat, "failed to run syrk");
  });
}

// c_syrk
template <typename T>
cudaTask cublasFlowCapturer::c_syrk(
  cublasFillMode_t uplo, cublasOperation_t tran,
  int n, int k,
  const T *alpha,
  const T *A, int lda,
  const T *beta,
  T *C, int ldc
) {
  return syrk(
    cublas_rfill(uplo), cublas_rtran<T>(tran), 
    n, k, alpha, A, lda, beta, C, ldc
  );
}

// syr2k
template <typename T>
cudaTask cublasFlowCapturer::syr2k(
  cublasFillMode_t uplo, cublasOperation_t tran,
  int n, int k,
  const T *alpha,
  const T *A, int lda,
  const T *B, int ldb,
  const T *beta,
  T *C, int ldc
) {
  return on(
  [this, uplo, tran, n, k, alpha, A, lda, B, ldb, beta, C, ldc] 
  (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSsyr2k(_handle,
        uplo, tran, n, k, alpha, A, lda, B, ldb, beta, C, ldc
      );
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDsyr2k(_handle,
        uplo, tran, n, k, alpha, A, lda, B, ldb, beta, C, ldc
      );
    }
    else static_assert(dependent_false_v<T>, "unknown cublas data type");
    TF_CHECK_CUBLAS(stat, "failed to run syr2k");
  });
}

// c_syr2k
template <typename T>
cudaTask cublasFlowCapturer::c_syr2k(
  cublasFillMode_t uplo, cublasOperation_t tran,
  int n, int k,
  const T *alpha,
  const T *A, int lda,
  const T *B, int ldb,
  const T *beta,
  T *C, int ldc
) {
  return syr2k(
    cublas_rfill(uplo), cublas_rtran<T>(tran),
    n, k, alpha, B, ldb, A, lda, beta, C, ldc
  );
}

// syrkx
template <typename T>
cudaTask cublasFlowCapturer::syrkx(
  cublasFillMode_t uplo, cublasOperation_t tran,
  int n, int k,
  const T *alpha,
  const T *A, int lda,
  const T *B, int ldb,
  const T *beta,
  T *C, int ldc
) {
  return on(
  [this, uplo, tran, n, k, alpha, A, lda, B, ldb, beta, C, ldc] 
  (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSsyrkx(_handle,
        uplo, tran, n, k, alpha, A, lda, B, ldb, beta, C, ldc
      );
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDsyrkx(_handle,
        uplo, tran, n, k, alpha, A, lda, B, ldb, beta, C, ldc
      );
    }
    else static_assert(dependent_false_v<T>, "unknown cublas data type");
    TF_CHECK_CUBLAS(stat, "failed to run syrkx");
  });
}

// c_syrkx
template <typename T>
cudaTask cublasFlowCapturer::c_syrkx(
  cublasFillMode_t uplo, cublasOperation_t tran,
  int n, int k,
  const T *alpha,
  const T *A, int lda,
  const T *B, int ldb,
  const T *beta,
  T *C, int ldc
) {
  return syrkx(
    cublas_rfill(uplo), cublas_rtran<T>(tran),
    n, k, alpha, B, ldb, A, lda, beta, C, ldc
  );
}

// trmm
template <typename T>
cudaTask cublasFlowCapturer::trmm(
  cublasSideMode_t side, cublasFillMode_t uplo,
  cublasOperation_t tran, cublasDiagType_t diag,
  int m, int n,
  const T *alpha,
  const T *A, int lda,
  const T *B, int ldb,
  T *C, int ldc
) {
  
  return on(
  [this, side, uplo, tran, diag, m, n, alpha, A, lda, B, ldb, C, ldc] 
  (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasStrmm(_handle,
        side, uplo, tran, diag, m, n, alpha, A, lda, B, ldb, C, ldc
      );
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDtrmm(_handle,
        side, uplo, tran, diag, m, n, alpha, A, lda, B, ldb, C, ldc
      );
    }
    else static_assert(dependent_false_v<T>, "unknown cublas data type");
    TF_CHECK_CUBLAS(stat, "failed to run trmm");
  });
}

// c_trmm
template <typename T>
cudaTask cublasFlowCapturer::c_trmm(
  cublasSideMode_t side, cublasFillMode_t uplo,
  cublasOperation_t tran, cublasDiagType_t diag,
  int m, int n,
  const T *alpha,
  const T *A, int lda,
  const T *B, int ldb,
  T *C, int ldc
) {
  return trmm(
    cublas_rside(side), cublas_rfill(uplo), tran, diag,
    n, m, alpha, A, lda, B, ldb, C, ldc
  );
}
    
// trsm
template <typename T>
cudaTask cublasFlowCapturer::trsm(
  cublasSideMode_t side, cublasFillMode_t uplo,
  cublasOperation_t tran, cublasDiagType_t diag,
  int m, int n,
  const T *alpha,
  const T *A, int lda,
  T *B, int ldb
) {

  return on(
  [this, side, uplo, tran, diag, m, n, alpha, A, lda, B, ldb] 
  (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasStrsm(_handle,
        side, uplo, tran, diag, m, n, alpha, A, lda, B, ldb
      );
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDtrsm(_handle,
        side, uplo, tran, diag, m, n, alpha, A, lda, B, ldb
      );
    }
    else static_assert(dependent_false_v<T>, "unknown cublas data type");
    TF_CHECK_CUBLAS(stat, "failed to run trsm");
  });
}

// c_trsm
template <typename T>
cudaTask cublasFlowCapturer::c_trsm(
  cublasSideMode_t side, cublasFillMode_t uplo,
  cublasOperation_t tran, cublasDiagType_t diag,
  int m, int n,
  const T *alpha,
  const T *A, int lda,
  T *B, int ldb
) {
  return trsm(
    cublas_rside(side), cublas_rfill(uplo), tran, diag,
    n, m, alpha, A, lda, B, ldb
  );
}

}  // end of namespace tf -----------------------------------------------------

