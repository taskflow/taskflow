#pragma once

#include "cublas_handle.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// C++ wrapper over cublas functions
// ----------------------------------------------------------------------------

/** 
@brief performs matrix-matrix multiplication

This function performs matrix-matrix multiplication:

<tt>C = alpha * op (A) * op (B) + beta * C</tt>,

where @c alpha and @c beta are scalars, and @c A, @c B, and @c C
are 2D matrices stored in column-major format 
with dimension @c op(A) as @c m by @c k,
dimension @c op(B) as @c k by @c n, and @c C as @c m by @c n.

@tparam T data type
@param ta transport operation @c op(A)
@param tb transport operation @c op(B)
@param m number of rows of matrix @c C and @c op(A)
@param n number of columns of matrix @c C and @c op(B)
@param k number of columns of @c op(A) and rows of @c op(B)
@param alpha pointer to the @c alpha scalar
@param A pointer to the address of @c A
@param lda leading dimension of 2D array used to store the matrix @c A
@param B pointer to the address of @c B
@param ldb leading dimension of 2D array used to store the matrix @c B
@param beta pointer to the @c beta scalar
@param C pointer to the address of @c C 
@param ldc leading dimension of 2D array used to store the matrix @c C

*/
template <typename T>
void cublas_gemm(
  cublasHandle_t native_handle,
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A, int lda,
  const T *B, int ldb,
  const T *beta,
  T *C, int ldc
) {

  cublasStatus_t stat;

  if constexpr(std::is_same_v<T, float>) {
    stat = cublasSgemm(native_handle,
      ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
    );
  }
  else if constexpr(std::is_same_v<T, double>) {
    stat = cublasDgemm(native_handle,
      ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
    );
  }
  else {
    static_assert(dependent_false_v<T>, "unknown cublas data type");
  }

  TF_CHECK_CUBLAS(stat, "failed to run gemm");
}

/** 
@brief similar to gemm but operates on C-styled row-major layout
*/
template <typename T>
void cublas_c_gemm(
  cublasHandle_t native_handle,
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A, int lda,
  const T *B, int ldb,
  const T *beta,
  T *C, int ldc
) {

  cublasStatus_t stat;

  if constexpr(std::is_same_v<T, float>) {
    stat = cublasSgemm(native_handle,
      tb, ta, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc
    );
  }
  else if constexpr(std::is_same_v<T, double>) {
    stat = cublasDgemm(native_handle,
      tb, ta, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc
    );
  }
  else {
    static_assert(dependent_false_v<T>, "unknown cublas data type");
  }

  TF_CHECK_CUBLAS(stat, "failed to run c_gemm");
}

/**
@brief performs matrix-matrix multiplication over a batch of matrices

@tparam T data type
@param ta transport operation @c op(A[i])
@param tb transport operation @c op(B[i])
@param m number of rows of matrix @c C[i] and @c op(A[i])
@param n number of columns of matrix @c C[i] and @c op(B[i])
@param k number of columns of @c op(A[i]) and rows of @c op(B[i])
@param alpha pointer to the @c alpha scalar
@param A array pointer to @c A batch
@param lda leading dimension of 2D array used to store the matrix @c A[i]
@param B array pointer to @c B batch
@param ldb leading dimension of 2D array used to store the matrix @c B[i]
@param beta pointer to the @c beta scalar
@param C array pointer to @c C batch
@param ldc leading dimension of 2D array used to store the matrix @c C[i]
@param bc batch size (number of matrices)

The batch must be @i uniform.
All instances in the batch must have the same dimensions <tt>(m, n, k)</tt>, 
leading dimensions <tt>(lda, ldb, ldc)</tt> and transpositions 
<tt>(ta, tb)</tt> for their respective @c A, @c B and @c C matrices. 
The address of the input matrices and the output matrix of each instance 
of the batch are read from arrays of pointers passed to the function by the caller.

<tt>C[i]= alpha * op (A[i]) * op (B[i]) + beta * C[i], i in [0, bc)</tt>,

where @c alpha and @c beta are scalars, and @c A[i], @c B[i], and @c C[i]
are 2D matrices stored in column-major format 
with dimension @c op(A) as @c m by @c k,
dimension @c op(B) as @c k by @c n, and @c C as @c m by @c n.

*/
template <typename T>
void cublas_gemm_batched(
  cublasHandle_t native_handle,
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A[], int lda,
  const T *B[], int ldb,
  const T *beta,
  T *C[], int ldc,
  int bc
) {

  cublasStatus_t stat;

  if constexpr(std::is_same_v<T, float>) {
    stat = cublasSgemmBatched(native_handle,
      ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bc
    );
  }
  else if constexpr(std::is_same_v<T, double>) {
    stat = cublasDgemmBatched(native_handle,
      ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bc
    );
  }
  else static_assert(dependent_false_v<T>, "unknown cublas data type");

  TF_CHECK_CUBLAS(stat, "failed to run gemm_batched");
}

/**
@brief similar to gemm_batched but operates on C-styled row-major layout
*/ 
template <typename T>
void cublas_c_gemm_batched(
  cublasHandle_t native_handle,
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A[], int lda,
  const T *B[], int ldb,
  const T *beta,
  T *C[], int ldc,
  int bc
) {
  cublasStatus_t stat;

  if constexpr(std::is_same_v<T, float>) {
    stat = cublasSgemmBatched(native_handle,
      tb, ta, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, bc
    );
  }
  else if constexpr(std::is_same_v<T, double>) {
    stat = cublasDgemmBatched(native_handle,
      tb, ta, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, bc
    );
  }
  else static_assert(dependent_false_v<T>, "unknown cublas data type");

  TF_CHECK_CUBLAS(stat, "failed to run c_gemm_batched");
}

/**
@brief performs matrix-matrix multiplication over a batch of matrices 
       with strided memory access

Here, we use @c A[i], @c B[i], @c C[i] as notation 
for A, B and C matrices in the @c i-th instance of the batch, 
implicitly assuming they are respectively address offsets 
@c sA, @c sB, @c sC away from @c A[i-1], @c B[i-1], @c C[i-1].

@tparam T data type
@param ta transport operation @c op(A[i])
@param tb transport operation @c op(B[i])
@param m number of rows of matrix @c C[i] and @c op(A[i])
@param n number of columns of matrix @c C[i] and @c op(B[i])
@param k number of columns of @c op(A[i]) and rows of @c op(B[i])
@param alpha pointer to the @c alpha scalar
@param A pointer to @c A batch
@param lda leading dimension of 2D array used to store the matrix @c A[i]
@param sA address offset between @c A[i] and @c A[i+1]
@param B pointer to @c B batch
@param ldb leading dimension of 2D array used to store the matrix @c B[i]
@param sB address offset between @c B[i] and @c B[i+1]
@param beta pointer to the @c beta scalar
@param C pointer to @c C batch
@param ldc leading dimension of 2D array used to store the matrix @c C[i]
@param sC address offset between @c C[i] and @c C[i+1]
@param bc batch size (number of matrices)

The batch must be @i uniform. 
All instances in the batch must have the same dimensions <tt>(m, n, k)</tt>, 
leading dimensions <tt>(lda, ldb, ldc)</tt> and transpositions 
<tt>(ta, tb)</tt> for their respective @c A, @c B and @c C matrices. 
Input matrices @c A, @c B and output matrix @c C for each instance of the batch 
are located at fixed address offsets from their locations in the previous instance. 
Pointers to @c A, @c B and @c C matrices for the first instance are passed 
to the function by the user along with the address offsets - 
@c sA, @c sB and @c sC that determine the locations 
of input and output matrices in future instances.

<tt>C + i*sC = alpha * op (A + i*sA) * op (B + i*sB) 
                  + beta * (C + i*sC), i in [0, bc)</tt>,

where @c alpha and @c beta are scalars, and @c A[i], @c B[i], and @c C[i]
are 2D matrices stored in column-major format 
with dimension @c op(A) as @c m by @c k,
dimension @c op(B) as @c k by @c n, and @c C as @c m by @c n.

On certain problem sizes, it might be advantageous to create multiple gemm tasks
to take advantage of concurrent kernels, rather than this method.
*/
template <typename T>
void cublas_gemm_sbatched(
  cublasHandle_t native_handle,
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A, int lda, long long int sA,
  const T *B, int ldb, long long int sB,
  const T *beta,
  T *C, int ldc, long long int sC,
  int bc
) {
      
  cublasStatus_t stat;

  if constexpr(std::is_same_v<T, float>) {
    stat = cublasSgemmStridedBatched(native_handle,
      ta, tb, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, bc
    );
  }
  else if constexpr(std::is_same_v<T, double>) {
    stat = cublasDgemmStridedBatched(native_handle,
      ta, tb, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, bc
    );
  }
  else static_assert(dependent_false_v<T>, "unknown cublas data type");
  
  TF_CHECK_CUBLAS(stat, "failed to run gemm_sbatched");
}

/** 
@brief similar to gemm_batached but operates on C-styled row-major layout
*/
template <typename T>
void cublas_c_gemm_sbatched(
  cublasHandle_t native_handle,
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A, int lda, long long int sA,
  const T *B, int ldb, long long int sB,
  const T *beta,
  T *C, int ldc, long long int sC,
  int bc
){
      
  cublasStatus_t stat;

  if constexpr(std::is_same_v<T, float>) {
    stat = cublasSgemmStridedBatched(native_handle,
      tb, ta, n, m, k, alpha, B, ldb, sB, A, lda, sA, beta, C, ldc, sC, bc
    );
  }
  else if constexpr(std::is_same_v<T, double>) {
    stat = cublasDgemmStridedBatched(native_handle,
      tb, ta, n, m, k, alpha, B, ldb, sB, A, lda, sA, beta, C, ldc, sC, bc
    );
  }
  else static_assert(dependent_false_v<T>, "unknown cublas data type");

  TF_CHECK_CUBLAS(stat, "failed to run c_gemm_sbatched");

}



}  // end of namespace tf -----------------------------------------------------

