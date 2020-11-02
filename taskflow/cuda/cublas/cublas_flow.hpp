#pragma once

#include "cublas_helper.hpp"
#include "cublas_level1.hpp"
#include "cublas_level3.hpp"

/** 
@file cublas_flow.hpp
*/

namespace tf {

// ----------------------------------------------------------------------------
// cublasFlowCapturer definition
// ----------------------------------------------------------------------------

/**
@class cublasFlowCapturer

@brief class object to construct a cuBLAS task graph

A %cublasFlowCapturer provides a higher-level interface over the cuBLAS library
and hide concurrency details from users.
All pointers used to %cublasFlowCapturer methods must be in GPU memory space or managed 
(i.e., @c cudaMallocManaged),
including scalars, @c alpha and @c beta, input data and output data pointers.

Currently,  %cublasFlowCapturer supports only float and double data types.
*/
class cublasFlowCapturer : public cudaFlowCapturerBase {

  public:
    
    /**
    @brief constructs a cublas flow capturer
     */
    cublasFlowCapturer() = default;
    
    /**
    @brief gets the native cublas handle associated with this %cublasFlowCapturer
    */
    cublasHandle_t native_handle();
    
    // ------------------------------------------------------------------------
    // Helper methods
    // ------------------------------------------------------------------------

    /**
    @brief copies vector data from host to device

    This method effectively calls <tt>cublas_vset_async(stream, args...)</tt>
    with @c stream managed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask vset(Ts&&... args);
    
    /**
    @brief copies vector data from device to host

    This method effectively calls <tt>cublas_vget_async(stream, args...)</tt>
    with @c stream managed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask vget(Ts&&... args);
    
    // ------------------------------------------------------------------------
    // Level-1 vector-vector operations
    // ------------------------------------------------------------------------

    /**
    @brief finds the smallest index of the element of the maximum 
           absolute magnitude
    
    This method calls native @c cublas<t>amax with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by 
    the %cublasFlowCapturer and @c args... are the given arguments.
    
    @tparam T data type
    
    @param n number of elements in vector @c x
    @param x pointer to the memory address of the vector
    @param incx stride between consecutive elements of @c x
    @param result the resulting index (1-based indexing)
    */
    template <typename T>
    cudaTask amax2(int n, const T* x, int incx, int* result);
    
    /** 
    @brief finds the smallest index of the element of the maximum absolute magnitude
    
    This method effectively calls tf::cublas_amax with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask amax(Ts&&... args);

    /** 
    @brief finds the smallest index of the element of the minimum absolute magnitude
    
    This method effectively calls tf::cublas_amin with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask amin(Ts&&... args);
    
    /**
    @brief finds the smallest index of the element of the minimum 
           absolute magnitude
    
    This method calls native @c cublas<t>amin with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by 
    the %cublasFlowCapturer and @c args... are the given arguments.
    
    @tparam T data type
    
    @param n number of elements in vector @c x
    @param x pointer to the memory address of the vector
    @param incx stride between consecutive elements of @c x
    @param result the resulting index (1-based indexing)
    */
    template <typename T>
    cudaTask amin2(int n, const T* x, int incx, int* result);
    
    /** 
    @brief finds the sum of absolute values of elements in a vector
    
    This method effectively calls tf::cublas_asum with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask asum(Ts&&... args);

    /**
    @brief finds the sum of absolute values of the elements over a vector
    
    This method calls native @c cublas<t>asum with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by 
    the %cublasFlowCapturer and @c args... are the given arguments.
    
    @tparam T data type
    
    @param n number of elements in vector @c x
    @param x pointer to the memory address of the vector
    @param incx stride between consecutive elements of @c x
    @param result the result
    */
    template <typename T>
    cudaTask asum2(int n, const T* x, int incx, T* result);
    
    /** 
    @brief multiplies a vector by a scalar and adds it to a vector
    
    This method effectively calls tf::cublas_axpy with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask axpy(Ts&&... args);

    /**
    @brief multiples a vector by a scalar and adds it to a vector
    
    This function multiplies the vector @c x by the scalar @c alpha and 
    adds it to the vector @c y overwriting the latest vector with the result. 
    Hence, the performed operation is:
    
      <tt>y[j] = alpha * x[k] + y[j]</tt>, 
      
    where @c j and @c k are indices of @c n elements with step sizes 
    @c incy and @c incx.
    
    This method calls native @c cublas<t>asum with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by 
    the %cublasFlowCapturer and @c args... are the given arguments.
    
    @tparam T data type
    
    @param n number of elements in vectors @c x and @c y
    @param alpha scalar used to multiplication
    @param x pointer to the memory address of the vector @c x
    @param incx stride between consecutive elements of @c x
    @param y pointer to the memory address of the vector @c y
    @param incy stride between consecutive elements of @c y
    */
    template <typename T>
    cudaTask axpy2(
      int n, const T *alpha, const T *x, int incx, T *y, int incy
    );

    /** 
    @brief copies a vector to another vector
    
    This method effectively calls tf::cublas_copy with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask vcopy(Ts&&... args);

    /**
    @brief copies a vector to another vector
    
    This function copies @c n elements from a vector @c x of a step size @c incx 
    to another vector @c y of step size @c incy.
     
    adds it to the vector @c y overwriting the latest vector with the result. 
    Hence, the performed operation is:
    
      <tt>y[j] = x[k]</tt>, 
      
    where @c j and @c k are indices of @c n elements with step sizes 
    @c incy and @c incx.
    
    This method calls native @c cublas<t>copy with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by 
    the %cublasFlowCapturer and @c args... are the given arguments.
    
    @tparam T data type
    
    @param n number of elements to copy
    @param x pointer to the memory address of the vector @c x
    @param incx stride between consecutive elements of @c x
    @param y pointer to the memory address of the vector @c y
    @param incy stride between consecutive elements of @c y
    */
    template <typename T>
    cudaTask vcopy2(int n, const T* x, int incx, T* y, int incy);
    
    /** 
    @brief computes the dot product of two vectors
    
    This method effectively calls tf::cublas_dot with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask dot(Ts&&... args);
    
    /**
    @brief computes the dot product of two vectors

    <tt>sum += x[i] * y[i]</tt>
    
    This method calls native @c cublas<t>dot with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by 
    the %cublasFlowCapturer and @c args... are the given arguments.

    @tparam T data type
    
    @param n number of elements to perform the dot product
    @param x pointer to the memory address of the vector @c x
    @param incx stride between consecutive elements of @c x
    @param y pointer to the memory address of the vector @c y
    @param incy stride between consecutive elements of @c y
    @param result the resulting dot product
    */
    template <typename>
    cudaTask dot2(int n, const T* x, int incx, const T* y, int incy, T* result);
    
    /** 
    @brief computes the Euclidean norm of a vector
    
    This method effectively calls tf::cublas_nrm2 with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask nrm2(Ts&&... args);

    /**
    @brief computes the Euclidean norm of a vector
    
    This method calls native @c cublas<t>nrm2 with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by 
    the %cublasFlowCapturer and @c args... are the given arguments.
    
    @tparam T data type
    
    @param n number of elements in vector @c x
    @param x pointer to the memory address of the vector
    @param incx stride between consecutive elements of @c x
    @param result the result
    */
    template <typename T>
    cudaTask nrm22(int n, const T* x, int incx, T* result);
    
    /** 
    @brief multiples a vector by a scalar
    
    This method effectively calls tf::cublas_scal with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask scal(Ts&&... args);

    /**
    @brief scales a vector by a scalar
    
    This method calls native @c cublas<t>scal with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by 
    the %cublasFlowCapturer and @c args... are the given arguments.
    
    @tparam T data type
    
    @param n number of elements in vector @c x
    @param scalar scalar used for multiplication
    @param x pointer to the memory address of the vector
    @param incx stride between consecutive elements of @c x
    */
    template <typename T>
    cudaTask scal2(int n, const T* scalar, T* x, int incx);
    
    /** 
    @brief swaps the elements of two vectors
    
    This method effectively calls tf::cublas_swap with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask swap(Ts&&... args);

    /**
    @brief swaps elements between two vectors
    
    This function interchanges the elements of vectors @c x and @c y. 
    Hence, the performed operation is:
    
    <tt>y[j] <-> x[k]</tt>,
    
    where @c j is the index of element in @c y with a step size @c incy and
    @c k is the index of element in @c x with a step size @c incx.
    
    This method calls native @c cublas<t>swap with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by 
    the %cublasFlowCapturer and @c args... are the given arguments.
    
    @tparam T data type
    
    @param n number of elements to perform the dot product
    @param x pointer to the memory address of the vector @c x
    @param incx stride between consecutive elements of @c x
    @param y pointer to the memory address of the vector @c y
    @param incy stride between consecutive elements of @c y
    */
    template <typename T>
    cudaTask swap2(int n, T* x, int incx, T* y, int incy);

    // ------------------------------------------------------------------------
    // TODO Level-2 matrix_vector operations
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // TODO Level-3 matrix-matrix operations
    // ------------------------------------------------------------------------
    
    /** 
    @brief performs matrix-matrix addition/transposition on column-major layout
    
    This method effectively calls tf::cublas_geam with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask geam(Ts&&... args);

    /**
    @brief performs matrix-matrix addition and transposition
    
    This function performs the matrix-matrix addition/transposition:
    
      <tt>C = alpha * op(A) + beta * op(B)</tt>,
    
    where @c alpha and @c beta are scalars, and @c A, @c B and @c C are matrices 
    stored in column-major format with dimensions @c op(A) as @c m by @c n, 
    @c op(B) as @c m by @c n and @c C as @c m by @c n, respectively. 
    
    The operation is out-of-place if @c C does not overlap @c A or @c B.
    
    The in-place mode supports the following two operations:
    
      1. <tt>C = alpha * C + beta * op(B)</tt>
      2. <Tt>C = alpha op(A) + beta * C</tt>
    
    The operation includes the following special cases:
    
      1. the user can reset matrix @c C to zero by setting @c alpha and 
         @c beta to 0
      2. the user can transpose matrix @c A by setting @c alpha to 1 and 
         @c beta to 0
    
    The input matrices are in column-major storage.
    
    This method calls native @c cublas<t>geam with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by 
    the %cublasFlowCapturer and @c args... are the given arguments.
    
    @tparam T data type
    
    @param ta transport operation @c op(A)
    @param tb transport operation @c op(B)
    @param m number of rows of matrix @c C and @c op(A)
    @param n number of columns of matrix @c C and @c op(B)
    @param alpha pointer to the @c alpha scalar
    @param A pointer to the address of @c A
    @param lda leading dimension of 2D array used to store the matrix @c A
    @param beta pointer to the @c beta scalar
    @param B pointer to the address of @c B
    @param ldb leading dimension of 2D array used to store the matrix @c B
    @param C pointer to the address of @c C 
    @param ldc leading dimension of 2D array used to store the matrix @c C
    */
    template <typename T>
    cudaTask geam2(
      cublasOperation_t ta, cublasOperation_t tb,
      int m, int n,
      const T *alpha,
      const T *A, int lda,
      const T *beta,
      const T *B, int ldb,
      T *C, int ldc
    );
    
    /** 
    @brief performs matrix-matrix addition/transposition on row-major layout
    
    This method effectively calls tf::cublas_c_geam with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask c_geam(Ts&&... args);

    /** 
    @brief similar to tf::cublasFlowCapturer::geam but on row-major layout
    */
    template <typename T>
    cudaTask c_geam2(
      cublasOperation_t ta, cublasOperation_t tb,
      int m, int n,
      const T *alpha,
      const T *A, int lda,
      const T *beta,
      const T *B, int ldb,
      T *C, int ldc
    );

    /** 
    @brief performs matrix-matrix multiplication on column-major layout
    
    This method effectively calls tf::cublas_gemm with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is managed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask gemm(Ts&&... args);

    /** 
    @brief performs matrix-matrix multiplication
    
    This function performs matrix-matrix multiplication:
    
    <tt>C = alpha * op (A) * op (B) + beta * C</tt>,
    
    where @c alpha and @c beta are scalars, and @c A, @c B, and @c C
    are 2D matrices stored in column-major format 
    with dimension @c op(A) as @c m by @c k,
    dimension @c op(B) as @c k by @c n, and @c C as @c m by @c n.
    
    The input matrices are in column-major storage.
    
    This method calls native @c cublas<t>gemm with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by 
    the %cublasFlowCapturer and @c args... are the given arguments.
    
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
    cudaTask gemm2(
      cublasOperation_t ta, cublasOperation_t tb,
      int m, int n, int k,
      const T *alpha,
      const T *A, int lda,
      const T *B, int ldb,
      const T *beta,
      T *C, int ldc
    );

    /** 
    @brief performs matrix-matrix multiplication on C-styled row-major layout
    
    This method effectively calls tf::cublas_c_gemm with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is managed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask c_gemm(Ts&&... args);
    
    /**
    @brief similar to tf::cublasFlowCapturer::gemm but operates on C-styled 
           row-major layout
    */
    template <typename T>
    cudaTask c_gemm2(
      cublasOperation_t ta, cublasOperation_t tb,
      int m, int n, int k,
      const T *alpha,
      const T *A, int lda,
      const T *B, int ldb,
      const T *beta,
      T *C, int ldc
    );

    /**
    @brief performs batched matrix-matrix multiplication on column-major layout
    
    This method effectively calls tf::cublas_gemm_batched with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is managed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask gemm_batched(Ts&&... args);

    /**
    @brief performs matrix-matrix multiplication over a batch of matrices 
    
    The batch must be @em uniform.
    All instances in the batch must have the same dimensions <tt>(m, n, k)</tt>, 
    leading dimensions <tt>(lda, ldb, ldc)</tt> and transpositions 
    <tt>(ta, tb)</tt> for their respective @c A, @c B and @c C matrices. 
    The address of the input matrices and the output matrix of each instance 
    of the batch are read from arrays of pointers passed to the function 
    by the caller.
    
    <tt>C[i]= alpha * op (A[i]) * op (B[i]) + beta * C[i], i in [0, bc)</tt>,
    
    where @c alpha and @c beta are scalars, and @c A[i], @c B[i], and @c C[i]
    are 2D matrices stored in column-major format 
    with dimension @c op(A) as @c m by @c k,
    dimension @c op(B) as @c k by @c n, and @c C as @c m by @c n.
    
    The input matrices are in column-major storage.
    
    This method calls native @c cublas<t>gemmBatched with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by 
    the %cublasFlowCapturer and @c args... are the given arguments.
    
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
    */
    template <typename T>
    cudaTask gemm_batched2(
      cublasOperation_t ta, cublasOperation_t tb,
      int m, int n, int k,
      const T *alpha,
      const T *A[], int lda,
      const T *B[], int ldb,
      const T *beta,
      T *C[], int ldc,
      int bc
    );
    
    /**
    @brief performs batched matrix-matrix multiplication on C-styled row-major layout
    
    This method effectively calls tf::cublas_c_gemm_batched with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is managed by the %cublasFlowCapturer.
    */ 
    template <typename... Ts>
    cudaTask c_gemm_batched(Ts&&... args);
    
    /**
    @brief similar to tf::cublasFlowCapturer::gemm_batched but operates on 
           C-styled row-major layout
    */
    template <typename T>
    cudaTask c_gemm_batched_2(
      cublasOperation_t ta, cublasOperation_t tb,
      int m, int n, int k,
      const T *alpha,
      const T *A[], int lda,
      const T *B[], int ldb,
      const T *beta,
      T *C[], int ldc,
      int bc
    );
    
    /**
    @brief performs batched matrix-matrix multiplication on column-major layout
           with strided memory access
    
    This method effectively calls tf::cublas_gemm_sbatched with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is managed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask gemm_sbatched(Ts&&... args);

    /**
    @brief performs matrix-matrix multiplication over a batch of matrices 
           with strided memory access
    
    Here, we use @c A[i], @c B[i], @c C[i] as notation 
    for A, B and C matrices in the @c i-th instance of the batch, 
    implicitly assuming they are respectively address offsets 
    @c sA, @c sB, @c sC away from @c A[i-1], @c B[i-1], @c C[i-1].
    
    The input matrices are in column-major storage.
    
    This method calls native @c cublas<t>gemmStridedBatched with 
    packed parameters, <tt>(handle, args...)</tt>, where @c handle is manaed by 
    the %cublasFlowCapturer and @c args... are the given arguments.
    
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
    
    The batch must be @em uniform. 
    All instances in the batch must have the same dimensions <tt>(m, n, k)</tt>, 
    leading dimensions <tt>(lda, ldb, ldc)</tt> and transpositions 
    <tt>(ta, tb)</tt> for their respective @c A, @c B and @c C matrices. 
    Input matrices @c A, @c B and output matrix @c C for each instance of the batch 
    are located at fixed address offsets from their locations in the previous instance. 
    Pointers to @c A, @c B and @c C matrices for the first instance are passed 
    to the function by the user along with the address @em offsets - 
    @c sA, @c sB and @c sC that determine the locations 
    of input and output matrices in future instances.
    
    <tt>C + i*sC = alpha * op (A + i*sA) * op (B + i*sB) + beta * (C + i*sC), i in [0, bc)</tt>,
    
    where @c alpha and @c beta are scalars, and @c A[i], @c B[i], and @c C[i]
    are 2D matrices stored in column-major format 
    with dimension @c op(A) as @c m by @c k,
    dimension @c op(B) as @c k by @c n, and @c C as @c m by @c n.
    
    On certain problem sizes, it might be advantageous to create multiple gemm tasks
    to take advantage of concurrent kernels, rather than this method.
    */
    template <typename T>
    cudaTask gemm_sbatched2(
      cublasOperation_t ta, cublasOperation_t tb,
      int m, int n, int k,
      const T *alpha,
      const T *A, int lda, long long int sA,
      const T *B, int ldb, long long int sB,
      const T *beta,
      T *C, int ldc, long long int sC,
      int bc
    );
    
    /** 
    @brief performs batched matrix-matrix multiplication on C-styled row-major 
           layout with strided memory access
    
    This method effectively calls tf::cublas_c_gemm_sbatched with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is managed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask c_gemm_sbatched(Ts&&... args);
    
    /**
    @brief similar to tf::cublasFlowCapturer::c_gemm_sbatched but operates on
           C-styled row-major layout
    */
    template <typename T>
    cudaTask c_gemm_sbatched2(
      cublasOperation_t ta, cublasOperation_t tb,
      int m, int n, int k,
      const T *alpha,
      const T *A, int lda, long long int sA,
      const T *B, int ldb, long long int sB,
      const T *beta,
      T *C, int ldc, long long int sC,
      int bc
    );
    
  private:
    
    cublasScopedPerThreadHandle _handle;

    void _stream(cudaStream_t);
};

// Procedure: _stream
inline void cublasFlowCapturer::_stream(cudaStream_t stream) {
  TF_CHECK_CUBLAS(
    cublasSetStream(_handle, stream), "failed to set cublas stream"
  );
}

// Function: native_handle
inline cublasHandle_t cublasFlowCapturer::native_handle() {
  return _handle;
}

// ---------------------------------------------------------------------------- 
// Helper functions
// ---------------------------------------------------------------------------- 

// Function: vset
template <typename... Ts>
cudaTask cublasFlowCapturer::vset(Ts&&... args) {
  return on([args...] (cudaStream_t stream) mutable {
    cublas_vset_async(stream, args...);
  });
}

// Function: vget
template <typename... Ts>
cudaTask cublasFlowCapturer::vget(Ts&&... args) {
  return on([args...] (cudaStream_t stream) mutable {
    cublas_vget_async(stream, args...);
  });
}
    
// ---------------------------------------------------------------------------- 
// Level-1 functions
// ---------------------------------------------------------------------------- 

// Function: amax
template <typename... Ts>
cudaTask cublasFlowCapturer::amax(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_amax(_handle, args...);
  });
}

// Function: amax2
template <typename T>
cudaTask cublasFlowCapturer::amax2(
  int n, const T* x, int incx, int* result
) {
  return on([this, n, x, incx, result] (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasIsamax(_handle, n, x, incx, result);
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasIdamax(_handle, n, x, incx, result);
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }

    TF_CHECK_CUBLAS(stat, "failed to run cublas<t>amax");
  });
}

// Function: amin
template <typename... Ts>
cudaTask cublasFlowCapturer::amin(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_amin(_handle, args...);
  });
}

// Function: amin2
template <typename... Ts>
cudaTask cublasFlowCapturer::amin2(
  int n, const T* x, int incx, int* result
) {
  return on([this, n, x, incx, result] (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasIsamin(_handle, n, x, incx, result);
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasIdamin(_handle, n, x, incx, result);
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }
    TF_CHECK_CUBLAS(stat, "failed to run cublas<t>amin");
  });
}

// Function: asum
template <typename... Ts>
cudaTask cublasFlowCapturer::asum(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_asum(_handle, args...);
  });
}

// Function: asum2
template <typename T>
cudaTask cublasFlowCapturer::asum2(
  int n, const T* x, int incx, T* result
) {
  return on([this, n, x, incx, result] (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSasum(handle, n, x, incx, result);
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDasum(handle, n, x, incx, result);
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }

    TF_CHECK_CUBLAS(stat, "failed to run cublas<t>asum");
  });
}

// Function: axpy
template <typename... Ts>
cudaTask cublasFlowCapturer::axpy(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_axpy(_handle, args...);
  });
}

// Function: axpy2
template <typename T>
cudaTask cublasFlowCapturer::axpy2(
  int n, const T *alpha, const T *x, int incx, T *y, int incy
) {
  return on([this, n, alpha, x, incx, y, incy] (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSaxpy(_handle, n, alpha, x, incx, y, incy);
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDaxpy(_handle, n, alpha, x, incx, y, incy);
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }
    TF_CHECK_CUBLAS(stat, "failed to run cublas<t>axpy");
  });
}

// Function: vcopy
template <typename... Ts>
cudaTask cublasFlowCapturer::vcopy(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_copy(_handle, args...);
  });
}

// Function: vcopy2
template <typename T>
cudaTask cublasFlowCapturer::vcopy2(
  int n, const T* x, int incx, T* y, int incy
) {
  return on([this, n, x, incx, y, incy] (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasScopy(_handle, n, x, incx, y, incy);
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDcopy(_handle, n, x, incx, y, incy);
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }

    TF_CHECK_CUBLAS(stat, "failed to run cublas<t>copy");
  });
}

// Function: dot
template <typename... Ts>
cudaTask cublasFlowCapturer::dot(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_dot(_handle, args...);
  });
}

// Function: dot
template <typename T>
cudaTask cublasFlowCapturer::dot2(
  int n, const T* x, int incx, const T* y, int incy, T* result
) {
  return on([this, n, x, incx, y, incy, result] (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSdot(_handle, n, x, incx, y, incy, result);
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDdot(_handle, n, x, incx, y, incy, result);
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }
    TF_CHECK_CUBLAS(stat, "failed to run cublas<t>dot");
  });
}

// Function: nrm2
template <typename... Ts>
cudaTask cublasFlowCapturer::nrm2(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_nrm2(_handle, args...);
  });
}

template <typename T>
cudaTask cublasFlowCapturer::nrm22(int n, const T* x, int incx, T* result) {
  return on([this, n, x, incx, result] (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSnrm2(_handle, n, x, incx, result);
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDnrm2(_handle, n, x, incx, result);
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }

    TF_CHECK_CUBLAS(stat, "failed to run cublas<t>nrm2");
  });
}

// Function: scal
template <typename... Ts>
cudaTask cublasFlowCapturer::scal(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_scal(_handle, args...);
  });
}

// Function: scal
template <typename T>
cudaTask cublasFlowCapturer::scal2(int n, const T* scalar, T* x, int incx) {
  return on([this, n, scalar, x, incx] (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSscal(_handle, n, scalar, x, incx);
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDscal(_handle, n, scalar, x, incx);
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }
    TF_CHECK_CUBLAS(stat, "failed to run cublas<t>scal");
  });
}

// Function: swap
template <typename... Ts>
cudaTask cublasFlowCapturer::swap(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_swap(_handle, args...);
  });
}

template <typename T>
cudaTask cublasFlowCapturer::swap2(int n, T* x, int incx, T* y, int incy) {
  return on([this, n, x, incx, y, incy] (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSswap(_handle, n, x, incx, y, incy);
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDswap(_handle, n, x, incx, y, incy);
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }

    TF_CHECK_CUBLAS(stat, "failed to run cublas<t>swap");
  });
}

// ---------------------------------------------------------------------------- 
// Level-3 functions
// ---------------------------------------------------------------------------- 

// Function: geam
template <typename... Ts>
cudaTask cublasFlowCapturer::geam(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_geam(_handle, args...);
  });
}

// Function: geam2
template <typename... Ts>
cudaTask cublasFlowCapturer::geam2(
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
template <typename... Ts>
cudaTask cublasFlowCapturer::c_geam(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_c_geam(_handle, args...);
  });
}

// Function: c_geam2
template <typename T>
cudaTask cublasFlowCapturer::c_geam2(
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
template <typename... Ts>
cudaTask cublasFlowCapturer::gemm(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_gemm(_handle, args...);
  });
}

// Function: gemm2
template <typename T>
cudaTask cublasFlowCapturer::gemm2(
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

// Function: c_gemm
template <typename... Ts>
cudaTask cublasFlowCapturer::c_gemm(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_c_gemm(_handle, args...);
  });
}
    
template <typename T>
cudaTask cublasFlowCapturer::c_gemm2(
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
template <typename... Ts>
cudaTask cublasFlowCapturer::gemm_batched(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_gemm_batched(_handle, args...);
  });
}
    
template <typename T>
cudaTask cublasFlowCapturer::gemm_batched2(
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
template <typename... Ts>
cudaTask cublasFlowCapturer::c_gemm_batched(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_c_gemm_batched(_handle, args...);
  });
}
    
template <typename T>
cudaTask cublasFlowCapturer::c_gemm_batched_2(
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
template <typename... Ts>
cudaTask cublasFlowCapturer::gemm_sbatched(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_gemm_sbatched(_handle, args...);
  });
}
    
template <typename T>
cudaTask cublasFlowCapturer::gemm_sbatched2(
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A, int lda, long long int sA,
  const T *B, int ldb, long long int sB,
  const T *beta,
  T *C, int ldc, long long int sC,
  int bc
) {
  return on([this, ta, tb, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, bs] 
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
template <typename... Ts>
cudaTask cublasFlowCapturer::c_gemm_sbatched(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_c_gemm_sbatched(_handle, args...);
  });
}

template <typename T>
cudaTask cublasFlowCapturer::c_gemm_sbatched2(
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A, int lda, long long int sA,
  const T *B, int ldb, long long int sB,
  const T *beta,
  T *C, int ldc, long long int sC,
  int bc
){
  return on([this, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, bc] 
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


