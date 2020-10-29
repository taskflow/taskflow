#pragma once

#include "cublas_handle.hpp"

namespace tf {

template <typename T>
struct is_cublas_data_type : std::false_type {};

template <>
struct is_cublas_data_type<float> : std::true_type {};

template <>
struct is_cublas_data_type<double> : std::true_type {};

template <>
struct is_cublas_data_type<cuComplex> : std::true_type {};

template <>
struct is_cublas_data_type<cuDoubleComplex> : std::true_type {};

template <typename T>
inline constexpr bool is_cublas_data_type_v = is_cublas_data_type<T>::value;

// ----------------------------------------------------------------------------
// cublasFlow definition
// ----------------------------------------------------------------------------

/**
@brief class object to construct a cuBLAS task graph

A cublasFlow provides a higher-level interface over the cuBLAS library.
By default, %cublasFlow uses column-major storage, and 1-based indexing. 
Since C/C++ uses row-major storage, we provides methods of prefix @c c_* 
for the equivalents. 

All pointers used to %cublasFlow methods must be on device or managed 
(i.e., @c cudaMallocManaged),
including scalars, @c alpha and @c beta, input data and output data pointers.

Currently, we %cublasFlow supports only float and double data types.

*/
class cublasFlow {

  friend class cudaFlow;

  public:
    
    // TODO: need to add the following methods
    // setvec  : cublasSetVectorAsync
    // getvec  : cublasGetVectorAsync
    // setmat  : cublasSetMatrixAsync (column major)
    // c_setmat: cublasSetMatrixAsync (row major)
    // getmat  : cublasGetMatrixAsync (column major)
    // c_getmat: cublasGetMatrixAsync (row major)
    
    // ------------------------------------------------------------------------
    // Helper methods
    // ------------------------------------------------------------------------
    
    /**
    @brief runs a callable with only a single kernel thread

    @tparam C callable type

    @param callable callable to run by a single kernel thread
    */
    template <typename C>
    cudaTask single_task(C&& callable);
    
    // ------------------------------------------------------------------------
    // Level-1 vector-vector operations
    // ------------------------------------------------------------------------
    template <typename T>
    cudaTask amax(int n, const T* x, int incx, int* result) {
      auto node = _graph.emplace_back(_graph,
        std::in_place_type_t<cudaNode::Capture>{},
        [this, n, x, incx, result] (cudaStream_t stream) mutable {
          _stream(stream);
          TF_CHECK_CUBLAS(
            cublasIsamax(_native_handle, n, x, incx, result),
            "failed to capture cublasIsamax"
          );
        }
      );
      return cudaTask(node);
    }

    // TODO: amin, asum, axpy, copy, dot, nrm2, scal, sawp, etc.

    // ------------------------------------------------------------------------
    // TODO Level-2 matrix_vector operations
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // TODO Level-3 matrix-matrix operations
    // ------------------------------------------------------------------------
    
    /** 
    @brief constructs a task to perform matrix-matrix multiplication

    This method constructs a task to perform matrix-matrix multiplication:

    <tt>C = alpha * op (A) * op (B) + beta * C</tt>,

    where @c alpha and @c beta are scalars, and @c A, @c B, and @c C
    are 2D matrices stored in column-major format 
    with dimension @c op(A) as @c m by @c k,
    dimension @c op(B) as @c k by @c n, and @c C as @c m by @c n.

    @tparam T data type
    @param transa transport operation @c op(A)
    @param transb transport operation @c op(B)
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
    cudaTask gemm(
      cublasOperation_t transa, cublasOperation_t transb,
      int m, int n, int k,
      const T *alpha,
      const T *A, int lda,
      const T *B, int ldb,
      const T *beta,
      T *C, int ldc
    );

    /** 
    @brief similar to gemm but operates on C-styled row-major layout
    */
    template <typename T>
    cudaTask c_gemm(
      cublasOperation_t transa, cublasOperation_t transb,
      int m, int n, int k,
      const T *alpha,
      const T *A, int lda,
      const T *B, int ldb,
      const T *beta,
      T *C, int ldc
    );

    /**
    @brief constructs a task to perform matrix-matrix multiplication 
           over a batch of matrices
    
    @tparam T data type
    @param transa transport operation @c op(A[i])
    @param transb transport operation @c op(B[i])
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
    @param batch_count batch size (number of matrices)

    The batch must be @i uniform.
    All instances in the batch must have the same dimensions <tt>(m, n, k)</tt>, 
    leading dimensions <tt>(lda, ldb, ldc)</tt> and transpositions 
    <tt>(transa, transb)</tt> for their respective @c A, @c B and @c C matrices. 
    The address of the input matrices and the output matrix of each instance 
    of the batch are read from arrays of pointers passed to the function by the caller.

    <tt>C[i]= alpha * op (A[i]) * op (B[i]) + beta * C[i], i in [0, batch_count)</tt>,

    where @c alpha and @c beta are scalars, and @c A[i], @c B[i], and @c C[i]
    are 2D matrices stored in column-major format 
    with dimension @c op(A) as @c m by @c k,
    dimension @c op(B) as @c k by @c n, and @c C as @c m by @c n.
    
    */
    template <typename T>
    cudaTask gemm_batched(
      cublasOperation_t transa, cublasOperation_t transb,
      int m, int n, int k,
      const T *alpha,
      const T *A[], int lda,
      const T *B[], int ldb,
      const T *beta,
      T *C[], int ldc,
      int batch_count
    );
    
    /**
    @brief similar to gemm_batched but operates on C-styled row-major layout
    */ 
    template <typename T>
    cudaTask c_gemm_batched(
      cublasOperation_t transa, cublasOperation_t transb,
      int m, int n, int k,
      const T *alpha,
      const T *A[], int lda,
      const T *B[], int ldb,
      const T *beta,
      T *C[], int ldc,
      int batch_count
    );
    
    /**
    @brief constructs a task to perform matrix-matrix multiplication 
           over a batch of matrices with strided memory access
    
    Here, we use @c A[i], @c B[i], @c C[i] as notation 
    for A, B and C matrices in the @c i-th instance of the batch, 
    implicitly assuming they are respectively address offsets 
    @c strideA, @c strideB, @c strideC away from @c A[i-1], @c B[i-1], @c C[i-1].

    @tparam T data type
    @param transa transport operation @c op(A[i])
    @param transb transport operation @c op(B[i])
    @param m number of rows of matrix @c C[i] and @c op(A[i])
    @param n number of columns of matrix @c C[i] and @c op(B[i])
    @param k number of columns of @c op(A[i]) and rows of @c op(B[i])
    @param alpha pointer to the @c alpha scalar
    @param A pointer to @c A batch
    @param lda leading dimension of 2D array used to store the matrix @c A[i]
    @param strideA address offset between @c A[i] and @c A[i+1]
    @param B pointer to @c B batch
    @param ldb leading dimension of 2D array used to store the matrix @c B[i]
    @param strideB address offset between @c B[i] and @c B[i+1]
    @param beta pointer to the @c beta scalar
    @param C pointer to @c C batch
    @param ldc leading dimension of 2D array used to store the matrix @c C[i]
    @param strideC address offset between @c C[i] and @c C[i+1]
    @param batch_count batch size (number of matrices)

    The batch must be @i uniform. 
    All instances in the batch must have the same dimensions <tt>(m, n, k)</tt>, 
    leading dimensions <tt>(lda, ldb, ldc)</tt> and transpositions 
    <tt>(transa, transb)</tt> for their respective @c A, @c B and @c C matrices. 
    Input matrices @c A, @c B and output matrix @c C for each instance of the batch 
    are located at fixed address offsets from their locations in the previous instance. 
    Pointers to @c A, @c B and @c C matrices for the first instance are passed 
    to the function by the user along with the address offsets - 
    @c strideA, @c strideB and @c strideC that determine the locations 
    of input and output matrices in future instances.

    <tt>C + i*strideC = alpha * op (A + i*strideA) * op (B + i*strideB) 
                      + beta * (C + i*strideC), i in [0, batch_count)</tt>,

    where @c alpha and @c beta are scalars, and @c A[i], @c B[i], and @c C[i]
    are 2D matrices stored in column-major format 
    with dimension @c op(A) as @c m by @c k,
    dimension @c op(B) as @c k by @c n, and @c C as @c m by @c n.

    On certain problem sizes, it might be advantageous to create multiple gemm tasks
    to take advantage of concurrent kernels, rather than this method.
    */
    template <typename T>
    cudaTask gemm_batched(
      cublasOperation_t transa, cublasOperation_t transb,
      int m, int n, int k,
      const T *alpha,
      const T *A, int lda, long long int strideA,
      const T *B, int ldb, long long int strideB,
      const T *beta,
      T *C, int ldc, long long int strideC,
      int batch_count
    );
    
    /** 
    @brief similar to gemm_batached but operates on C-styled row-major layout
    */
    template <typename T>
    cudaTask c_gemm_batched(
      cublasOperation_t transa, cublasOperation_t transb,
      int m, int n, int k,
      const T *alpha,
      const T *A, int lda, long long int strideA,
      const T *B, int ldb, long long int strideB,
      const T *beta,
      T *C, int ldc, long long int strideC,
      int batch_count
    );
    
    // reference: https://docs.anaconda.com/accelerate/2.0/cublas/
    
  private:

    cudaGraph& _graph;

    cublasHandle_t _native_handle;

    cublasFlow(cudaGraph&, cublasHandle_t);

    void _stream(cudaStream_t);
};

// Constructor
inline cublasFlow::cublasFlow(cudaGraph& graph, cublasHandle_t handle) : 
  _graph {graph}, _native_handle {handle} {
}

// Procedure: _stream
inline void cublasFlow::_stream(cudaStream_t stream) {
  TF_CHECK_CUBLAS(
    cublasSetStream(_native_handle, stream), "failed to set cublas stream"
  );
}

// ---------------------------------------------------------------------------- 
// Helper functions
// ---------------------------------------------------------------------------- 

// Function: single_task
template <typename C>
cudaTask cublasFlow::single_task(C&& callable) {
  auto node = _graph.emplace_back(_graph,
    std::in_place_type_t<cudaNode::Capture>{},
    [c=std::forward<C>(callable)] (cudaStream_t stream) mutable {
      cuda_single_task<C><<<1, 1, 0, stream>>>(c);
    }
  );
  return cudaTask(node);
}
    
// ---------------------------------------------------------------------------- 
// Level-3 functions
// ---------------------------------------------------------------------------- 

// Function: gemm
template <typename T>
cudaTask cublasFlow::gemm(
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A, int lda,
  const T *B, int ldb,
  const T *beta,
  T *C, int ldc
) {
  auto node = _graph.emplace_back(_graph,
    std::in_place_type_t<cudaNode::Capture>{},
    [this, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc] 
    (cudaStream_t stream) mutable {

      _stream(stream);

      cublasStatus_t stat;

      if constexpr(std::is_same_v<T, float>) {
        stat = cublasSgemm(_native_handle,
          ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
        );
      }
      else if constexpr(std::is_same_v<T, double>) {
        stat = cublasDgemm(_native_handle,
          ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
        );
      }
      else {
        static_assert(dependent_false_v<T>, "unknown cublas data type");
      }

      TF_CHECK_CUBLAS(stat, "failed to capture gemm");
    }
  );
  return cudaTask(node);
}

// Function: c_gemm
template <typename T>
cudaTask cublasFlow::c_gemm(
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A, int lda,
  const T *B, int ldb,
  const T *beta,
  T *C, int ldc
) {
  auto node = _graph.emplace_back(_graph,
    std::in_place_type_t<cudaNode::Capture>{},
    [this, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc] 
    (cudaStream_t stream) mutable {

      _stream(stream);

      cublasStatus_t stat;

      if constexpr(std::is_same_v<T, float>) {
        stat = cublasSgemm(_native_handle,
          tb, ta, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc
        );
      }
      else if constexpr(std::is_same_v<T, double>) {
        stat = cublasDgemm(_native_handle,
          tb, ta, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc
        );
      }
      else {
        static_assert(dependent_false_v<T>, "unknown cublas data type");
      }

      TF_CHECK_CUBLAS(stat, "failed to capture c_gemm");
    }
  );
  return cudaTask(node);
}
    
// Function: gemm_batched
template <typename T>
cudaTask cublasFlow::gemm_batched(
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A[], int lda,
  const T *B[], int ldb,
  const T *beta,
  T *C[], int ldc,
  int batch_count
) {
  auto node = _graph.emplace_back(_graph,
    std::in_place_type_t<cudaNode::Capture>{},
    [this, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count] 
    (cudaStream_t stream) mutable {

      _stream(stream);

      cublasStatus_t stat;

      if constexpr(std::is_same_v<T, float>) {
        stat = cublasSgemmBatched(_native_handle,
          ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count
        );
      }
      else if constexpr(std::is_same_v<T, double>) {
        stat = cublasDgemmBatched(_native_handle,
          ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count
        );
      }
      else static_assert(dependent_false_v<T>, "unknown cublas data type");

      TF_CHECK_CUBLAS(stat, "failed to capture gemm_batched");
    }
  );
  return cudaTask(node);
}

// Function: c_gemm_batched
template <typename T>
cudaTask cublasFlow::c_gemm_batched(
  cublasOperation_t ta, cublasOperation_t tb,
  int m, int n, int k,
  const T *alpha,
  const T *A[], int lda,
  const T *B[], int ldb,
  const T *beta,
  T *C[], int ldc,
  int batch_count
) {
  auto node = _graph.emplace_back(_graph,
    std::in_place_type_t<cudaNode::Capture>{},
    [this, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count] 
    (cudaStream_t stream) mutable {

      _stream(stream);

      cublasStatus_t stat;

      if constexpr(std::is_same_v<T, float>) {
        stat = cublasSgemmBatched(_native_handle,
          tb, ta, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, batch_count
        );
      }
      else if constexpr(std::is_same_v<T, double>) {
        stat = cublasDgemmBatched(_native_handle,
          tb, ta, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, batch_count
        );
      }
      else static_assert(dependent_false_v<T>, "unknown cublas data type");

      TF_CHECK_CUBLAS(stat, "failed to capture c_gemm_batched");
    }
  );
  return cudaTask(node);
}

// ----------------------------------------------------------------------------
// cudaFlow 
// ----------------------------------------------------------------------------

// Function: subflow
template <typename C, std::enable_if_t<is_cublasflow_v<C>, void>*>
cudaTask cudaFlow::subflow(C&& c) {
  
  // insert a subflow node
  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Subflow>{}
  );
  
  auto& node_handle = std::get<cudaNode::Subflow>(node->_handle);
  
  // acquire per-thread cublas handle and stream
  cublasScopedPerThreadHandle cublas_handle(_device);
  cudaScopedPerThreadStream stream(_device);
  
  // turn the stream into capture and associate it with the cublas handle
  TF_CHECK_CUDA(
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), 
    "failed to turn stream into capture mode"
  );
  
  // construct a cublas flow from the callable
  cublasFlow cbf(node_handle.graph, cublas_handle);

  c(cbf);

  // TODO: need an efficient algorithm
  auto ordered = node_handle.graph._toposort();
  for(auto& node : ordered) {
    //std::cout << node->_name << '\n';
    std::get<cudaNode::Capture>(node->_handle).work(stream);  
  }

  cudaGraph_t graph;
  
  // stop the capture to get a cuda graph
  TF_CHECK_CUDA(cudaStreamEndCapture(stream, &graph), "failed to end capture");

  //cuda_dump_graph(std::cout, graph);

  TF_CHECK_CUDA(
    cudaGraphAddChildGraphNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, graph
    ), 
    "failed to add a cuda subflow task"
  );
  
  TF_CHECK_CUDA(cudaGraphDestroy(graph), "failed to destroy captured graph");
  
  return cudaTask(node);
}

}  // end of namespace tf -----------------------------------------------------


