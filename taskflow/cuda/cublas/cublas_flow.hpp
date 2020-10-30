#pragma once

#include "cublas_helper.hpp"
#include "cublas_level3.hpp"

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
@class cublasFlow

@brief class object to construct a cuBLAS task graph

A %cublasFlow provides a higher-level interface over the cuBLAS library
and hide concurrency details from users.

All pointers used to %cublasFlow methods must be in GPU memory space or managed 
(i.e., @c cudaMallocManaged),
including scalars, @c alpha and @c beta, input data and output data pointers.

Currently,  %cublasFlow supports only float and double data types.
*/
class cublasFlow : public cudaFlowCapturer {

  friend class cudaFlow;

  public:
    
    /**
    @brief gets the native cublas handle associated with this %cublasFlow
    */
    cublasHandle_t native_handle();
    
    // ------------------------------------------------------------------------
    // Helper methods
    // ------------------------------------------------------------------------
    
    /**
    @brief copies vector data from host to device

    This method effectively calls <tt>cublas_setvec_async(stream, args...)</tt>
    with @c stream managed by the %cublasFlow.
    */
    template <typename... Ts>
    cudaTask setvec(Ts&&... args);
    
    /**
    @brief copies vector data from device to host

    This method effectively calls <tt>cublas_getvec_async(stream, args...)</tt>
    with @c stream managed by the %cublasFlow.
    */
    template <typename... Ts>
    cudaTask getvec(Ts&&... args);
    
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
            cublasIsamax(_handle, n, x, incx, result),
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
    @brief performs matrix-matrix multiplication on column-major layout
    
    This method effectively calls <tt>cublas_gemm(handle, args...)</tt>
    with @c handle managed by the %cublasFlow.
    */
    template <typename... Ts>
    cudaTask gemm(Ts&&... args);

    /** 
    @brief performs matrix-matrix multiplication on C-styled row-major layout
    
    This method effectively calls <tt>cublas_c_gemm(handle, args...)</tt>
    with @c handle managed by the %cublasFlow.
    */
    template <typename... Ts>
    cudaTask c_gemm(Ts&&... args);

    /**
    @brief performs batched matrix-matrix multiplication on column-major layout
    
    This method effectively calls <tt>cublas_gemm_batched(handle, args...)</tt>
    with @c handle managed by the %cublasFlow.
    */
    template <typename... Ts>
    cudaTask gemm_batched(Ts&&... args);
    
    /**
    @brief performs batched matrix-matrix multiplication on C-styled row-major layout
    
    This method effectively calls <tt>cublas_c_gemm_batched(handle, args...)</tt>
    with @c handle managed by the %cublasFlow.
    */ 
    template <typename... Ts>
    cudaTask c_gemm_batched(Ts&&... args);
    
    /**
    @brief performs batched matrix-matrix multiplication on column-major layout
           with strided memory access
    
    This method effectively calls <tt>cublas_gemm_sbatched(handle, args...)</tt>
    with @c handle managed by the %cublasFlow.
    */
    template <typename... Ts>
    cudaTask gemm_sbatched(Ts&&... args);
    
    /** 
    @brief performs batched matrix-matrix multiplication on C-styled row-major 
           layout with strided memory access
    
    This method effectively calls <tt>cublas_c_gemm_sbatched(handle, args...)</tt>
    with @c handle managed by the %cublasFlow.
    */
    template <typename... Ts>
    cudaTask c_gemm_sbatched(Ts&&... args);
    
    // reference: https://docs.anaconda.com/accelerate/2.0/cublas/

    
  private:

    //cublasHandle_t _handle;
    
    cublasScopedPerThreadHandle _handle;

    cublasFlow(cudaGraph&);

    void _stream(cudaStream_t);
};

// Constructor
inline cublasFlow::cublasFlow(cudaGraph& graph) : 
  cudaFlowCapturer {graph} {
}

// Procedure: _stream
inline void cublasFlow::_stream(cudaStream_t stream) {
  TF_CHECK_CUBLAS(
    cublasSetStream(_handle, stream), "failed to set cublas stream"
  );
}

// Function: native_handle
inline cublasHandle_t cublasFlow::native_handle() {
  return _handle;
}

// ---------------------------------------------------------------------------- 
// Helper functions
// ---------------------------------------------------------------------------- 

// Function: setvec
template <typename... Ts>
cudaTask cublasFlow::setvec(Ts&&... args) {
  return on([args...] (cudaStream_t stream) mutable {
    cublas_setvec_async(stream, args...);
  });
}

// Function: getvec
template <typename... Ts>
cudaTask cublasFlow::getvec(Ts&&... args) {
  return on([args...] (cudaStream_t stream) mutable {
    cublas_getvec_async(stream, args...);
  });
}

// ---------------------------------------------------------------------------- 
// Level-3 functions
// ---------------------------------------------------------------------------- 

// Function: gemm
template <typename... Ts>
cudaTask cublasFlow::gemm(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_gemm(_handle, args...);
  });
}

// Function: c_gemm
template <typename... Ts>
cudaTask cublasFlow::c_gemm(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_c_gemm(_handle, args...);
  });
}
    
// Function: gemm_batched
template <typename... Ts>
cudaTask cublasFlow::gemm_batched(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_gemm_batched(_handle, args...);
  });
}

// Function: c_gemm_batched
template <typename... Ts>
cudaTask cublasFlow::c_gemm_batched(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_c_gemm_batched(_handle, args...);
  });
}

// Function: gemm_sbatched (strided)    
template <typename... Ts>
cudaTask cublasFlow::gemm_sbatched(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_gemm_sbatched(_handle, args...);
  });
}

// Function: c_gemm_sbatched (strided)    
template <typename... Ts>
cudaTask cublasFlow::c_gemm_sbatched(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_c_gemm_sbatched(_handle, args...);
  });
}

// ----------------------------------------------------------------------------
// cudaFlow::cublas
// ----------------------------------------------------------------------------

// Function: cublas
template <typename C, std::enable_if_t<is_cublas_flow_v<C>, void>*>
cudaTask cudaFlow::cublas(C&& c) {
  
  // insert a subflow node
  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Subflow>{}
  );
  
  // construct a cublas flow from the callable
  auto& node_handle = std::get<cudaNode::Subflow>(node->_handle);
  cublasFlow cbf(node_handle.graph);
  c(cbf);

  // obtain the optimized captured graph
  auto captured = cbf._capture();
  //cuda_dump_graph(std::cout, captured);

  TF_CHECK_CUDA(
    cudaGraphAddChildGraphNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, captured
    ), 
    "failed to add a cublas flow task"
  );
  
  TF_CHECK_CUDA(
    cudaGraphDestroy(captured), "failed to destroy captured cublasFlow graph"
  );
  
  return cudaTask(node);
}

}  // end of namespace tf -----------------------------------------------------


