#pragma once

#include "cublas_helper.hpp"
#include "cublas_level1.hpp"
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
    @brief finds the sum of absolute values of elements in a vector
    
    This method effectively calls tf::cublas_asum with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask asum(Ts&&... args);
    
    /** 
    @brief multiplies a vector by a scalar and adds it to a vector
    
    This method effectively calls tf::cublas_axpy with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask axpy(Ts&&... args);

    /** 
    @brief copies a vector to another vector
    
    This method effectively calls tf::cublas_copy with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask vcopy(Ts&&... args);
    
    /** 
    @brief computes the dot product of two vectors
    
    This method effectively calls tf::cublas_dot with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask dot(Ts&&... args);
    
    /** 
    @brief computes the Euclidean norm of a vector
    
    This method effectively calls tf::cublas_nrm2 with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask nrm2(Ts&&... args);
    
    /** 
    @brief multiples a vector by a scalar
    
    This method effectively calls tf::cublas_scal with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask scal(Ts&&... args);
    
    /** 
    @brief swaps the elements of two vectors
    
    This method effectively calls tf::cublas_swap with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask swap(Ts&&... args);

    // TODO: axpy, copy, dot, nrm2, scal, sawp, etc.

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
    @brief performs matrix-matrix addition/transposition on row-major layout
    
    This method effectively calls tf::cublas_c_geam with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is manaed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask c_geam(Ts&&... args);
    
    /** 
    @brief performs matrix-matrix multiplication on column-major layout
    
    This method effectively calls tf::cublas_gemm with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is managed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask gemm(Ts&&... args);

    /** 
    @brief performs matrix-matrix multiplication on C-styled row-major layout
    
    This method effectively calls tf::cublas_c_gemm with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is managed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask c_gemm(Ts&&... args);

    /**
    @brief performs batched matrix-matrix multiplication on column-major layout
    
    This method effectively calls tf::cublas_gemm_batched with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is managed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask gemm_batched(Ts&&... args);
    
    /**
    @brief performs batched matrix-matrix multiplication on C-styled row-major layout
    
    This method effectively calls tf::cublas_c_gemm_batched with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is managed by the %cublasFlowCapturer.
    */ 
    template <typename... Ts>
    cudaTask c_gemm_batched(Ts&&... args);
    
    /**
    @brief performs batched matrix-matrix multiplication on column-major layout
           with strided memory access
    
    This method effectively calls tf::cublas_gemm_sbatched with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is managed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask gemm_sbatched(Ts&&... args);
    
    /** 
    @brief performs batched matrix-matrix multiplication on C-styled row-major 
           layout with strided memory access
    
    This method effectively calls tf::cublas_c_gemm_sbatched with packed parameters,
    <tt>(handle, args...)</tt>, where @c handle is managed by the %cublasFlowCapturer.
    */
    template <typename... Ts>
    cudaTask c_gemm_sbatched(Ts&&... args);
    
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

// Function: amin
template <typename... Ts>
cudaTask cublasFlowCapturer::amin(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_amin(_handle, args...);
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

// Function: axpy
template <typename... Ts>
cudaTask cublasFlowCapturer::axpy(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_axpy(_handle, args...);
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

// Function: dot
template <typename... Ts>
cudaTask cublasFlowCapturer::dot(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_dot(_handle, args...);
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

// Function: scal
template <typename... Ts>
cudaTask cublasFlowCapturer::scal(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_scal(_handle, args...);
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

// Function: geam
template <typename... Ts>
cudaTask cublasFlowCapturer::c_geam(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_c_geam(_handle, args...);
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

// Function: c_gemm
template <typename... Ts>
cudaTask cublasFlowCapturer::c_gemm(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_c_gemm(_handle, args...);
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

// Function: c_gemm_batched
template <typename... Ts>
cudaTask cublasFlowCapturer::c_gemm_batched(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_c_gemm_batched(_handle, args...);
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

// Function: c_gemm_sbatched (strided)    
template <typename... Ts>
cudaTask cublasFlowCapturer::c_gemm_sbatched(Ts&&... args) {
  return on([this, args...] (cudaStream_t stream) mutable {
    _stream(stream);
    cublas_c_gemm_sbatched(_handle, args...);
  });
}

}  // end of namespace tf -----------------------------------------------------


