#pragma once

#include "cublas_handle.hpp"

namespace tf {

// ---------------------------------------------------------------------------- 
// cublasFlowCapturere level-1 functions
// ---------------------------------------------------------------------------- 

// Function: amax
template <typename T>
cudaTask cublasFlowCapturer::amax(
  int n, const T* x, int incx, int* result
) {
  return factory()->on([this, n, x, incx, result] (cudaStream_t stream) mutable {
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
template <typename T>
cudaTask cublasFlowCapturer::amin(
  int n, const T* x, int incx, int* result
) {
  return factory()->on([this, n, x, incx, result] (cudaStream_t stream) mutable {
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
template <typename T>
cudaTask cublasFlowCapturer::asum(
  int n, const T* x, int incx, T* result
) {
  return factory()->on([this, n, x, incx, result] (cudaStream_t stream) mutable {
    _stream(stream);
    cublasStatus_t stat;
    if constexpr(std::is_same_v<T, float>) {
      stat = cublasSasum(_handle, n, x, incx, result);
    }
    else if constexpr(std::is_same_v<T, double>) {
      stat = cublasDasum(_handle, n, x, incx, result);
    }
    else {
      static_assert(dependent_false_v<T>, "unknown cublas data type");
    }

    TF_CHECK_CUBLAS(stat, "failed to run cublas<t>asum");
  });
}

// Function: axpy
template <typename T>
cudaTask cublasFlowCapturer::axpy(
  int n, const T *alpha, const T *x, int incx, T *y, int incy
) {
  return factory()->on([this, n, alpha, x, incx, y, incy] (cudaStream_t stream) mutable {
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
template <typename T>
cudaTask cublasFlowCapturer::vcopy(
  int n, const T* x, int incx, T* y, int incy
) {
  return factory()->on([this, n, x, incx, y, incy] (cudaStream_t stream) mutable {
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
template <typename T>
cudaTask cublasFlowCapturer::dot(
  int n, const T* x, int incx, const T* y, int incy, T* result
) {
  return factory()->on([this, n, x, incx, y, incy, result] (cudaStream_t stream) mutable {
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

template <typename T>
cudaTask cublasFlowCapturer::nrm2(int n, const T* x, int incx, T* result) {
  return factory()->on([this, n, x, incx, result] (cudaStream_t stream) mutable {
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
template <typename T>
cudaTask cublasFlowCapturer::scal(int n, const T* scalar, T* x, int incx) {
  return factory()->on([this, n, scalar, x, incx] (cudaStream_t stream) mutable {
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

template <typename T>
cudaTask cublasFlowCapturer::swap(int n, T* x, int incx, T* y, int incy) {
  return factory()->on([this, n, x, incx, y, incy] (cudaStream_t stream) mutable {
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

}  // end of namespace tf -----------------------------------------------------



