#pragma once

#include "cublas_handle.hpp"

namespace tf {

/** 
@file cublas_level1.hpp
*/

// ----------------------------------------------------------------------------
// amax
// ----------------------------------------------------------------------------

/**
@brief finds the smallest index of the element of the maximum absolute magnitude

@tparam T data type

@param handle cublas library handle
@param n number of elements in vector @c x
@param x pointer to the memory address of the vector
@param incx stride between consecutive elements of @c x
@param result the resulting index (1-based indexing)
*/
template <typename T>
void cublas_amax(
  cublasHandle_t handle, int n, const T* x, int incx, int* result
) {
  cublasStatus_t stat;
  if constexpr(std::is_same_v<T, float>) {
    stat = cublasIsamax(handle, n, x, incx, result);
  }
  else if constexpr(std::is_same_v<T, double>) {
    stat = cublasIdamax(handle, n, x, incx, result);
  }
  else {
    static_assert(dependent_false_v<T>, "unknown cublas data type");
  }

  TF_CHECK_CUBLAS(stat, "failed to run cublas<t>amax");
}

/**
@brief finds the smallest index of the element of the minimum absolute magnitude

@tparam T data type

@param handle cublas library handle
@param n number of elements in vector @c x
@param x pointer to the memory address of the vector
@param incx stride between consecutive elements of @c x
@param result the resulting index (1-based indexing)
*/
template <typename T>
void cublas_amin(
  cublasHandle_t handle, int n, const T* x, int incx, int* result
) {
  cublasStatus_t stat;
  if constexpr(std::is_same_v<T, float>) {
    stat = cublasIsamin(handle, n, x, incx, result);
  }
  else if constexpr(std::is_same_v<T, double>) {
    stat = cublasIdamin(handle, n, x, incx, result);
  }
  else {
    static_assert(dependent_false_v<T>, "unknown cublas data type");
  }

  TF_CHECK_CUBLAS(stat, "failed to run cublas<t>amin");
}

/**
@brief finds the sum of absolute values of the elements over a vector

@tparam T data type

@param handle cublas library handle
@param n number of elements in vector @c x
@param x pointer to the memory address of the vector
@param incx stride between consecutive elements of @c x
@param result the result
*/
template <typename T>
void cublas_asum(
  cublasHandle_t handle, int n, const T* x, int incx, T* result
) {
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
}

/**
@brief multiples a vector by a scalar and adds it to a vector

This function multiplies the vector @c x by the scalar @c alpha and 
adds it to the vector @c y overwriting the latest vector with the result. 
Hence, the performed operation is:

  <tt>y[j] = alpha * x[k] + y[j]</tt>, 
  
where @c j and @c k are indices of @c n elements with step sizes 
@c incy and @c incx.

@tparam T data type

@param handle cublas library handle
@param n number of elements in vectors @c x and @c y
@param alpha scalar used to multiplication
@param x pointer to the memory address of the vector @c x
@param incx stride between consecutive elements of @c x
@param y pointer to the memory address of the vector @c y
@param incy stride between consecutive elements of @c y
*/
template <typename T>
void cublas_axpy(
  cublasHandle_t handle, 
  int n, const T *alpha, const T *x, int incx, T *y, int incy
) {
  cublasStatus_t stat;
  if constexpr(std::is_same_v<T, float>) {
    stat = cublasSaxpy(handle, n, alpha, x, incx, y, incy);
  }
  else if constexpr(std::is_same_v<T, double>) {
    stat = cublasDaxpy(handle, n, alpha, x, incx, y, incy);
  }
  else {
    static_assert(dependent_false_v<T>, "unknown cublas data type");
  }

  TF_CHECK_CUBLAS(stat, "failed to run cublas<t>axpy");
}

/**
@brief copies a vector to another vector

This function copies @c n elements from a vector @c x of a step size @c incx 
to another vector @c y of step size @c incy.
 
adds it to the vector @c y overwriting the latest vector with the result. 
Hence, the performed operation is:

  <tt>y[j] = x[k]</tt>, 
  
where @c j and @c k are indices of @c n elements with step sizes 
@c incy and @c incx.

@tparam T data type

@param handle cublas library handle
@param n number of elements to copy
@param x pointer to the memory address of the vector @c x
@param incx stride between consecutive elements of @c x
@param y pointer to the memory address of the vector @c y
@param incy stride between consecutive elements of @c y
*/
template <typename T>
void cublas_copy(
  cublasHandle_t handle, int n, const T* x, int incx, T* y, int incy
) {
  cublasStatus_t stat;
  if constexpr(std::is_same_v<T, float>) {
    stat = cublasScopy(handle, n, x, incx, y, incy);
  }
  else if constexpr(std::is_same_v<T, double>) {
    stat = cublasDcopy(handle, n, x, incx, y, incy);
  }
  else {
    static_assert(dependent_false_v<T>, "unknown cublas data type");
  }

  TF_CHECK_CUBLAS(stat, "failed to run cublas<t>copy");
}

/**
@brief computes the dot product of two vectors

@tparam T data type

@param handle cublas library handle
@param n number of elements to perform the dot product
@param x pointer to the memory address of the vector @c x
@param incx stride between consecutive elements of @c x
@param y pointer to the memory address of the vector @c y
@param incy stride between consecutive elements of @c y
@param result the resulting dot product
*/
template <typename T>
void cublas_dot(
  cublasHandle_t handle, int n, 
  const T* x, int incx, 
  const T* y, int incy,
  T* result
) {
  cublasStatus_t stat;
  if constexpr(std::is_same_v<T, float>) {
    stat = cublasSdot(handle, n, x, incx, y, incy, result);
  }
  else if constexpr(std::is_same_v<T, double>) {
    stat = cublasDdot(handle, n, x, incx, y, incy, result);
  }
  else {
    static_assert(dependent_false_v<T>, "unknown cublas data type");
  }

  TF_CHECK_CUBLAS(stat, "failed to run cublas<t>dot");
}

/**
@brief computes the Euclidean norm of a vector

@tparam T data type

@param handle cublas library handle
@param n number of elements in vector @c x
@param x pointer to the memory address of the vector
@param incx stride between consecutive elements of @c x
@param result the result
*/
template <typename T>
void cublas_nrm2(
  cublasHandle_t handle, int n, const T* x, int incx, T* result
) {
  cublasStatus_t stat;
  if constexpr(std::is_same_v<T, float>) {
    stat = cublasSnrm2(handle, n, x, incx, result);
  }
  else if constexpr(std::is_same_v<T, double>) {
    stat = cublasDnrm2(handle, n, x, incx, result);
  }
  else {
    static_assert(dependent_false_v<T>, "unknown cublas data type");
  }

  TF_CHECK_CUBLAS(stat, "failed to run cublas<t>nrm2");
}

/**
@brief scales a vector by a scalar

@tparam T data type

@param handle cublas library handle
@param n number of elements in vector @c x
@param scalar scalar used for multiplication
@param x pointer to the memory address of the vector
@param incx stride between consecutive elements of @c x
*/
template <typename T>
void cublas_scal(
  cublasHandle_t handle, int n, const T* scalar, T* x, int incx
) {
  cublasStatus_t stat;
  if constexpr(std::is_same_v<T, float>) {
    stat = cublasSscal(handle, n, scalar, x, incx);
  }
  else if constexpr(std::is_same_v<T, double>) {
    stat = cublasDscal(handle, n, scalar, x, incx);
  }
  else {
    static_assert(dependent_false_v<T>, "unknown cublas data type");
  }

  TF_CHECK_CUBLAS(stat, "failed to run cublas<t>scal");
}

/**
@brief swap elements between two vectors

This function interchanges the elements of vectors @c x and @c y. 
Hence, the performed operation is:

<tt>y[j] <-> x[k]</tt>,

where @c j is the index of element in @c y with a step size @c incy and
@c k is the index of element in @c x with a step size @c incx.

@tparam T data type

@param handle cublas library handle
@param n number of elements to perform the dot product
@param x pointer to the memory address of the vector @c x
@param incx stride between consecutive elements of @c x
@param y pointer to the memory address of the vector @c y
@param incy stride between consecutive elements of @c y
*/
template <typename T>
void cublas_swap(
  cublasHandle_t handle, int n, T* x, int incx, T* y, int incy
) {
  cublasStatus_t stat;
  if constexpr(std::is_same_v<T, float>) {
    stat = cublasSswap(handle, n, x, incx, y, incy);
  }
  else if constexpr(std::is_same_v<T, double>) {
    stat = cublasDswap(handle, n, x, incx, y, incy);
  }
  else {
    static_assert(dependent_false_v<T>, "unknown cublas data type");
  }

  TF_CHECK_CUBLAS(stat, "failed to run cublas<t>swap");
}

}  // end of namespace tf -----------------------------------------------------



