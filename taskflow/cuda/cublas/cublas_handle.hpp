#pragma once

#include "cublas_error.hpp"

namespace tf {

/**
@brief function object class to create a cublas handle.
*/
struct cublasHandleCreator {
  cublasHandle_t operator () () const {
    cublasHandle_t handle;

    TF_CHECK_CUBLAS(
      cublasCreate(&handle), "failed to create a cublas handle"
    );

    TF_CHECK_CUBLAS(
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE),
      "failed to set cublas pointer mode on device"
    );

    //std::cout << "create cublas handle " << handle << '\n';
    return handle;
  }
};

/**
@brief function object class to delete a cublas handle.
*/
struct cublasHandleDeleter {
  void operator () (cublasHandle_t ptr) const {
    //std::cout << "destroy cublas handle " << ptr << '\n';
    cublasDestroy(ptr);
  }
};

/**
@brief alias of per-thread cublas handle pool type
 */
using cublasPerThreadHandlePool = cudaPerThreadDeviceObjectPool<
  cublasHandle_t, cublasHandleCreator, cublasHandleDeleter
>;

/**
@brief per-thread cublas stream pool
*/
inline thread_local cublasPerThreadHandlePool cublas_per_thread_handle_pool;

// ----------------------------------------------------------------------------
// cublasScopedPerThreadHandle definition
// ----------------------------------------------------------------------------

/**
@brief class to provide RAII-styled guard of cublas handle acquisition

Sample usage:
    
@code{.cpp}
{
  cublasScopedPerThreadHandle handle(1);  // acquires a cublas handle on device 1

  // use handle as a normal cublas handle (cublasHandle_t)
  cublasSetStream(handle, stream);

}  // leaving the scope to release the handle back to the pool on device 1
@endcode

 */
class cublasScopedPerThreadHandle {
  
  public:

  /**
  @brief constructs a scoped handle under the given device

  The constructor acquires a handle from a per-thread handle pool.
  */
  explicit cublasScopedPerThreadHandle(int d) : 
    _ptr {cublas_per_thread_handle_pool.acquire(d)} {
  }
  
  /**
  @brief constructs a scoped handle under the default device 0.

  The constructor acquires a handle from a per-thread handle pool.
  */
  cublasScopedPerThreadHandle() : 
    _ptr {cublas_per_thread_handle_pool.acquire(cuda_get_device())} {
  }

  /**
  @brief destructs the scoped handle guard

  The destructor releases the handle to the per-thread handle pool.
  */
  ~cublasScopedPerThreadHandle() {
    cublas_per_thread_handle_pool.release(std::move(_ptr));
  }

  /**
  @brief implicit conversion to the native cublas handle (cublasHandle_t)
   */
  operator cublasHandle_t () const {
    return _ptr->object;
  }

  /**
  @brief returns the number of shared owners
   */
  long use_count() const {
    return _ptr.use_count();
  }

  private:

  std::shared_ptr<cublasPerThreadHandlePool::cudaDeviceObject> _ptr;

};



}  // end of namespace tf -----------------------------------------------------


