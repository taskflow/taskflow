#pragma once

#include "cuda_blas_error.hpp"

namespace tf {

/**
@class cudaBLASHandleManager

@brief class object to manage a cuBLAS handle per device

Creating a cuBLAS handle on a device is expensive (e.g., 200ms)
and it is desirable to reuse the handle as much as possible.
The manager class abstracts the creation and destroy of a cuBLAS handle 
on each device.
*/
class cudaBLASHandleManager {
   
  public:
    
    /**
    @brief constructs a cuBLAS handle manager
    */
    cudaBLASHandleManager() = default;
    
    /**
    @brief disables copy constructor
    */
    cudaBLASHandleManager(const cudaBLASHandleManager&) = delete;
    
    /**
    @brief disables move constructor
    */
    cudaBLASHandleManager(cudaBLASHandleManager&&) = delete;
    
    /**
    @brief disables copy assignment
    */
    cudaBLASHandleManager& operator = (const cudaBLASHandleManager&) = delete;
    
    /**
    @brief disables move assignment
    */
    cudaBLASHandleManager& operator = (cudaBLASHandleManager&&) = delete;
    
    /**
    @brief destructs the manager object and destroys the associated cuBLAS handles
    */
    ~cudaBLASHandleManager() {
      clear();
    }

    /**
    @brief destroys the associated cuBLAS handles

    The method destroys all associated handles to this manager
    by calling the @c cublasDestroy().
    */
    void clear() {
      for(auto& [d, h] : _handles) {
        cudaScopedDevice ctx(d);
        cublasDestroy(h);
      }
      _handles.clear();
    }
    
    /**
    @brief gets a cuBLAS handle pertaining to the given device

    The method initializes the handle to the cuBLAS library context 
    by calling the @c cublasCreate() function. 
    */
    cublasHandle_t get(int d) {
      if(auto itr = _handles.find(d); itr == _handles.end()) {
        cudaScopedDevice ctx(d);
        cublasHandle_t handle;
        auto stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("cublas initialization failed\n");
        }
        _handles[d] = handle;
        return handle;
      }
      else return itr->second;
    }
    
  private: 
    
    std::unordered_map<int, cublasHandle_t> _handles;
};

/**
@brief per thread cuBLAS handle manager
*/
inline thread_local cudaBLASHandleManager cuda_blas_per_thread_handle_manager;

/**
@brief gets per-thread cublas handle
*/
inline cublasHandle_t cuda_blas_per_thread_handle(int d) {
  return cuda_blas_per_thread_handle_manager.get(d);
}

}  // end of namespace tf -----------------------------------------------------


