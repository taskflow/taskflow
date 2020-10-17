#pragma once

#include "cublas_error.hpp"

namespace tf {

/**
@class cublasHandleManager

@brief class object to manage a cuBLAS handle per device

Creating a cuBLAS handle on a device is expensive (e.g., 200ms)
and it is desirable to reuse the handle as much as possible.
The manager class abstracts the creation and destroy of a cuBLAS handle 
on each device.
*/
class cublasHandleManager {
   
  public:
    
    /**
    @brief constructs a cuBLAS handle manager
    */
    cublasHandleManager() = default;
    
    /**
    @brief disables copy constructor
    */
    cublasHandleManager(const cublasHandleManager&) = delete;
    
    /**
    @brief disables move constructor
    */
    cublasHandleManager(cublasHandleManager&&) = delete;
    
    /**
    @brief disables copy assignment
    */
    cublasHandleManager& operator = (const cublasHandleManager&) = delete;
    
    /**
    @brief disables move assignment
    */
    cublasHandleManager& operator = (cublasHandleManager&&) = delete;
    
    /**
    @brief destructs the manager object and destroys the associated cuBLAS handles
    */
    ~cublasHandleManager() {
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
inline thread_local cublasHandleManager cublas_per_thread_handle_manager;

/**
@brief gets per-thread cublas handle
*/
inline cublasHandle_t cublas_per_thread_handle(int d) {
  return cublas_per_thread_handle_manager.get(d);
}

}  // end of namespace tf -----------------------------------------------------
