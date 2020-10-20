#pragma once

#include "cuda_error.hpp"

namespace tf {

/**
@brief per-thread object to manage a CUDA library handle (e.g., cuBLAS) on a device

@tparam H handle type
@tparam C function object to create a library handle
@tparam D function object to delete a library handle

Creating a CUDA library handle on a device is typically expensive (e.g., 200ms).
It is desirable to reuse the handle as much as possible.
The manager class abstracts the creation and destruction of a CUDA library 
handle on each device.

*/
template <typename H, typename C, typename D>
class cudaPerThreadHandlePool {
  
  public: 

  struct cudaHandle {

    int device;
    H native_handle;
    cudaStream_t stream;

    cudaHandle(int);
    ~cudaHandle();

    cudaHandle(const cudaHandle&) = delete;
    cudaHandle(cudaHandle&&) = delete;
  };

  private:
  
  // Master thread hold the storage to the pool.
  // Due to some ordering, cuda context may be destroyed when the master
  // program thread destroys the cuda handle.
  // Therefore, we use a decentralized approach to let child thread
  // destroy cuda handles while the master thread only keeps a weak reference
  // to those handles for reuse.
  struct cudaHandlePool {
  
    std::shared_ptr<cudaHandle> acquire(int);
    void release(int, std::weak_ptr<cudaHandle>);
  
    std::mutex mutex;
    std::unordered_map<int, std::vector<std::weak_ptr<cudaHandle>>> pool;
  };

  public:
    
    cudaPerThreadHandlePool() = default;
    
    std::shared_ptr<cudaHandle> acquire(int);

    void release(std::shared_ptr<cudaHandle>&&);

    size_t footprint_size() const;
    
  private: 

    inline static cudaHandlePool _cuda_handle_pool;

    std::unordered_set<std::shared_ptr<cudaHandle>> _footprint;
}; 

// ----------------------------------------------------------------------------
// cudaPerThreadHandle::cudaHanalde definition
// ----------------------------------------------------------------------------

template <typename H, typename C, typename D>
cudaPerThreadHandlePool<H, C, D>::cudaHandle::cudaHandle(int d) : device {d} {
  cudaScopedDevice ctx(device);
  native_handle = C{}();
  
  TF_CHECK_CUDA(cudaStreamCreate(&stream), 
    "failed to create a cublas stream on device ", device
  );
}

template <typename H, typename C, typename D>
cudaPerThreadHandlePool<H, C, D>::cudaHandle::~cudaHandle() {
  cudaScopedDevice ctx(device);
  cudaStreamDestroy(stream);
  D{}(native_handle);
}

// ----------------------------------------------------------------------------
// cudaPerThreadHandle::cudaHanaldePool definition
// ----------------------------------------------------------------------------

template <typename H, typename C, typename D>
std::shared_ptr<typename cudaPerThreadHandlePool<H, C, D>::cudaHandle>
cudaPerThreadHandlePool<H, C, D>::cudaHandlePool::acquire(int d) {
  std::scoped_lock lock(mutex);
  if(auto itr = pool.find(d); itr != pool.end()) {
    while(!itr->second.empty()) {
      auto sptr = itr->second.back().lock();
      itr->second.pop_back();
      if(sptr) {
        return sptr;
      }
    }
  }
  return nullptr;
}

template <typename H, typename C, typename D>
void cudaPerThreadHandlePool<H, C, D>::cudaHandlePool::release(
  int d, std::weak_ptr<cudaHandle> ptr
) {
  std::scoped_lock lock(mutex);
  pool[d].push_back(ptr);
}

// ----------------------------------------------------------------------------
// cudaPerThreadHandle definition
// ----------------------------------------------------------------------------

template <typename H, typename C, typename D>
std::shared_ptr<typename cudaPerThreadHandlePool<H, C, D>::cudaHandle> 
cudaPerThreadHandlePool<H, C, D>::acquire(int d) {

  auto ptr = _cuda_handle_pool.acquire(d);

  if(!ptr) {
    ptr = std::make_shared<cudaHandle>(d);
  }

  return ptr;
}

template <typename H, typename C, typename D>
void cudaPerThreadHandlePool<H, C, D>::release(std::shared_ptr<cudaHandle>&& ptr) {
  _cuda_handle_pool.release(ptr->device, ptr);
  _footprint.insert(std::move(ptr));
}

template <typename H, typename C, typename D>
size_t cudaPerThreadHandlePool<H, C, D>::footprint_size() const {
  return _footprint.size();
}

}  // end of namespace tf -----------------------------------------------------

