#pragma once

#include "cuda_error.hpp"

namespace tf {

/**
@brief per-thread object pool to manage CUDA device object

@tparam H object type
@tparam C function object to create a library object
@tparam D function object to delete a library object

A CUDA device object has a lifetime associated with a device,
for example, @c cudaStream_t, @c cublasHandle_t, etc.
Creating a device object is typically expensive (e.g., 10-200 ms)
and destroying it may trigger implicit device synchronization.
For applications tha intensively make use of device objects,
it is desirable to reuse them as much as possible.

There exists an one-to-one relationship between CUDA devices in CUDA Runtime API
and CUcontexts in the CUDA Driver API within a process.
The specific context which the CUDA Runtime API uses for a device 
is called the device's primary context. 
From the perspective of the CUDA Runtime API, 
a device and its primary context are synonymous.

We design the device object pool in a decentralized fashion by keeping
(1) a global pool to keep track of potentially usable objects and
(2) a per-thread pool to footprint objects with shared ownership.
The global pool does not own the object and therefore do not destruct any of them.
The per-thread pool owns the object with shared ownership and will perform
destruction depending on which thread holds the last reference count.
The motivation of this decentralized control is to avoid device objects
from being destroyed while the context had been destroyed due to driver shutdown.

*/
template <typename H, typename C, typename D>
class cudaPerThreadDeviceObjectPool {
  
  public: 
  
  /**
  @brief structure to store a context object 
   */
  struct cudaDeviceObject {

    int device;
    H object;

    cudaDeviceObject(int);
    ~cudaDeviceObject();

    cudaDeviceObject(const cudaDeviceObject&) = delete;
    cudaDeviceObject(cudaDeviceObject&&) = delete;
  };

  private:
  
  // Master thread hold the storage to the pool.
  // Due to some ordering, cuda context may be destroyed when the master
  // program thread destroys the cuda object.
  // Therefore, we use a decentralized approach to let child thread
  // destroy cuda objects while the master thread only keeps a weak reference
  // to those objects for reuse.
  struct cudaGlobalDeviceObjectPool {
  
    std::shared_ptr<cudaDeviceObject> acquire(int);
    void release(int, std::weak_ptr<cudaDeviceObject>);
  
    std::mutex mutex;
    std::unordered_map<int, std::vector<std::weak_ptr<cudaDeviceObject>>> pool;
  };

  public:
    
    /**
    @brief default constructor
     */ 
    cudaPerThreadDeviceObjectPool() = default;
    
    /**
    @brief acquires a device object with shared ownership
     */
    std::shared_ptr<cudaDeviceObject> acquire(int);
    
    /**
    @brief releases a device object with moved ownership
    */
    void release(std::shared_ptr<cudaDeviceObject>&&);
    
    /**
    @brief queries the number of device objects with shared ownership
     */
    size_t footprint_size() const;
    
  private: 

    inline static cudaGlobalDeviceObjectPool _cuda_object_pool;

    std::unordered_set<std::shared_ptr<cudaDeviceObject>> _footprint;
}; 

// ----------------------------------------------------------------------------
// cudaPerThreadDeviceObject::cudaHanalde definition
// ----------------------------------------------------------------------------

template <typename H, typename C, typename D>
cudaPerThreadDeviceObjectPool<H, C, D>::cudaDeviceObject::cudaDeviceObject(int d) : 
  device {d} {
  cudaScopedDevice ctx(device);
  object = C{}();
}

template <typename H, typename C, typename D>
cudaPerThreadDeviceObjectPool<H, C, D>::cudaDeviceObject::~cudaDeviceObject() {
  cudaScopedDevice ctx(device);
  D{}(object);
}

// ----------------------------------------------------------------------------
// cudaPerThreadDeviceObject::cudaHanaldePool definition
// ----------------------------------------------------------------------------

template <typename H, typename C, typename D>
std::shared_ptr<typename cudaPerThreadDeviceObjectPool<H, C, D>::cudaDeviceObject>
cudaPerThreadDeviceObjectPool<H, C, D>::cudaGlobalDeviceObjectPool::acquire(int d) {
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
void cudaPerThreadDeviceObjectPool<H, C, D>::cudaGlobalDeviceObjectPool::release(
  int d, std::weak_ptr<cudaDeviceObject> ptr
) {
  std::scoped_lock lock(mutex);
  pool[d].push_back(ptr);
}

// ----------------------------------------------------------------------------
// cudaPerThreadDeviceObject definition
// ----------------------------------------------------------------------------

template <typename H, typename C, typename D>
std::shared_ptr<typename cudaPerThreadDeviceObjectPool<H, C, D>::cudaDeviceObject> 
cudaPerThreadDeviceObjectPool<H, C, D>::acquire(int d) {

  auto ptr = _cuda_object_pool.acquire(d);

  if(!ptr) {
    ptr = std::make_shared<cudaDeviceObject>(d);
  }

  return ptr;
}

template <typename H, typename C, typename D>
void cudaPerThreadDeviceObjectPool<H, C, D>::release(
  std::shared_ptr<cudaDeviceObject>&& ptr
) {
  _cuda_object_pool.release(ptr->device, ptr);
  _footprint.insert(std::move(ptr));
}

template <typename H, typename C, typename D>
size_t cudaPerThreadDeviceObjectPool<H, C, D>::footprint_size() const {
  return _footprint.size();
}

}  // end of namespace tf -----------------------------------------------------



