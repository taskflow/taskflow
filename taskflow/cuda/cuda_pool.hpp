#pragma once

#include "cuda_error.hpp"

namespace tf {

/**
@brief per-thread object pool to manage CUDA context objects

@tparam H object type
@tparam C function object to create a library object
@tparam D function object to delete a library object

Creating a CUDA library object on a device is typically expensive (e.g., 200ms).
It is desirable to reuse the object as much as possible.
The manager class abstracts the creation and destruction of a CUDA library 
object on each device.

*/
template <typename H, typename C, typename D>
class cudaPerThreadContextObjectPool {
  
  public: 

  struct cudaContextObject {

    int device;
    H object;

    cudaContextObject(int);
    ~cudaContextObject();

    cudaContextObject(const cudaContextObject&) = delete;
    cudaContextObject(cudaContextObject&&) = delete;
  };

  private:
  
  // Master thread hold the storage to the pool.
  // Due to some ordering, cuda context may be destroyed when the master
  // program thread destroys the cuda object.
  // Therefore, we use a decentralized approach to let child thread
  // destroy cuda objects while the master thread only keeps a weak reference
  // to those objects for reuse.
  struct cudaGlobalContextObjectPool {
  
    std::shared_ptr<cudaContextObject> acquire(int);
    void release(int, std::weak_ptr<cudaContextObject>);
  
    std::mutex mutex;
    std::unordered_map<int, std::vector<std::weak_ptr<cudaContextObject>>> pool;
  };

  public:
    
    cudaPerThreadContextObjectPool() = default;
    
    std::shared_ptr<cudaContextObject> acquire(int);

    void release(std::shared_ptr<cudaContextObject>&&);

    size_t footprint_size() const;
    
  private: 

    inline static cudaGlobalContextObjectPool _cuda_object_pool;

    std::unordered_set<std::shared_ptr<cudaContextObject>> _footprint;
}; 

// ----------------------------------------------------------------------------
// cudaPerThreadContextObject::cudaHanalde definition
// ----------------------------------------------------------------------------

template <typename H, typename C, typename D>
cudaPerThreadContextObjectPool<H, C, D>::cudaContextObject::cudaContextObject(int d) : 
  device {d} {
  cudaScopedDevice ctx(device);
  object = C{}();
}

template <typename H, typename C, typename D>
cudaPerThreadContextObjectPool<H, C, D>::cudaContextObject::~cudaContextObject() {
  cudaScopedDevice ctx(device);
  D{}(object);
}

// ----------------------------------------------------------------------------
// cudaPerThreadContextObject::cudaHanaldePool definition
// ----------------------------------------------------------------------------

template <typename H, typename C, typename D>
std::shared_ptr<typename cudaPerThreadContextObjectPool<H, C, D>::cudaContextObject>
cudaPerThreadContextObjectPool<H, C, D>::cudaGlobalContextObjectPool::acquire(int d) {
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
void cudaPerThreadContextObjectPool<H, C, D>::cudaGlobalContextObjectPool::release(
  int d, std::weak_ptr<cudaContextObject> ptr
) {
  std::scoped_lock lock(mutex);
  pool[d].push_back(ptr);
}

// ----------------------------------------------------------------------------
// cudaPerThreadContextObject definition
// ----------------------------------------------------------------------------

template <typename H, typename C, typename D>
std::shared_ptr<typename cudaPerThreadContextObjectPool<H, C, D>::cudaContextObject> 
cudaPerThreadContextObjectPool<H, C, D>::acquire(int d) {

  auto ptr = _cuda_object_pool.acquire(d);

  if(!ptr) {
    ptr = std::make_shared<cudaContextObject>(d);
  }

  return ptr;
}

template <typename H, typename C, typename D>
void cudaPerThreadContextObjectPool<H, C, D>::release(
  std::shared_ptr<cudaContextObject>&& ptr) {
  _cuda_object_pool.release(ptr->device, ptr);
  _footprint.insert(std::move(ptr));
}

template <typename H, typename C, typename D>
size_t cudaPerThreadContextObjectPool<H, C, D>::footprint_size() const {
  return _footprint.size();
}

// ----------------------------------------------------------------------------

// Function object class to create a cuda stream
struct cudaStreamCreator {
  cudaStream_t operator () () const {
    cudaStream_t stream;
    TF_CHECK_CUDA(
      cudaStreamCreate(&stream), "failed to create a cuda stream"
    );
    return stream;
  }
};

// Function object class to delete a cuda stream
struct cudaStreamDeleter {
  void operator () (cudaStream_t ptr) const {
    cudaStreamDestroy(ptr);
  }
};


using cudaPerThreadStreamPool = cudaPerThreadContextObjectPool<
  cudaStream_t, cudaStreamCreator, cudaStreamDeleter
>;

/**
@brief per thread cudaStream pool
*/
inline thread_local cudaPerThreadStreamPool cuda_per_thread_stream_pool;


class cudaScopedPerThreadStream {
  
  public:

  explicit cudaScopedPerThreadStream(int d) : 
    _ptr {cuda_per_thread_stream_pool.acquire(d)} {
  }
  
  cudaScopedPerThreadStream() : 
    _ptr {cuda_per_thread_stream_pool.acquire(0)} {
  }

  ~cudaScopedPerThreadStream() {
    cuda_per_thread_stream_pool.release(std::move(_ptr));
  }

  operator cudaStream_t () const {
    return _ptr->object;
  }

  private:

  std::shared_ptr<cudaPerThreadStreamPool::cudaContextObject> _ptr;

};

}  // end of namespace tf -----------------------------------------------------



