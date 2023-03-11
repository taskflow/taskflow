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
The global pool does not own the object and therefore does not destruct any of them.
The per-thread pool keeps the footprints of objects with shared ownership
and will destruct them if the thread holds the last reference count after it joins.
The motivation of this decentralized control is to avoid device objects
from being destroyed while the context had been destroyed due to driver shutdown.

*/
template <typename H, typename C, typename D>
class cudaPerThreadDeviceObjectPool {

  public:

  /**
  @brief structure to store a context object
   */
  struct Object {

    int device;
    H value;

    Object(int);
    ~Object();

    Object(const Object&) = delete;
    Object(Object&&) = delete;
  };

  private:

  // Master thread hold the storage to the pool.
  // Due to some ordering, cuda context may be destroyed when the master
  // program thread destroys the cuda object.
  // Therefore, we use a decentralized approach to let child thread
  // destroy cuda objects while the master thread only keeps a weak reference
  // to those objects for reuse.
  struct cudaGlobalDeviceObjectPool {

    std::shared_ptr<Object> acquire(int);
    void release(int, std::weak_ptr<Object>);

    std::mutex mutex;
    std::unordered_map<int, std::vector<std::weak_ptr<Object>>> pool;
  };

  public:

    /**
    @brief default constructor
     */
    cudaPerThreadDeviceObjectPool() = default;

    /**
    @brief acquires a device object with shared ownership
     */
    std::shared_ptr<Object> acquire(int);

    /**
    @brief releases a device object with moved ownership
    */
    void release(std::shared_ptr<Object>&&);

    /**
    @brief queries the number of device objects with shared ownership
     */
    size_t footprint_size() const;

  private:

    inline static cudaGlobalDeviceObjectPool _shared_pool;

    std::unordered_set<std::shared_ptr<Object>> _footprint;
};

// ----------------------------------------------------------------------------
// cudaPerThreadDeviceObject::cudaHanale definition
// ----------------------------------------------------------------------------

template <typename H, typename C, typename D>
cudaPerThreadDeviceObjectPool<H, C, D>::Object::Object(int d) :
  device {d} {
  cudaScopedDevice ctx(device);
  value = C{}();
}

template <typename H, typename C, typename D>
cudaPerThreadDeviceObjectPool<H, C, D>::Object::~Object() {
  cudaScopedDevice ctx(device);
  D{}(value);
}

// ----------------------------------------------------------------------------
// cudaPerThreadDeviceObject::cudaHanaldePool definition
// ----------------------------------------------------------------------------

template <typename H, typename C, typename D>
std::shared_ptr<typename cudaPerThreadDeviceObjectPool<H, C, D>::Object>
cudaPerThreadDeviceObjectPool<H, C, D>::cudaGlobalDeviceObjectPool::acquire(int d) {
  std::scoped_lock<std::mutex> lock(mutex);
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
  int d, std::weak_ptr<Object> ptr
) {
  std::scoped_lock<std::mutex> lock(mutex);
  pool[d].push_back(ptr);
}

// ----------------------------------------------------------------------------
// cudaPerThreadDeviceObject definition
// ----------------------------------------------------------------------------

template <typename H, typename C, typename D>
std::shared_ptr<typename cudaPerThreadDeviceObjectPool<H, C, D>::Object>
cudaPerThreadDeviceObjectPool<H, C, D>::acquire(int d) {

  auto ptr = _shared_pool.acquire(d);

  if(!ptr) {
    ptr = std::make_shared<Object>(d);
  }

  return ptr;
}

template <typename H, typename C, typename D>
void cudaPerThreadDeviceObjectPool<H, C, D>::release(
  std::shared_ptr<Object>&& ptr
) {
  _shared_pool.release(ptr->device, ptr);
  _footprint.insert(std::move(ptr));
}

template <typename H, typename C, typename D>
size_t cudaPerThreadDeviceObjectPool<H, C, D>::footprint_size() const {
  return _footprint.size();
}

// ----------------------------------------------------------------------------
// cudaObject
// ----------------------------------------------------------------------------

/**
@class cudaObject

@brief class to create an RAII-styled and move-only wrapper for CUDA objects
*/
template <typename T, typename C, typename D>
class cudaObject {
  
  public:

  /**
  @brief constructs a CUDA object from the given one
  */
  explicit cudaObject(T obj) : object(obj) {}
  
  /**
  @brief constructs a new CUDA object
  */
  cudaObject() : object{ C{}() } {}
    
  /**
  @brief disabled copy constructor
  */
  cudaObject(const cudaObject&) = delete;
  
  /**
  @brief move constructor
  */
  cudaObject(cudaObject&& rhs) : object{rhs.object} {
    rhs.object = nullptr;
  }

  /**
  @brief destructs the CUDA object
  */
  ~cudaObject() { D{}(object); }
  
  /**
  @brief disabled copy assignment
  */
  cudaObject& operator = (const cudaObject&) = delete;

  /**
  @brief move assignment
  */
  cudaObject& operator = (cudaObject&& rhs) {
    D {} (object);
    object = rhs.object;
    rhs.object = nullptr;
    return *this;
  }
  
  /**
  @brief implicit conversion to the native CUDA stream (cudaObject_t)

  Returns the underlying stream of type @c cudaObject_t.
  */
  operator T () const {
    return object;
  }
    
  /**
  @brief deletes the current CUDA object (if any) and creates a new one
  */
  void create() {
    D {} (object);
    object = C{}();
  }
  
  /**
  @brief resets this CUDA object to the given one
  */
  void reset(T new_obj) {
    D {} (object);
    object = new_obj;
  }
  
  /**
  @brief deletes the current CUDA object
  */
  void clear() {
    reset(nullptr);
  }

  /**
  @brief releases the ownership of the CUDA object
  */
  T release() {
    auto tmp = object;
    object = nullptr;
    return tmp;
  }
  
  protected:

  /**
  @brief the CUDA object
  */
  T object;
};

}  // end of namespace tf -----------------------------------------------------



