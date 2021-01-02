#pragma once

#include "cuda_pool.hpp"

/**
@file cuda_stream.hpp
@brief CUDA stream utilities include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// cudaStreamCreator and cudaStreamDeleter for per-thread stream pool
// ----------------------------------------------------------------------------

/** @private */
struct cudaStreamCreator {

  /**
  @brief operator to create a CUDA stream
   */
  cudaStream_t operator () () const {
    cudaStream_t stream;
    TF_CHECK_CUDA(cudaStreamCreate(&stream), "failed to create a CUDA stream");
    return stream;
  }
};

/** @private */
struct cudaStreamDeleter {

  /**
  @brief operator to destroy a CUDA stream
  */
  void operator () (cudaStream_t stream) const {
    cudaStreamDestroy(stream);
  }
};

/**
@brief alias of per-thread stream pool type
 */
using cudaPerThreadStreamPool = cudaPerThreadDeviceObjectPool<
  cudaStream_t, cudaStreamCreator, cudaStreamDeleter
>;

/**
@brief acquires the per-thread cuda stream pool
*/
inline cudaPerThreadStreamPool& cuda_per_thread_stream_pool() {
  thread_local cudaPerThreadStreamPool pool;
  return pool;
}

// ----------------------------------------------------------------------------
// cudaScopedPerThreadStream definition
// ----------------------------------------------------------------------------

/**
@brief class that provides RAII-styled guard of stream acquisition

Sample usage:
    
@code{.cpp}
{
  tf::cudaScopedPerThreadStream stream(1);  // acquires a stream on device 1

  // use stream as a normal cuda stream (cudaStream_t)
  cudaStreamWaitEvent(stream, ...);

}  // leaving the scope releases the stream back to the pool on device 1
@endcode

The scoped per-thread stream is primarily used by tf::Executor to execute
CUDA tasks (e.g., tf::cudaFlow, tf::cudaFlowCapturer).

%cudaScopedPerThreadStream is non-copyable.
*/
class cudaScopedPerThreadStream {
  
  public:
  
  /**
  @brief constructs a scoped stream under the given device

  The constructor acquires a stream from a per-thread stream pool.

  @param device device context of the requested stream
  */
  explicit cudaScopedPerThreadStream(int device) : 
    _ptr {cuda_per_thread_stream_pool().acquire(device)} {
  }
  
  /**
  @brief constructs a scoped stream under the current device.

  The constructor acquires a stream from a per-thread stream pool.
  */
  cudaScopedPerThreadStream() : 
    _ptr {cuda_per_thread_stream_pool().acquire(cuda_get_device())} {
  }

  /**
  @brief destructs the scoped stream guard

  The destructor releases the stream to the per-thread stream pool.
  */
  ~cudaScopedPerThreadStream() {
    if(_ptr) {
      cuda_per_thread_stream_pool().release(std::move(_ptr));
    }
  }
  
  /**
  @brief implicit conversion to the native CUDA stream (cudaStream_t)
   */
  operator cudaStream_t () const {
    return _ptr->value;
  }
  
  /**
  @brief disabled copy constructor
   */
  cudaScopedPerThreadStream(const cudaScopedPerThreadStream&) = delete;
  
  /**
  @brief default move constructor
  */
  cudaScopedPerThreadStream(cudaScopedPerThreadStream&&) = default;

  /**
  @brief disabled copy assignment
  */
  cudaScopedPerThreadStream& operator = (const cudaScopedPerThreadStream&) = delete;

  /**
  @brief default move assignment
  */
  cudaScopedPerThreadStream& operator = (cudaScopedPerThreadStream&&) = delete;

  private:

  std::shared_ptr<cudaPerThreadStreamPool::Object> _ptr;

};

// ----------------------------------------------------------------------------
// cudaStreamCreator and cudaStreamDeleter for per-thread event pool
// ----------------------------------------------------------------------------

/** @private */
struct cudaEventCreator {

  /**
  @brief operator to create a CUDA event
   */
  cudaEvent_t operator () () const {
    cudaEvent_t event;
    TF_CHECK_CUDA(cudaEventCreate(&event), "failed to create a CUDA event");
    return event;
  }
};

/** @private */
struct cudaEventDeleter {

  /**
  @brief operator to destroy a CUDA event
  */
  void operator () (cudaEvent_t event) const {
    cudaEventDestroy(event);
  }
};

/**
@brief alias of per-thread event pool type
 */
using cudaPerThreadEventPool = cudaPerThreadDeviceObjectPool<
  cudaEvent_t, cudaEventCreator, cudaEventDeleter
>;

/**
@brief per-thread cuda event pool
*/
inline cudaPerThreadEventPool& cuda_per_thread_event_pool() {
  thread_local cudaPerThreadEventPool pool;
  return pool;
}

// ----------------------------------------------------------------------------
// cudaScopedPerThreadEvent definition
// ----------------------------------------------------------------------------

/**
@brief class that provides RAII-styled guard of event acquisition

Sample usage:
    
@code{.cpp}
{
  tf::cudaScopedPerThreadEvent event(1);  // acquires a event on device 1

  // use event as a normal cuda event (cudaEvent_t)
  cudaStreamWaitEvent(stream, event);

}  // leaving the scope releases the event back to the pool on device 1
@endcode

The scoped per-thread event is primarily used by tf::Executor to execute
CUDA tasks (e.g., tf::cudaFlow, tf::cudaFlowCapturer).

%cudaScopedPerThreadEvent is non-copyable.
*/
class cudaScopedPerThreadEvent {
  
  public:
  
  /**
  @brief constructs a scoped event under the given device

  The constructor acquires a event from a per-thread event pool.

  @param device device context of the requested event
  */
  explicit cudaScopedPerThreadEvent(int device) : 
    _ptr {cuda_per_thread_event_pool().acquire(device)} {
  }
  
  /**
  @brief constructs a scoped event under the current device.

  The constructor acquires a event from a per-thread event pool.
  */
  cudaScopedPerThreadEvent() : 
    _ptr {cuda_per_thread_event_pool().acquire(cuda_get_device())} {
  }

  /**
  @brief destructs the scoped event guard

  The destructor releases the event to the per-thread event pool.
  */
  ~cudaScopedPerThreadEvent() {
    if(_ptr) {
      cuda_per_thread_event_pool().release(std::move(_ptr));
    }
  }
  
  /**
  @brief implicit conversion to the native CUDA event (cudaEvent_t)
   */
  operator cudaEvent_t () const {
    return _ptr->value;
  }
  
  /**
  @brief disabled copy constructor
   */
  cudaScopedPerThreadEvent(const cudaScopedPerThreadEvent&) = delete;
  
  /**
  @brief default move constructor
  */
  cudaScopedPerThreadEvent(cudaScopedPerThreadEvent&&) = default;

  /**
  @brief disabled copy assignment
  */
  cudaScopedPerThreadEvent& operator = (const cudaScopedPerThreadEvent&) = delete;

  /**
  @brief default move assignment
  */
  cudaScopedPerThreadEvent& operator = (cudaScopedPerThreadEvent&&) = delete;

  private:

  std::shared_ptr<cudaPerThreadEventPool::Object> _ptr;

};


}  // end of namespace tf -----------------------------------------------------



