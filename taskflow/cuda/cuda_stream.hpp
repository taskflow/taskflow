#pragma once

#include "cuda_pool.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// c++ wrapper over cudaStream functions
// ----------------------------------------------------------------------------

/**
@brief turns on capture for the given stream
*/
inline void start_stream_capture(cudaStream_t stream) {
  TF_CHECK_CUDA(
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), 
    "failed to turn stream into capture mode"
  );
}

/**
@brief turns off capture for the given stream and returns the captured graph
*/  
inline cudaGraph_t cease_stream_capture(cudaStream_t stream) {
  cudaGraph_t graph;
  TF_CHECK_CUDA(cudaStreamEndCapture(stream, &graph), "failed to end capture");
  return graph;
}

// ----------------------------------------------------------------------------
// cudaStreamCreator and cudaStreamDeleter for per-thread stream pool
// ----------------------------------------------------------------------------

/**
@brief function object class to create a CUDA stream
*/
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

/**
@brief function object class to destroy a cuda stream
*/
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
@brief per thread cuda stream pool
*/
inline thread_local cudaPerThreadStreamPool cuda_per_thread_stream_pool;

// ----------------------------------------------------------------------------
// cudaScopedPerThreadStream definition
// ----------------------------------------------------------------------------

/**
@brief class to provide RAII-styled guard of stream acquisition

Sample usage:
    
@code{.cpp}
{
  cudaScopedPerThreadStream stream(1);  // acquires a stream on device 1

  // use stream as a normal cuda stream (cudaStream_t)
  cudaStreamWaitEvent(stream, ...);

}  // leaving the scope to release the stream back to the pool on device 1
@endcode

*/
class cudaScopedPerThreadStream {
  
  public:
  
  /**
  @brief constructs a scoped stream under the given device

  The constructor acquires a stream from a per-thread stream pool.
  */
  explicit cudaScopedPerThreadStream(int device) : 
    _ptr {cuda_per_thread_stream_pool.acquire(device)} {
  }
  
  /**
  @brief constructs a scoped stream under the current device.

  The constructor acquires a stream from a per-thread stream pool.
  */
  cudaScopedPerThreadStream() : 
    _ptr {cuda_per_thread_stream_pool.acquire(cuda_get_device())} {
  }

  /**
  @brief destructs the scoped stream guard

  The destructor releases the stream to the per-thread stream pool.
  */
  ~cudaScopedPerThreadStream() {
    cuda_per_thread_stream_pool.release(std::move(_ptr));
  }
  
  /**
  @brief implicit conversion to the native cuda stream (cudaStream_t)
   */
  operator cudaStream_t () const {
    return _ptr->object;
  }

  private:

  std::shared_ptr<cudaPerThreadStreamPool::cudaDeviceObject> _ptr;

};

}  // end of namespace tf -----------------------------------------------------



