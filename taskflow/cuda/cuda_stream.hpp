#pragma once

#include <memory>

#include "cuda_error.hpp"

/**
@file cuda_stream.hpp
@brief CUDA stream utilities include file
*/

namespace tf {


// ----------------------------------------------------------------------------
// cudaEvent
// ----------------------------------------------------------------------------

/**
@class cudaEvent

@brief class to create a CUDA event with unique ownership

The `cudaEvent` class encapsulates a `cudaEvent_t` using `std::unique_ptr`, ensuring that
CUDA events are properly created and destroyed with a unique ownership.
The event is created by `cudaEventCreate` (or `cudaEventCreateWithFlags`) and
destroyed by `cudaEventDestroy`.
*/
class cudaEvent : public std::unique_ptr<
  std::remove_pointer_t<cudaEvent_t>,
  cudaProxyDeleter<cudaEventDestroy>
> {

  public:

  /**
  @brief constructs a new event using `cudaEventCreate`
  */
  cudaEvent() : unique_ptr([](){
    cudaEvent_t event;
    TF_CHECK_CUDA(cudaEventCreate(&event), "failed to create a CUDA event");
    return event;
  }()) {
  }

  /**
  @brief constructs a new event using `cudaEventCreateWithFlags` with the given `flags`
  */
  explicit cudaEvent(unsigned int flags) : unique_ptr([flags](){
    cudaEvent_t event;
    TF_CHECK_CUDA(
      cudaEventCreateWithFlags(&event, flags),
      "failed to create a CUDA event with flags=", flags
    );
    return event;
  }()) {
  }

  /**
  @brief constructs an event that takes ownership of the given `cudaEvent_t`
  */
  explicit cudaEvent(cudaEvent_t event) : unique_ptr(event) {
  }

  /**
  @brief implicitly converts to the managed `cudaEvent_t`
  */
  operator cudaEvent_t () const noexcept { return get(); }
};

// ----------------------------------------------------------------------------
// cudaStream
// ----------------------------------------------------------------------------

/**
@class cudaStream

@brief class to create a CUDA stream with unique ownership

The `cudaStream` class encapsulates a `cudaStream_t` using `std::unique_ptr`, ensuring that
CUDA streams are properly created and destroyed with a unique ownership.
The stream is created by `cudaStreamCreate` and destroyed by `cudaStreamDestroy`.
*/
class cudaStream : public std::unique_ptr<
  std::remove_pointer_t<cudaStream_t>,
  cudaProxyDeleter<cudaStreamDestroy>
> {

  public:

  /**
  @brief constructs a new stream using `cudaStreamCreate`
  */
  cudaStream() : unique_ptr([](){
    cudaStream_t stream;
    TF_CHECK_CUDA(cudaStreamCreate(&stream), "failed to create a CUDA stream");
    return stream;
  }()) {
  }

  /**
  @brief constructs a stream that takes ownership of the given `cudaStream_t`
  */
  explicit cudaStream(cudaStream_t stream) : unique_ptr(stream) {
  }

  /**
  @brief implicitly converts to the managed `cudaStream_t`
  */
  operator cudaStream_t () const noexcept { return get(); }

  /**
  @brief synchronizes the associated stream

  Equivalently calling @c cudaStreamSynchronize to block 
  until this stream has completed all operations.
  */
  cudaStream& synchronize() {
    TF_CHECK_CUDA(
      cudaStreamSynchronize(this->get()), "failed to synchronize a CUDA stream"
    );
    return *this;
  }
  
  /**
  @brief begins graph capturing on the stream

  When a stream is in capture mode, all operations pushed into the stream 
  will not be executed, but will instead be captured into a graph, 
  which will be returned via cudaStream::end_capture. 

  A thread's mode can be one of the following:
  + @c cudaStreamCaptureModeGlobal: This is the default mode. 
    If the local thread has an ongoing capture sequence that was not initiated 
    with @c cudaStreamCaptureModeRelaxed at @c cuStreamBeginCapture, 
    or if any other thread has a concurrent capture sequence initiated with 
    @c cudaStreamCaptureModeGlobal, this thread is prohibited from potentially 
    unsafe API calls.

  + @c cudaStreamCaptureModeThreadLocal: If the local thread has an ongoing capture 
    sequence not initiated with @c cudaStreamCaptureModeRelaxed, 
    it is prohibited from potentially unsafe API calls. 
    Concurrent capture sequences in other threads are ignored.

  + @c cudaStreamCaptureModeRelaxed: The local thread is not prohibited 
    from potentially unsafe API calls. Note that the thread is still prohibited 
    from API calls which necessarily conflict with stream capture, for example, 
    attempting @c cudaEventQuery on an event that was last recorded 
    inside a capture sequence.
  */
  void begin_capture(cudaStreamCaptureMode m = cudaStreamCaptureModeGlobal) const {
    TF_CHECK_CUDA(
      cudaStreamBeginCapture(this->get(), m), 
      "failed to begin capture on stream ", this->get(), " with thread mode ", m
    );
  }

  /**
  @brief ends graph capturing on the stream
  
  Equivalently calling @c cudaStreamEndCapture to
  end capture on stream and returning the captured graph. 
  Capture must have been initiated on stream via a call to cudaStream::begin_capture. 
  If capture was invalidated, due to a violation of the rules of stream capture, 
  then a NULL graph will be returned.
  */
  cudaGraph_t end_capture() const {
    cudaGraph_t native_g;
    TF_CHECK_CUDA(
      cudaStreamEndCapture(this->get(), &native_g), 
      "failed to end capture on stream ", this->get()
    );
    return native_g;
  }
  
  /**
  @brief records an event on the stream

  Equivalently calling @c cudaEventRecord to record an event on this stream,
  both of which must be on the same CUDA context.
  */
  void record(cudaEvent_t event) const {
    TF_CHECK_CUDA(
      cudaEventRecord(event, this->get()), 
      "failed to record event ", event, " on stream ", this->get()
    );
  }

  /**
  @brief waits on an event

  Equivalently calling @c cudaStreamWaitEvent to make all future work 
  submitted to stream wait for all work captured in event.
  */
  void wait(cudaEvent_t event) const {
    TF_CHECK_CUDA(
      cudaStreamWaitEvent(this->get(), event, 0), 
      "failed to wait for event ", event, " on stream ", this->get()
    );
  }

  /**
  @brief runs the given executable CUDA graph using `cudaGraphLaunch`

  A tf::cudaGraphExec implicitly converts to `cudaGraphExec_t` and can be passed directly.

  @param exec the given `cudaGraphExec_t`
  */
  cudaStream& run(cudaGraphExec_t exec) {
    TF_CHECK_CUDA(
      cudaGraphLaunch(exec, this->get()), "failed to launch a CUDA executable graph"
    );
    return *this;
  }
};

}  // end of namespace tf -----------------------------------------------------



