#pragma once

#include "cuda_error.hpp"

/**
@file cuda_stream.hpp
@brief CUDA stream utilities include file
*/

namespace tf {


// ----------------------------------------------------------------------------
// cudaEventBase
// ----------------------------------------------------------------------------
  
/**
@struct cudaEventCreator

@brief functor to create a `cudaEvent_t` object
*/
struct cudaEventCreator {
    
  /**
  @brief creates a new `cudaEvent_t` object using `cudaEventCreate`
  */
  cudaEvent_t operator () () const {
    cudaEvent_t event;
    TF_CHECK_CUDA(cudaEventCreate(&event), "failed to create a CUDA event");
    return event;
  }
  
  /**
  @brief creates a new `cudaEvent_t` object using `cudaEventCreate` with the given `flag`
  */
  cudaEvent_t operator () (unsigned int flag) const {
    cudaEvent_t event;
    TF_CHECK_CUDA(
      cudaEventCreateWithFlags(&event, flag),
      "failed to create a CUDA event with flag=", flag
    );
    return event;
  }
  
  /**
  @brief returns the given `cudaEvent_t` object
  */
  cudaEvent_t operator () (cudaEvent_t event) const {
    return event;
  }
};

/**
@struct cudaEventDeleter

@brief functor to delete a `cudaEvent_t` object
*/
struct cudaEventDeleter {

  /**
  @brief deletes the given `cudaEvent_t` object using `cudaEventDestroy`
  */
  void operator () (cudaEvent_t event) const {
    cudaEventDestroy(event);
  }
};

/**
@class cudaEventBase

@brief class to create a smart pointer wrapper for managing `cudaEvent_t`

@tparam Creator functor to create the stream (used in constructor)
@tparam Deleter functor to delete the stream (used in destructor)

The `cudaEventBase` class encapsulates a `cudaEvent_t` using `std::unique_ptr`, ensuring that
CUDA events are properly created and destroyed with a unique ownership.
*/
template <typename Creator, typename Deleter>
class cudaEventBase : public std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, Deleter> {

  static_assert(std::is_pointer_v<cudaEvent_t>, "cudaEvent_t is not a pointer type");

  public:
  
  /**
  @brief base type for the underlying unique pointer

  This alias provides a shorthand for the underlying `std::unique_ptr` type that manages
  CUDA event resources with an associated deleter.
  */
  using base_type = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, Deleter>;

  /**
  @brief constructs a `cudaEvent` object by passing the given arguments to the event creator

  Constructs a `cudaEvent` object by passing the given arguments to the event creator

  @param args arguments to pass to the event creator
  */
  template <typename... ArgsT>
  explicit cudaEventBase(ArgsT&& ... args) : base_type(
    Creator{}(std::forward<ArgsT>(args)...), Deleter()
  ) {
  }  
  
  /**
  @brief constructs a `cudaEvent` from the given rhs using move semantics
  */
  cudaEventBase(cudaEventBase&&) = default;

  /**
  @brief assign the rhs to `*this` using move semantics
  */
  cudaEventBase& operator = (cudaEventBase&&) = default;
  
  private:

  cudaEventBase(const cudaEventBase&) = delete;
  cudaEventBase& operator = (const cudaEventBase&) = delete;
};

/**
@brief default smart pointer type to manage a `cudaEvent_t` object with unique ownership
*/
using cudaEvent = cudaEventBase<cudaEventCreator, cudaEventDeleter>;

// ----------------------------------------------------------------------------
// cudaStream
// ----------------------------------------------------------------------------

/**
@struct cudaStreamCreator 

@brief functor to create a `cudaStream_t` object
*/
struct cudaStreamCreator {
  
  /**
  @brief constructs a new `cudaStream_t` object using `cudaStreamCreate`
  */
  cudaStream_t operator () () const {
    cudaStream_t stream;
    TF_CHECK_CUDA(cudaStreamCreate(&stream), "failed to create a CUDA stream");
    return stream;
  }
  
  /**
  @brief returns the given `cudaStream_t` object
  */
  cudaStream_t operator () (cudaStream_t stream) const {
    return stream;
  }
};

/**
@struct cudaStreamDeleter

@brief functor to delete a `cudaStream_t` object
*/
struct cudaStreamDeleter {

  /**
  @brief deletes the given `cudaStream_t` object
  */
  void operator () (cudaStream_t stream) const {
    cudaStreamDestroy(stream);
  }
};

/**
@class cudaStreamBase

@brief class to create a smart pointer wrapper for managing `cudaStream_t`

@tparam Creator functor to create the stream (used in constructor)
@tparam Deleter functor to delete the stream (used in destructor)

The `cudaStream` class encapsulates a `cudaStream_t` using `std::unique_ptr`, ensuring that
CUDA events are properly created and destroyed with a unique ownership.
*/
template <typename Creator, typename Deleter>
class cudaStreamBase : public std::unique_ptr<std::remove_pointer_t<cudaStream_t>, Deleter> {

  static_assert(std::is_pointer_v<cudaStream_t>, "cudaStream_t is not a pointer type");
  
  public:
  
  /**
  @brief base type for the underlying unique pointer

  This alias provides a shorthand for the underlying `std::unique_ptr` type that manages
  CUDA stream resources with an associated deleter.
  */
  using base_type = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, Deleter>;

  /**
  @brief constructs a `cudaStream` object by passing the given arguments to the stream creator

  Constructs a `cudaStream` object by passing the given arguments to the stream creator

  @param args arguments to pass to the stream creator
  */
  template <typename... ArgsT>
  explicit cudaStreamBase(ArgsT&& ... args) : base_type(
    Creator{}(std::forward<ArgsT>(args)...), Deleter()
  ) {
  }  
  
  /**
  @brief constructs a `cudaStream` from the given rhs using move semantics
  */
  cudaStreamBase(cudaStreamBase&&) = default;

  /**
  @brief assign the rhs to `*this` using move semantics
  */
  cudaStreamBase& operator = (cudaStreamBase&&) = default;
  
  /**
  @brief synchronizes the associated stream

  Equivalently calling @c cudaStreamSynchronize to block 
  until this stream has completed all operations.
  */
  cudaStreamBase& synchronize() {
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
  @brief runs the given executable CUDA graph

  @param exec the given `cudaGraphExec`
  */
  template <typename C, typename D>
  cudaStreamBase& run(const cudaGraphExecBase<C, D>& exec);

  /**
  @brief runs the given executable CUDA graph
  
  @param exec the given `cudaGraphExec_t`
  */
  cudaStreamBase& run(cudaGraphExec_t exec);

  private:

  cudaStreamBase(const cudaStreamBase&) = delete;
  cudaStreamBase& operator = (const cudaStreamBase&) = delete;
};

/**
@brief default smart pointer type to manage a `cudaStream_t` object with unique ownership
*/
using cudaStream = cudaStreamBase<cudaStreamCreator, cudaStreamDeleter>;

}  // end of namespace tf -----------------------------------------------------



