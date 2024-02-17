#pragma once

#include "cuda_object.hpp"

/**
@file cuda_stream.hpp
@brief CUDA stream utilities include file
*/

namespace tf {



// ----------------------------------------------------------------------------
// cudaStream
// ----------------------------------------------------------------------------

/**
@private
*/
struct cudaStreamCreator {
  cudaStream_t operator () () const {
    cudaStream_t stream;
    TF_CHECK_CUDA(cudaStreamCreate(&stream), "failed to create a CUDA stream");
    return stream;
  }
};

/**
@private
*/
struct cudaStreamDeleter {
  void operator () (cudaStream_t stream) const {
    if(stream) {
      cudaStreamDestroy(stream);
    }
  }
};

/**
@class cudaStream

@brief class to create an RAII-styled wrapper over a native CUDA stream

A cudaStream object is an RAII-styled wrapper over a native CUDA stream
(@c cudaStream_t).
A cudaStream object is move-only.
*/
class cudaStream : 

  public cudaObject <cudaStream_t, cudaStreamCreator, cudaStreamDeleter> {
  
  public:

    /**
    @brief constructs an RAII-styled object from the given CUDA stream

    Constructs a cudaStream object which owns @c stream.
    */
    explicit cudaStream(cudaStream_t stream) : cudaObject(stream) {
    }
    
    /**
    @brief default constructor
    */
    cudaStream() = default;
    
    /**
    @brief synchronizes the associated stream

    Equivalently calling @c cudaStreamSynchronize to block 
    until this stream has completed all operations.
    */
    void synchronize() const {
      TF_CHECK_CUDA(
        cudaStreamSynchronize(object), "failed to synchronize a CUDA stream"
      );
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
        cudaStreamBeginCapture(object, m), 
        "failed to begin capture on stream ", object, " with thread mode ", m
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
        cudaStreamEndCapture(object, &native_g), 
        "failed to end capture on stream ", object
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
        cudaEventRecord(event, object), 
        "failed to record event ", event, " on stream ", object
      );
    }
    
    /**
    @brief waits on an event

    Equivalently calling @c cudaStreamWaitEvent to make all future work 
    submitted to stream wait for all work captured in event.
    */
    void wait(cudaEvent_t event) const {
      TF_CHECK_CUDA(
        cudaStreamWaitEvent(object, event, 0), 
        "failed to wait for event ", event, " on stream ", object
      );
    }
};

// ----------------------------------------------------------------------------
// cudaEvent
// ----------------------------------------------------------------------------
  
/**
@private
*/
struct cudaEventCreator {

  cudaEvent_t operator () () const {
    cudaEvent_t event;
    TF_CHECK_CUDA(cudaEventCreate(&event), "failed to create a CUDA event");
    return event;
  }
  
  cudaEvent_t operator () (unsigned int flag) const {
    cudaEvent_t event;
    TF_CHECK_CUDA(
      cudaEventCreateWithFlags(&event, flag),
      "failed to create a CUDA event with flag=", flag
    );
    return event;
  }
};

/**
@private
*/
struct cudaEventDeleter {
  void operator () (cudaEvent_t event) const {
    if (event != nullptr) {
      cudaEventDestroy(event);
    }
  }
};

/**
@class cudaEvent

@brief class to create an RAII-styled wrapper over a native CUDA event

A cudaEvent object is an RAII-styled wrapper over a native CUDA event 
(@c cudaEvent_t).
A cudaEvent object is move-only.
*/
class cudaEvent :
  public cudaObject<cudaEvent_t, cudaEventCreator, cudaEventDeleter> {

  public:

    /**
    @brief constructs an RAII-styled CUDA event object from the given CUDA event
    */
    explicit cudaEvent(cudaEvent_t event) : cudaObject(event) { }   

    /**
    @brief constructs an RAII-styled CUDA event object
    */
    cudaEvent() = default;
    
    /**
    @brief constructs an RAII-styled CUDA event object with the given flag
    */
    explicit cudaEvent(unsigned int flag) : cudaObject(cudaEventCreator{}(flag)) { }
};


}  // end of namespace tf -----------------------------------------------------



