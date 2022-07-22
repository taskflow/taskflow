#pragma once

#include "cuda_pool.hpp"

/**
@file cuda_stream.hpp
@brief CUDA stream utilities include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// cudaStream
// ----------------------------------------------------------------------------

/**
@class cudaStream

@brief class to create an RAII-styled wrapper over a native CUDA stream

A cudaStream object is an RAII-styled wrapper over a native CUDA stream
(@c cudaStream_t).
A cudaStream object is move-only.
*/
class cudaStream {

  struct cudaStreamCreator {
    cudaStream_t operator () () const {
      cudaStream_t stream;
      TF_CHECK_CUDA(cudaStreamCreate(&stream), "failed to create a CUDA stream");
      return stream;
    }
  };
  
  struct cudaStreamDeleter {
    void operator () (cudaStream_t stream) const {
      if(stream) {
        cudaStreamDestroy(stream);
      }
    }
  };
  
  public:

    /**
    @brief constructs an RAII-styled object from the given CUDA stream

    Constructs a cudaStream object which owns @c stream.
    */
    explicit cudaStream(cudaStream_t stream) : _stream(stream) {
    }
    
    /**
    @brief constructs an RAII-styled object for a new CUDA stream

    Equivalently calling @c cudaStreamCreate to create a stream.
    */
    cudaStream() : _stream{ cudaStreamCreator{}() } {
    }
    
    /**
    @brief disabled copy constructor
    */
    cudaStream(const cudaStream&) = delete;
    
    /**
    @brief move constructor
    */
    cudaStream(cudaStream&& rhs) : _stream{rhs._stream} {
      rhs._stream = nullptr;
    }

    /**
    @brief destructs the CUDA stream
    */
    ~cudaStream() {
      cudaStreamDeleter {} (_stream);
    }
    
    /**
    @brief disabled copy assignment
    */
    cudaStream& operator = (const cudaStream&) = delete;

    /**
    @brief move assignment
    */
    cudaStream& operator = (cudaStream&& rhs) {
      cudaStreamDeleter {} (_stream);
      _stream = rhs._stream;
      rhs._stream = nullptr;
      return *this;
    }
    
    /**
    @brief implicit conversion to the native CUDA stream (cudaStream_t)

    Returns the underlying stream of type @c cudaStream_t.
    */
    operator cudaStream_t () const {
      return _stream;
    }
    
    /**
    @brief synchronizes the associated stream

    Equivalently calling @c cudaStreamSynchronize to block 
    until this stream has completed all operations.
    */
    void synchronize() const {
      TF_CHECK_CUDA(
        cudaStreamSynchronize(_stream), "failed to synchronize a CUDA stream"
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
        cudaStreamBeginCapture(_stream, m), 
        "failed to begin capture on stream ", _stream, " with thread mode ", m
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
        cudaStreamEndCapture(_stream, &native_g), 
        "failed to end capture on stream ", _stream
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
        cudaEventRecord(event, _stream), 
        "failed to record event ", event, " on stream ", _stream
      );
    }
    
    /**
    @brief waits on an event

    Equivalently calling @c cudaStreamWaitEvent to make all future work 
    submitted to stream wait for all work captured in event.
    */
    void wait(cudaEvent_t event) const {
      TF_CHECK_CUDA(
        cudaStreamWaitEvent(_stream, event, 0), 
        "failed to wait for event ", event, " on stream ", _stream
      );
    }

  private:

    cudaStream_t _stream {nullptr};
};

// ----------------------------------------------------------------------------
// cudaEvent
// ----------------------------------------------------------------------------

/**
@class cudaEvent

@brief class to create an RAII-styled wrapper over a native CUDA event

A cudaEvent object is an RAII-styled wrapper over a native CUDA event 
(@c cudaEvent_t).
A cudaEvent object is move-only.
*/
class cudaEvent {

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
  
  struct cudaEventDeleter {
    void operator () (cudaEvent_t event) const {
      cudaEventDestroy(event);
    }
  };

  public:

    /**
    @brief constructs an RAII-styled CUDA event object from the given CUDA event
    */
    explicit cudaEvent(cudaEvent_t event) : _event(event) { }   

    /**
    @brief constructs an RAII-styled CUDA event object
    */
    cudaEvent() : _event{ cudaEventCreator{}() } { }
    
    /**
    @brief constructs an RAII-styled CUDA event object with the given flag
    */
    explicit cudaEvent(unsigned int flag) : _event{ cudaEventCreator{}(flag) } { }
    
    /**
    @brief disabled copy constructor
    */
    cudaEvent(const cudaEvent&) = delete;
    
    /**
    @brief move constructor
    */
    cudaEvent(cudaEvent&& rhs) : _event{rhs._event} {
      rhs._event = nullptr;
    }

    /**
    @brief destructs the CUDA event
    */
    ~cudaEvent() {
      cudaEventDeleter {} (_event);
    }
    
    /**
    @brief disabled copy assignment
    */
    cudaEvent& operator = (const cudaEvent&) = delete;

    /**
    @brief move assignment
    */
    cudaEvent& operator = (cudaEvent&& rhs) {
      cudaEventDeleter {} (_event);
      _event = rhs._event;
      rhs._event = nullptr;
      return *this;
    }
  
    /**
    @brief implicit conversion to the native CUDA event (cudaEvent_t)

    Returns the underlying event of type @c cudaEvent_t.
    */
    operator cudaEvent_t () const {
      return _event;
    }
    
  private:

    cudaEvent_t _event {nullptr};
};


}  // end of namespace tf -----------------------------------------------------



