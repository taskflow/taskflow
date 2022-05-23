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
    if(stream) {
      cudaStreamDestroy(stream);
    }
  }
};

/** @private */
struct cudaStreamSynchronizer {
  
  void operator () (cudaStream_t stream) const {
    TF_CHECK_CUDA(
      cudaStreamSynchronize(stream), "failed to synchronize a CUDA stream"
    );
  }

};

// ----------------------------------------------------------------------------
// cudaStream
// ----------------------------------------------------------------------------

/**
@class cudaStream

@brief class to create a CUDA stream in an RAII-styled wrapper

A cudaStream object is an RAII-styled wrapper over a native CUDA stream
(@c cudaStream_t).
A cudaStream object is move-only.
*/
class cudaStream {

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
    @brief replaces the managed stream

    Destructs the managed stream and resets it to the given stream.
    */
    void reset(cudaStream_t stream = nullptr) {
      cudaStreamDeleter {} (_stream);
      _stream = stream;
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
      cudaStreamSynchronizer{}(_stream);
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
// cudaEventCreator and cudaEventDeleter for per-thread event pool
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

// ----------------------------------------------------------------------------
// cudaEvent
// ----------------------------------------------------------------------------

/**
@class cudaEvent

@brief class to create a CUDA event in an RAII-styled wrapper

A cudaEvent object is an RAII-styled wrapper over a native CUDA stream
(@c cudaEvent_t).
A cudaEvent object is move-only.
*/
class cudaEvent {

  public:

    /**
    @brief constructs an RAII-styled object from the given CUDA event
    */
    explicit cudaEvent(cudaEvent_t event) : _event(event) {
    }
    
    /**
    @brief constructs an RAII-styled object for a new CUDA event
    */
    cudaEvent() : _event{ cudaEventCreator{}() } {
    }
    
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
    
    /**
    @brief replaces the managed event

    Destructs the managed event and resets it to the given event.
    */
    void reset(cudaEvent_t event = nullptr) {
      cudaEventDeleter {} (_event);
      _event = event;
    }
    
  private:

    cudaEvent_t _event {nullptr};
};


}  // end of namespace tf -----------------------------------------------------



