#pragma once

#include "cuda_task.hpp"
#include "cuda_algorithm/cuda_for_each.hpp"

namespace tf {

/**
@class cudaFlowCapturerBase

@brief base class of methods to capture CUDA operations through
       CUDA streams
*/
class cudaFlowCapturerBase {

  friend class cudaFlowCapturer;

  public:

    /**
    @brief default constructor
     */
    cudaFlowCapturerBase() = default;

    /**
    @brief default virtual destructor
     */
    virtual ~cudaFlowCapturerBase() = default;
  
    /**
    @brief runs a callable with only a single kernel thread

    @tparam C callable type

    @param callable callable to run by a single kernel thread
    */
    template <typename C>
    cudaTask single_task(C&& callable);
    
    /**
    @brief captures a sequential CUDA operations from the given callable
    
    @tparam C callable type constructible with @c std::function<void(cudaStream_t)>
    @param callable a callable to capture CUDA operations with the stream

    This methods applies a stream created by the flow to capture 
    a sequence of CUDA operations defined in the callable.
    */
    template <typename C>
    cudaTask on(C&& callable);
    
    /**
    @brief captures a memcpy task
    
    This method effectively calls tf::cuda_memcpy_async with packed
    arguments <tt>(stream, args...)</tt> where @c stream is managed
    by the flow capturer.
    */ 
    template <typename... ArgsT>
    cudaTask memcpy(ArgsT&&... args);

    /**
    @brief captures a memset task
    
    This method effectively calls tf::cuda_memset_async with packed
    arguments <tt>(stream, args...)</tt> where @c stream is managed
    by the flow capturer.
    */ 
    template <typename... ArgsT>
    cudaTask memset(ArgsT&&... args);

    /**
    @brief captures a kernel task
    
    This method effectively calls tf::cuda_offload_async with packed
    arguments <tt>(stream, args...)</tt> where @c stream is managed
    by the flow capturer.

    The arguments @c args are in the order of (1) grid dimsntion,
    (2) block dimension, (3) shared memory size, (4) kernel function,
    and (5) parameters to pass to the kernel function.
    */ 
    template <typename... ArgsT>
    cudaTask kernel(ArgsT&&... args);

  private:

    cudaGraph* _graph {nullptr};

    cudaFlowCapturerBase(cudaGraph&);
};

// Constructor
inline cudaFlowCapturerBase::cudaFlowCapturerBase(cudaGraph& g) :
  _graph {&g} {
}

// Function: single_task
template <typename C>
cudaTask cudaFlowCapturerBase::single_task(C&& callable) {
  return on([c=std::forward<C>(callable)] (cudaStream_t stream) mutable {
    cuda_single_task<C><<<1, 1, 0, stream>>>(c);
  });
}

// Function: capture
template <typename C>
cudaTask cudaFlowCapturerBase::on(C&& callable) {
  auto node = _graph->emplace_back(*_graph,
    std::in_place_type_t<cudaNode::Capture>{}, std::forward<C>(callable)
  );
  return cudaTask(node);
}

// Function: memcpy
template <typename... ArgsT>
cudaTask cudaFlowCapturerBase::memcpy(ArgsT&&... args) {
  return on([args...] (cudaStream_t stream) mutable {
    cuda_memcpy_async(stream, args...);
  });
}

// Function: memset
template <typename... ArgsT>
cudaTask cudaFlowCapturerBase::memset(ArgsT&&... args) {
  return on([args...] (cudaStream_t stream) mutable {
    cuda_memset_async(stream, args...);
  });
}
    
// Function: kernel
template <typename... ArgsT>
cudaTask cudaFlowCapturerBase::kernel(ArgsT&&... args) {
  return on([args...] (cudaStream_t stream) mutable {
    cuda_offload_async(stream, args...);
  });
}



// ----------------------------------------------------------------------------
// cudaFlowCapturer
// ----------------------------------------------------------------------------

/**
@class cudaFlowCapturer

@brief class object to construct a CUDA graph through stream capture

A %cudaFlowCapturer inherits all the base methods from tf::cudaFlowCapturerBase 
to construct a CUDA graph through stream capturer. 
This class also defines a factory interface tf::cudaFlowCapturer::make_capturer 
for users to create custom capturers and manages their lifetimes.

*/
class cudaFlowCapturer : public cudaFlowCapturerBase {

  friend class cudaFlow;

  public:
    
    /**
    @brief creates a custom capturer derived from tf::cudaFlowCapturerBase
     */
    template <typename T, typename... ArgsT>
    T* make_capturer(ArgsT&&... args);

  private:
    
    std::vector<std::unique_ptr<cudaFlowCapturerBase>> _capturers;

    cudaFlowCapturer(cudaGraph&);
    
    cudaGraph_t _capture();
};

// constructor
inline cudaFlowCapturer::cudaFlowCapturer(cudaGraph& g) :
  cudaFlowCapturerBase{g} {
}

// Function: make_capturer
template <typename T, typename... ArgsT>
T* cudaFlowCapturer::make_capturer(ArgsT&&... args) {
  auto ptr = std::make_unique<T>(std::forward<ArgsT>(args)...);
  ptr->_graph = this->_graph;
  auto raw = ptr.get();
  _capturers.push_back(std::move(ptr));
  return raw;
}

// Procedure
inline cudaGraph_t cudaFlowCapturer::_capture() {

  // acquire per-thread stream and turn it into capture mode
  cudaScopedPerThreadStream stream;
  start_stream_capture(stream);

  // TODO: need an efficient algorithm
  auto ordered = _graph->_toposort();
  for(auto& node : ordered) {
    std::get<cudaNode::Capture>(node->_handle).work(stream);  
  }

  auto g = cease_stream_capture(stream);
  
  //cuda_dump_graph(std::cout, g);
  
  return g;
}

}  // end of namespace tf -----------------------------------------------------

