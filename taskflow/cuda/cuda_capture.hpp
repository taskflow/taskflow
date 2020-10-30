#pragma once

#include "cuda_task.hpp"
#include "cuda_algorithm/cuda_for_each.hpp"

namespace tf {

/**
@class cudaFlowCapturer

@brief class object to construct a CUDA graph through stream capture
*/
class cudaFlowCapturer {

  friend class cudaFlow;

  public:
    
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

  protected:

    cudaGraph& _graph;

    cudaFlowCapturer(cudaGraph&);

    cudaGraph_t _capture();
};

// constructor
inline cudaFlowCapturer::cudaFlowCapturer(cudaGraph& g) : _graph {g} {
}

// Function: single_task
template <typename C>
cudaTask cudaFlowCapturer::single_task(C&& callable) {
  auto node = _graph.emplace_back(_graph,
    std::in_place_type_t<cudaNode::Capture>{},
    [c=std::forward<C>(callable)] (cudaStream_t stream) mutable {
      cuda_single_task<C><<<1, 1, 0, stream>>>(c);
    }
  );
  return cudaTask(node);
}

// Function: capture
template <typename C>
cudaTask cudaFlowCapturer::on(C&& callable) {
  auto node = _graph.emplace_back(_graph,
    std::in_place_type_t<cudaNode::Capture>{}, std::forward<C>(callable)
  );
  return cudaTask(node);
}

// Procedure
inline cudaGraph_t cudaFlowCapturer::_capture() {

  // acquire per-thread stream and turn it into capture mode
  cudaScopedPerThreadStream stream;
  start_stream_capture(stream);

  // TODO: need an efficient algorithm
  auto ordered = _graph._toposort();
  for(auto& node : ordered) {
    std::get<cudaNode::Capture>(node->_handle).work(stream);  
  }

  return cease_stream_capture(stream);
}


}  // end of namespace tf -----------------------------------------------------
