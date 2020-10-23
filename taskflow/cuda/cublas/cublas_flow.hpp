#pragma once

#include "cublas_handle.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// cublasFlow definition
// ----------------------------------------------------------------------------

/**
@brief class object to construct a cuBLAS task graph
*/
class cublasFlow {

  friend class cudaFlow;

  public:

    template <typename T>
    std::enable_if_t<std::is_same_v<T, float>, cudaTask>
    amax(int n, const T* x, int incx, int* result) {
      auto node = _graph.emplace_back(_graph,
        std::in_place_type_t<cudaNode::Capture>{},
        [&, n, x, incx, result] () mutable {
          TF_CHECK_CUBLAS(
            cublasIsamax(_native_handle, n, x, incx, result),
            "failed to launch cublasIsamax"
          );
        }
      );
      return cudaTask(node);
    }

  private:

    cudaGraph& _graph;

    cublasHandle_t _native_handle;

    cublasFlow(cudaGraph&, cublasHandle_t);
};

// Constructor
inline cublasFlow::cublasFlow(cudaGraph& graph, cublasHandle_t handle) : 
  _graph {graph}, _native_handle {handle} {
}

// ----------------------------------------------------------------------------
// cudaFlow 
// ----------------------------------------------------------------------------

// Function: childflow
template <typename C, std::enable_if_t<is_cublasflow_v<C>, void>*>
cudaTask cudaFlow::childflow(C&& c) {
  
  // insert a childflow node
  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Childflow>{}
  );
  
  auto& node_handle = std::get<cudaNode::Childflow>(node->_handle);
  
  // acquire per-thread cublas handle and stream
  cublasScopedPerThreadHandle cublas_handle(_device);
  cudaScopedPerThreadStream stream(_device);
  
  // turn the stream into capture and associate it with the cublas handle
  TF_CHECK_CUDA(
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), 
    "failed to turn stream into capture mode"
  );
  cublasSetStream(cublas_handle, stream);
  
  // construct a cublas flow from the callable
  cublasFlow cbf(node_handle.graph, cublas_handle);

  c(cbf);

  // TODO (dian-lun): need to topologically sort the nodes
  // for now I didn't do anything but just assume a linear chain
  for(auto& node : node_handle.graph._nodes) {
    std::get<cudaNode::Capture>(node->_handle).work();  
  }

  cudaGraph_t graph;
  
  // stop the capture to get a cuda graph
  TF_CHECK_CUDA(cudaStreamEndCapture(stream, &graph), "failed to end capture");

  //cuda_dump_graph(std::cout, graph);

  TF_CHECK_CUDA(
    cudaGraphAddChildGraphNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, graph
    ), 
    "failed to add a cuda childflow task"
  );
  
  TF_CHECK_CUDA(cudaGraphDestroy(graph), "failed to destroy captured graph");
  
  return cudaTask(node);
}

}  // end of namespace tf -----------------------------------------------------







