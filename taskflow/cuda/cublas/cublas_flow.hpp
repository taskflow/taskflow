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
      //TF_CHECK_CUBLAS(
      //  cublasIsamax(_native_handle, n, x, incx, result),
      //  "???"
      //);
      //return cudaTask(nullptr);
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

  //auto node = _graph.emplace_back(
  //  _graph, std::in_place_type_t<cudaNode::Childflow>{}
  //);
  //
  //auto& h = std::get<cudaNode::Childflow>(node->_handle);

  //auto ptr = cublas_per_thread_handle_pool.acquire(_device);

  //cudaStream_t stream;

  //TF_CHECK_CUDA(
  //  cudaStreamCreate(&stream), "failed to create a stream"
  //);

  //TF_CHECK_CUDA(
  //  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), 
  //  "failed to begin capture on a stream"
  //);

  //cublasSetStream(ptr->native_handle, stream);

  //cublasFlow cbf(h.graph, ptr->native_handle);

  //c(cbf);

  //// TODO:
  //// topological_sort(h.graph)
  //for(auto& node : h.graph._nodes) {
  //  cublasSetStream(ptr->native_handle(), /* another stream */)
  //  cudaStreamEvenRecrod(stream);
  //  std::get<cudaNode::Capture>(node->_handle).work();  
  //}


  //cudaGraph_t graph;

  //TF_CHECK_CUDA(
  //  cudaStreamEndCapture(stream, &graph),
  //  "failed to end capture on a stream"
  //);

  //cudaGraphNode_t nodes[100];
  //size_t num_nodes = 100;

  //TF_CHECK_CUDA(
  //  cudaGraphGetNodes(graph, nodes, &num_nodes), "failed to get nodes"
  //);

  //std::cout << num_nodes << '\n';



  //cublas_per_thread_handle_pool.release(std::move(ptr));

  //TF_THROW("ffff");
  //
  ////TF_CHECK_CUDA(
  ////  cudaGraphAddChildNode(
  ////    &node->_native_handle, _graph._native_handle, graph 
  ////  ),
  ////  "failed to create a no-operation (empty) node"
  ////);
  //
  //TF_CHECK_CUDA(
  //  cudaGraphDestroy(graph), "failed to destroy captured graph"
  //);
  //
  //return cudaTask(node);
}

}  // end of namespace tf -----------------------------------------------------







