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

  public:

    template <typename T>
    cudaTask amax(int N, const T* x, int incx, int* result) {
      /*
       [](cublasHandle_t){
       }
       */
      //return cudaTask(node);
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

  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Childflow>{}
  );
  
  auto& h = std::get<cudaNode::Childflow>(node->_handle);

  auto ptr = cublas_per_thread_handle_pool.acquire(_device);

  cublasFlow cbf(h.graph, ptr->native_handle);
  
  c(cbf);

  cublas_per_thread_handle_pool.release(std::move(ptr));
  
  return cudaTask(node);
}

}  // end of namespace tf -----------------------------------------------------







