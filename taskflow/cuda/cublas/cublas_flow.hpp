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
    
    /**
    @brief constructs a cublasFlow object
     */
    cublasFlow(cudaGraph& graph);
    
    template <typename T>
    cudaTask amax(int N, const T* x, int incx, int* result) {
      /*
       [](cublasHandle_t){
       }
       */
      //return cudaTask(node);
    }

  private:

    //cudaFlow& _cudaflow;
    cudaGraph& _graph;

};

// Constructor
inline cublasFlow::cublasFlow(cudaGraph& graph) : _graph {graph} {
}

// ----------------------------------------------------------------------------
// cudaFlow 
// ----------------------------------------------------------------------------

// Function: subflow
template <typename C, std::enable_if_t<is_cublas_subflow_v<C>, void>*>
cudaTask cudaFlow::subflow(C&& c) {

  auto node = _graph.emplace_back(std::in_place_type_t<cudaNode::Subflow>{});
  
  auto& h = std::get<cudaNode::Subflow>(node->_handle);

  cublasFlow cbf(h.graph);
  
  c(cbf);
  
  return cudaTask(node);
}

}  // end of namespace tf -----------------------------------------------------







