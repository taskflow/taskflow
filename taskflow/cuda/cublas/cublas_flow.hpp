#pragma once

#include "cublas_handle.hpp"

namespace tf {

class cublasNode {
  
   
  private:

    std::string _name;

    std::vector<cublasNode*> _successors;
};

class cublasFlow {

  public:
    
    template <typename T>
    cublasNode* amax(int N, const T* x, int incx, int* result); 

  private:

    cudaFlow& _cudaflow;

};


// ----------------------------------------------------------------------------
// cudaFlow::child([](cublasFlow&))
// ----------------------------------------------------------------------------

// Function: child
template <typename C, std::enable_if_t<is_cublas_child_v<C>, void>*>
cudaTask cudaFlow::child(C&& callable) {
}

}  // end of namespace tf -----------------------------------------------------







