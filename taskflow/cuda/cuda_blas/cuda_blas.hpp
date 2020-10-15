#pragma once

#include "cuda_blas_handle.hpp"

namespace tf {


class cudaBLASNode {
  
   
  private:

    std::string _name;

    std::vector<cudaBLASNode*> _successors;
};

class cudaBLAS {

  public:
    
    template <typename T>
    cudaBLASNode* amax(int N, const T* x, int incx, int* result); 

  private:

};

//cf.blas([] (tf::cudaBLAS& blas) {
//  auto task = blas.gemm<float>(); 
//  auto task2 = blas.for_each
//
//  tf::cudaBLAS blas(cf);
//  blas.
//});


}  // end of namespace tf -----------------------------------------------------
