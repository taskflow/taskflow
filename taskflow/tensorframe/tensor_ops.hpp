#pragma once

#include "tensor.hpp"

namespace tf {

// TODO
template <typename T>
void tensor_add (Tensor<T>& res, Tensor<T>& lhs, Tensor<T>& rhs) {

  if(res._shape != lhs._shape || lhs._shape != rhs._shape) {
    TF_THROW("tensor shapes do not match!");
  }

  // case 1: all tensors have data in memory
  if(res._storage_level == MEMORY && 
     lhs._storage_level == MEMORY && 
     rhs._storage_level == MEMORY) {
     
    return;
  }

  // case 2: TODO

}


}  // end of namespace tf -----------------------------------------------------
