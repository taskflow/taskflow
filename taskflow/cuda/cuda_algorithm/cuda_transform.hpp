#pragma once

#include "../cuda_error.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// transform
// ----------------------------------------------------------------------------

// Kernel: for_each
template <typename I, typename F, typename... S>
__global__ void cuda_transform(I first, size_t N, F op, S... srcs) {
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    //data[i] = op(src[i]...);
    *(first + i) = op((*(srcs+i))...);
  }
}

}  // end of namespace tf -----------------------------------------------------






