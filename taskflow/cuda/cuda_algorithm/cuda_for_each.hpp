#pragma once

#include "../cuda_error.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// single_task
// ----------------------------------------------------------------------------

// Kernel: single_task
template <typename C>
__global__ void cuda_single_task(C callable) {
  callable();
}

// ----------------------------------------------------------------------------
// for_each
// ----------------------------------------------------------------------------

// Kernel: for_each
template <typename I, typename F>
__global__ void cuda_for_each(I first, size_t N, F op) {
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    op(*(first+i));
  }
}

// ----------------------------------------------------------------------------
// for_each_index
// ----------------------------------------------------------------------------

// Kernel: for_each_index
template <typename I, typename F>
__global__ void cuda_for_each_index(I beg, I inc, size_t N, F op) {
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    op(static_cast<I>(i)*inc + beg);
  }
}


}  // end of namespace tf -----------------------------------------------------






