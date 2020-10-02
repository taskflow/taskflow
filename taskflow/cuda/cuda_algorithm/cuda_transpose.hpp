#pragma once

#include "../cuda_error.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// row-wise matrix transpose
// ----------------------------------------------------------------------------
//
template <typename T>
__global__ void cuda_transpose(
  const T* d_in, 
  T* d_out,
  size_t rows,
  size_t cols
) {
  __shared__ T tile[32][32];
  size_t x = blockIdx.x * 32 + threadIdx.x;
  size_t y = blockIdx.y * 32 + threadIdx.y;

  for(size_t i = 0; i < 32; i += 8) {
    if(x < cols && (y + i) < rows) {
      tile[threadIdx.y + i][threadIdx.x] = d_in[(y + i) * cols + x];
    }
  }

  __syncthreads();

  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;

  for(size_t i = 0; i < 32; i += 8) {
    if(x < rows && (y + i) < cols) {
      d_out[(y + i) * rows + x] = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}

}  // end of namespace --------------------------------------------------------

