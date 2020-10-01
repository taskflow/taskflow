#pragma once

#include "cuda_graph.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// for_each
// ----------------------------------------------------------------------------

// Kernel: for_each
template <typename I, typename F>
__global__ void cuda_for_each(I first, size_t N, F functor) {
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    functor(*(first+i));
  }
}

// ----------------------------------------------------------------------------
// for_each
// ----------------------------------------------------------------------------

// Kernel: for_each_index
template <typename I, typename F>
__global__ void cuda_for_each_index(I beg, I inc, size_t N, F functor) {
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    functor(static_cast<I>(i)*inc + beg);
  }
}

// ----------------------------------------------------------------------------
// transform
// ----------------------------------------------------------------------------

// Kernel: for_each
template <typename T, typename F, typename... S>
__global__ void cuda_transform(T* data, size_t N, F functor, S*... src) {
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    data[i] = functor(src[i]...);
  }
}

// ----------------------------------------------------------------------------
// row-wise matrix transpose
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// reduce
// ----------------------------------------------------------------------------

template <typename T, typename O>
__global__ void cuda_reduce(T* data, size_t N, T& result, O&& bop) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
}

}  // end of namespace tf -----------------------------------------------------







