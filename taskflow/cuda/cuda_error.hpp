#pragma once

#include <cuda.h>
#include <iostream>
#include <sstream>
#include <exception>

#include "../utility/stream.hpp"

#define TF_CUDA_EXPAND( x ) x
#define TF_CUDA_REMOVE_FIRST_HELPER(N, ...) __VA_ARGS__
#define TF_CUDA_REMOVE_FIRST(...) TF_CUDA_EXPAND(TF_CUDA_REMOVE_FIRST_HELPER(__VA_ARGS__))
#define TF_CUDA_GET_FIRST_HELPER(N, ...) N
#define TF_CUDA_GET_FIRST(...) TF_CUDA_EXPAND(TF_CUDA_GET_FIRST_HELPER(__VA_ARGS__))

#define TF_CHECK_CUDA(...)                                       \
if(TF_CUDA_GET_FIRST(__VA_ARGS__) != cudaSuccess) {              \
  std::ostringstream oss;                                        \
  auto __ev__ = TF_CUDA_GET_FIRST(__VA_ARGS__);                  \
  oss << "[" << __FILE__ << ":" << __LINE__ << "] "              \
      << (cudaGetErrorString(__ev__)) << " ("                    \
      << (cudaGetErrorName(__ev__)) << ") - ";                   \
  tf::ostreamize(oss, TF_CUDA_REMOVE_FIRST(__VA_ARGS__));        \
  throw std::runtime_error(oss.str());                           \
}

