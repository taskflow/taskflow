#pragma once

#include <cuda.h>
#include <iostream>
#include <sstream>
#include <exception>

#include "../utility/stream.hpp"

#define TF_CUDA_REMOVE_FIRST_HELPER(N, ...) __VA_ARGS__
#define TF_CUDA_REMOVE_FIRST(...) TF_CUDA_REMOVE_FIRST_HELPER(__VA_ARGS__)
#define TF_CUDA_GET_FIRST_HELPER(N, ...) N
#define TF_CUDA_GET_FIRST(...) TF_CUDA_GET_FIRST_HELPER(__VA_ARGS__)

#define TF_CHECK_CUDA(...)                                       \
if(TF_CUDA_GET_FIRST(__VA_ARGS__) != cudaSuccess) {              \
  std::ostringstream oss;                                        \
  auto ev = TF_CUDA_GET_FIRST(__VA_ARGS__);                      \
  auto unknown_str  = "unknown error";                           \
  auto unknown_name = "cudaErrorUnknown";                        \
  auto error_str  = ::cudaGetErrorString(ev);                    \
  auto error_name = ::cudaGetErrorName(ev);                      \
  oss << "[" << __FILE__ << ":" << __LINE__ << "] "              \
      << (error_str  ? error_str  : unknown_str)                 \
      << " ("                                                    \
      << (error_name ? error_name : unknown_name)                \
      << ") - ";                                                 \
  tf::ostreamize(oss, TF_CUDA_REMOVE_FIRST(__VA_ARGS__));  \
  throw std::runtime_error(oss.str());                           \
}

