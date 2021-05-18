#pragma once

#include <cublas_v2.h>

namespace tf {

// cuBLAS API errors
constexpr const char* cublas_error_to_string(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "unknown cublas error";
}  



#define TF_CHECK_CUBLAS(...)                                   \
if(TF_CUDA_GET_FIRST(__VA_ARGS__) != CUBLAS_STATUS_SUCCESS) {  \
  std::ostringstream oss;                                      \
  auto ev = TF_CUDA_GET_FIRST(__VA_ARGS__);                    \
  auto error_str  = cublas_error_to_string(ev);                \
  oss << "[" << __FILE__ << ":" << __LINE__ << " "             \
      << error_str << "] ";                                    \
  tf::ostreamize(oss, TF_CUDA_REMOVE_FIRST(__VA_ARGS__));      \
  throw std::runtime_error(oss.str());                         \
}


}  // end of namespace tf -----------------------------------------------------

