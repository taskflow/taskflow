#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cublas.hpp>

// ----------------------------------------------------------------------------
// utilities
// ----------------------------------------------------------------------------

template <typename T>
std::vector<T> transpose(int M, int N, std::vector<T>& in) {
  std::vector<T> out(in.size());
  for(int i=0; i<M; i++) {
    for(int j=0; j<N; j++) {
      out[i*N + j] = in[j*N + i];
    }
  }
  return out;
}

// ----------------------------------------------------------------------------
// Testcase: gemm and c_gemm
// ----------------------------------------------------------------------------

template <typename T, bool ROW_MAJOR>
void gemm(
  const int M, 
  const int N, 
  const int K,
  const std::vector<T>& hA,
  const std::vector<T>& hB,
  const std::vector<T>& golden,
  bool tranA,
  bool tranB
) {
  
  tf::Taskflow taskflow;
  tf::Executor executor;

  for(size_t d=0; d<tf::cuda_get_num_devices(); d++) {

    auto dA = tf::cuda_malloc_device<T>(K*M, d);
    auto dB = tf::cuda_malloc_device<T>(K*N, d);
    auto dC = tf::cuda_malloc_device<T>(M*N, d);
    auto dAlpha = tf::cuda_malloc_shared<T>(1);
    auto dBeta  = tf::cuda_malloc_shared<T>(1);

    *dAlpha = 1;
    *dBeta  = 0;
  
    T* hC = new T[N*M];

    auto cudaflow = taskflow.emplace_on([=, &hA, &hB](tf::cudaFlow& cf){

      REQUIRE(tf::cuda_get_device() == d);
      
      auto copyA = cf.copy(dA, hA.data(), K*M);
      auto copyB = cf.copy(dB, hB.data(), K*N);

      tf::cudaTask gemm; 
      
      if(tranA && !tranB) {        // C = A^T * B (r-major)
        if constexpr(ROW_MAJOR) {
          gemm = cf.childflow([&](tf::cublasFlow& flow){
            flow.c_gemm(
              CUBLAS_OP_T, CUBLAS_OP_N,
              M, N, K, dAlpha, dA, M, dB, N, dBeta, dC, N
            );
          });
        }
        else {
          gemm = cf.childflow([&](tf::cublasFlow& flow){
            flow.gemm(
              CUBLAS_OP_N, CUBLAS_OP_T,
              N, M, K, dAlpha, dB, N, dA, M, dBeta, dC, N
            );
          });
        }
      }
      else if(!tranA && !tranB) {  // C = A * B (r-major)
        if constexpr (ROW_MAJOR) {
          gemm = cf.childflow([&](tf::cublasFlow& flow){
            flow.c_gemm(
              CUBLAS_OP_N, CUBLAS_OP_N,
              M, N, K, dAlpha, dA, K, dB, N, dBeta, dC, N
            );
          });
        }
        else {
          gemm = cf.childflow([&](tf::cublasFlow& flow){
            flow.gemm(
              CUBLAS_OP_N, CUBLAS_OP_N,
              N, M, K, dAlpha, dB, N, dA, K, dBeta, dC, N
            );
          });
        }
      }
      else if(!tranA && tranB) {   // C = A * B^T (r-major)
        if constexpr(ROW_MAJOR) {
          gemm = cf.childflow([&](tf::cublasFlow& flow){
            flow.c_gemm(
              CUBLAS_OP_N, CUBLAS_OP_T,
              M, N, K, dAlpha, dA, K, dB, K, dBeta, dC, N
            );
          });
        }
        else {
          gemm = cf.childflow([&](tf::cublasFlow& flow){
            flow.gemm(
              CUBLAS_OP_T, CUBLAS_OP_N,
              N, M, K, dAlpha, dB, K, dA, K, dBeta, dC, N
            );
          });
        }
      }
      else {                       // C = A^T * B^T (r-major)
        if constexpr(ROW_MAJOR) {
          gemm = cf.childflow([&](tf::cublasFlow& flow){
            flow.c_gemm(
              CUBLAS_OP_T, CUBLAS_OP_T,
              M, N, K, dAlpha, dA, M, dB, K, dBeta, dC, N
            );
          });
        }
        else {
          gemm = cf.childflow([&](tf::cublasFlow& flow){
            flow.gemm(
              CUBLAS_OP_T, CUBLAS_OP_T,
              N, M, K, dAlpha, dB, K, dA, M, dBeta, dC, N
            );
          });
        }
      }
      
      auto copyC = cf.copy(hC, dC, M*N);

      gemm.precede(copyC)
          .succeed(copyA, copyB);
    }, d);

    auto verify = taskflow.emplace([=, &golden](){
      for(size_t i=0; i<golden.size(); i++) {
        REQUIRE(std::fabs(hC[i]-golden[i]) < 0.0001);
      }
      tf::cuda_free(dA);
      tf::cuda_free(dB);
      tf::cuda_free(dC);
      tf::cuda_free(dAlpha);
      tf::cuda_free(dBeta);
      delete [] hC;
    });
    
    cudaflow.precede(verify);
  }

  executor.run(taskflow).wait();
}

// C = A^T * B
template <typename T, bool ROW_MAJOR>
void gemm_tn() {

  int N = 4, M = 2, K = 3;

  const std::vector<T> hA = {
    11, 14,
    12, 15,
    13, 16
  };  // 3x2

  const std::vector<T> hB = {
    11, 12, 13, 14,
    15, 16, 17, 18,
    19, 20, 21, 22
  };  // 3x4

  const std::vector<T> golden = {
    548, 584, 620, 656,
    683, 728, 773, 818 
  };  // 2x4
  
  gemm<T, ROW_MAJOR>(M, N, K, hA, hB, golden, true, false);
}

// C = A * B
template <typename T, bool ROW_MAJOR>
void gemm_nn() {

  int N = 4, M = 2, K = 3;

  const std::vector<T> hA = {
    11, 12, 13, 
    14, 15, 16
  };

  const std::vector<T> hB = {
    11, 12, 13, 14,
    15, 16, 17, 18,
    19, 20, 21, 22
  };

  const std::vector<T> golden = {
    548, 584, 620, 656,
    683, 728, 773, 818 
  }; //  MxN

  gemm<T, ROW_MAJOR>(M, N, K, hA, hB, golden, false, false);
}

// C = A * B^T
template <typename T, bool ROW_MAJOR>
void gemm_nt() {

  int M = 2, N = 4, K = 3;

  const std::vector<T> hA = {
    11, 12, 13, 
    14, 15, 16
  }; // MxK

  const std::vector<T> hB = {
    11, 15, 19,
    12, 16, 20,
    13, 17, 21,
    14, 18, 22
  }; // NxK

  const std::vector<T> golden = {
    548, 584, 620, 656,
    683, 728, 773, 818 
  }; //  MxN

  gemm<T, ROW_MAJOR>(M, N, K, hA, hB, golden, false, true);
}

// C = A^T * B^T
template <typename T, bool ROW_MAJOR>
void gemm_tt() {

  int M = 2, N = 4, K = 3;

  const std::vector<T> hA = {
    11, 14,
    12, 15,
    13, 16
  }; // KxM

  const std::vector<T> hB = {
    11, 15, 19,
    12, 16, 20,
    13, 17, 21,
    14, 18, 22
  }; // NxK

  const std::vector<T> golden = {
    548, 584, 620, 656,
    683, 728, 773, 818 
  }; //  MxN

  gemm<T, ROW_MAJOR>(M, N, K, hA, hB, golden, true, true);
}

// gemm (column-major)
TEST_CASE("gemm_nn.float") {
  gemm_nn<float, false>();
}

TEST_CASE("gemm_nn.double") {
  gemm_nn<double, false>();
}

TEST_CASE("gemm_tn.float") {
  gemm_tn<float, false>();
}

TEST_CASE("gemm_tn.double") {
  gemm_tn<double, false>();
}

TEST_CASE("gemm_nt.float") {
  gemm_nt<float, false>();
}

TEST_CASE("gemm_nt.double") {
  gemm_nt<double, false>();
}

TEST_CASE("gemm_tt.float") {
  gemm_tt<float, false>();
}

TEST_CASE("gemm_tt.double") {
  gemm_tt<double, false>();
}

// c_gemm (row_major)
TEST_CASE("c_gemm_nn.float") {
  gemm_nn<float, true>();
}

TEST_CASE("c_gemm_nn.double") {
  gemm_nn<double, true>();
}

TEST_CASE("c_gemm_tn.float") {
  gemm_tn<float, true>();
}

TEST_CASE("c_gemm_tn.double") {
  gemm_tn<double, true>();
}

TEST_CASE("c_gemm_nt.float") {
  gemm_nt<float, true>();
}

TEST_CASE("c_gemm_nt.double") {
  gemm_nt<double, true>();
}

TEST_CASE("c_gemm_tt.float") {
  gemm_tt<float, true>();
}

TEST_CASE("c_gemm_tt.double") {
  gemm_tt<double, true>();
}
