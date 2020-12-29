#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cublasflow.hpp>

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

template <typename T>
void print_matrix(int M, int N, const std::vector<T>& mat) {
  for(int i=0; i<M; i++) {
    for(int j=0; j<N; j++) {
      std::cout << mat[i*N+j] << ' ';
    }
    std::cout << '\n';
  }
}

// ----------------------------------------------------------------------------

template <typename T>
void geam(
  bool row_major,
  const int M, 
  const int N, 
  const std::vector<T>& hA,
  const std::vector<T>& hB,
  const std::vector<T>& golden,
  bool tranA,
  bool tranB
) {
  
  tf::Taskflow taskflow;
  tf::Executor executor;

  for(size_t d=0; d<tf::cuda_get_num_devices(); d++) {

    auto dA = tf::cuda_malloc_device<T>(M*N, d);
    auto dB = tf::cuda_malloc_device<T>(M*N, d);
    auto dC = tf::cuda_malloc_device<T>(M*N, d);
    auto dAlpha = tf::cuda_malloc_device<T>(1, d);
    auto dBeta  = tf::cuda_malloc_device<T>(1, d);

    T* hC = new T[N*M];

    auto cudaflow = taskflow.emplace_on([=, &hA, &hB](tf::cudaFlow& cf){

      REQUIRE(tf::cuda_get_device() == d);
      
      auto copyA = cf.copy(dA, hA.data(), M*N);
      auto copyB = cf.copy(dB, hB.data(), M*N);
      auto alpha = cf.single_task([=] __device__ () { *dAlpha = 1; });
      auto beta  = cf.single_task([=] __device__ () { *dBeta  = 2; });

      tf::cudaTask geam; 
      
      if(tranA && !tranB) {        // C = A^T + B
        if (row_major) {
          geam = cf.capture([&](tf::cudaFlowCapturer& cap){
            cap.make_capturer<tf::cublasFlowCapturer>()->c_geam(
              CUBLAS_OP_T, CUBLAS_OP_N,
              M, N, dAlpha, dA, M, dBeta, dB, N, dC, N
            );
          });
        }
        else {
          geam = cf.capture([&](tf::cudaFlowCapturer& cap){
            cap.make_capturer<tf::cublasFlowCapturer>()->geam(
              CUBLAS_OP_T, CUBLAS_OP_N,
              N, M, dAlpha, dA, M, dBeta, dB, N, dC, N
            );
          });
        }
      }
      else if(!tranA && !tranB) {  // C = A + B (r-major)
        if (row_major) {
          geam = cf.capture([&](tf::cudaFlowCapturer& cap){
            cap.make_capturer<tf::cublasFlowCapturer>()->c_geam(
              CUBLAS_OP_N, CUBLAS_OP_N,
              M, N, dAlpha, dA, N, dBeta, dB, N, dC, N
            );
          });
        }
        else {
          geam = cf.capture([&](tf::cudaFlowCapturer& cap){
            cap.make_capturer<tf::cublasFlowCapturer>()->geam(
              CUBLAS_OP_N, CUBLAS_OP_N,
              N, M, dAlpha, dA, N, dBeta, dB, N, dC, N
            );
          });
        }
      }
      else if(!tranA && tranB) {   // C = A + B^T (r-major)
        if(row_major) {
          geam = cf.capture([&](tf::cudaFlowCapturer& cap){
            cap.make_capturer<tf::cublasFlowCapturer>()->c_geam(
              CUBLAS_OP_N, CUBLAS_OP_T,
              M, N, dAlpha, dA, N, dBeta, dB, M, dC, N
            );
          });
        }
        else {
          geam = cf.capture([&](tf::cudaFlowCapturer& cap){
            cap.make_capturer<tf::cublasFlowCapturer>()->geam(
              CUBLAS_OP_N, CUBLAS_OP_T,
              N, M, dAlpha, dA, N, dBeta, dB, M, dC, N
            );
          });
        }
      }
      else {                       // C = A^T * B^T (r-major)
        if (row_major) {
          geam = cf.capture([&](tf::cudaFlowCapturer& cap){
            cap.make_capturer<tf::cublasFlowCapturer>()->c_geam(
              CUBLAS_OP_T, CUBLAS_OP_T,
              M, N, dAlpha, dA, M, dBeta, dB, M, dC, N
            );
          });
        }
        else {
          geam = cf.capture([&](tf::cudaFlowCapturer& cap){
            cap.make_capturer<tf::cublasFlowCapturer>()->geam(
              CUBLAS_OP_T, CUBLAS_OP_T,
              N, M, dAlpha, dA, M, dBeta, dB, M, dC, N
            );
          });
        }
      }
      
      auto copyC = cf.copy(hC, dC, M*N);

      geam.precede(copyC)
          .succeed(copyA, copyB, alpha, beta);
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

// C = A^T + B
template <typename T>
void geam_tn(bool row_major) {

  int M = 2, N = 3;

  const std::vector<T> hA = {
    11, 14,
    12, 15,
    13, 16
  };  // 3x2

  const std::vector<T> hB = {
     1,  1,  1,
    -1, -1, -1
  };  // 2x3

  const std::vector<T> golden = {
    13, 14, 15,
    12, 13, 14
  };  // 2x3
  
  geam<T>(row_major, M, N, hA, hB, golden, true, false);
}

// C = A + B
template <typename T>
void geam_nn(bool row_major) {

  int M = 2, N = 3;

  const std::vector<T> hA = {
    11, 12, 13,
    14, 15, 16
  };  // 2x3

  const std::vector<T> hB = {
     1,  1,  1,
    -1, -1, -1
  };  // 2x3

  const std::vector<T> golden = {
    13, 14, 15,
    12, 13, 14
  };  // 2x3
  
  geam<T>(row_major, M, N, hA, hB, golden, false, false);
}

// C = A + B^T
template <typename T>
void geam_nt(bool row_major) {

  int M = 2, N = 3;

  const std::vector<T> hA = {
    11, 12, 13,
    14, 15, 16
  };  // 2x3

  const std::vector<T> hB = {
    1, -1,
    1, -1,
    1, -1
  };  // 3x2

  const std::vector<T> golden = {
    13, 14, 15,
    12, 13, 14
  };  // 2x3
  
  geam<T>(row_major, M, N, hA, hB, golden, false, true);
}

// C = A^T + B^T
template <typename T>
void geam_tt(bool row_major) {

  int M = 2, N = 3;

  const std::vector<T> hA = {
    11, 14,
    12, 15,
    13, 16
  };  // 3x2

  const std::vector<T> hB = {
    1, -1,
    1, -1,
    1, -1
  };  // 3x2

  const std::vector<T> golden = {
    13, 14, 15,
    12, 13, 14
  };  // 2x3
  
  geam<T>(row_major, M, N, hA, hB, golden, true, true);
}

// column major
TEST_CASE("geam_tn.float" * doctest::timeout(300)) {
  geam_tn<float>(false);
}

TEST_CASE("geam_nn.float" * doctest::timeout(300)) {
  geam_nn<float>(false);
}

TEST_CASE("geam_nt.float" * doctest::timeout(300)) {
  geam_nt<float>(false);
}

TEST_CASE("geam_tt.float" * doctest::timeout(300)) {
  geam_tt<float>(false);
}

TEST_CASE("geam_tn.double" * doctest::timeout(300)) {
  geam_tn<double>(false);
}

TEST_CASE("geam_nn.double" * doctest::timeout(300)) {
  geam_nn<double>(false);
}

TEST_CASE("geam_nt.double" * doctest::timeout(300)) {
  geam_nt<double>(false);
}

TEST_CASE("geam_tt.double" * doctest::timeout(300)) {
  geam_tt<double>(false);
}

// row major
TEST_CASE("c_geam_tn.float" * doctest::timeout(300)) {
  geam_tn<float>(true);
}

TEST_CASE("c_geam_nn.float" * doctest::timeout(300)) {
  geam_nn<float>(true);
}

TEST_CASE("c_geam_nt.float" * doctest::timeout(300)) {
  geam_nt<float>(true);
}

TEST_CASE("c_geam_tt.float" * doctest::timeout(300)) {
  geam_tt<float>(true);
}

TEST_CASE("c_geam_tn.double" * doctest::timeout(300)) {
  geam_tn<double>(true);
}

TEST_CASE("c_geam_nn.double" * doctest::timeout(300)) {
  geam_nn<double>(true);
}

TEST_CASE("c_geam_nt.double" * doctest::timeout(300)) {
  geam_nt<double>(true);
}

TEST_CASE("c_geam_tt.double" * doctest::timeout(300)) {
  geam_tt<double>(true);
}

// ----------------------------------------------------------------------------
// Testcase: gemm and c_gemm
// ----------------------------------------------------------------------------

template <typename T>
void gemm(
  bool row_major,
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
    auto dAlpha = tf::cuda_malloc_device<T>(1, d);
    auto dBeta  = tf::cuda_malloc_device<T>(1, d);

    T* hC = new T[N*M];

    auto cudaflow = taskflow.emplace_on([=, &hA, &hB](tf::cudaFlow& cf){

      REQUIRE(tf::cuda_get_device() == d);
      
      auto copyA = cf.copy(dA, hA.data(), K*M);
      auto copyB = cf.copy(dB, hB.data(), K*N);
      auto alpha = cf.single_task([=] __device__ () { *dAlpha = 1; });
      auto beta  = cf.single_task([=] __device__ () { *dBeta  = 0; });

      tf::cudaTask gemm; 
      
      if(tranA && !tranB) {        // C = A^T * B (r-major)
        if (row_major) {
          gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
            flow.make_capturer<tf::cublasFlowCapturer>()->c_gemm(
              CUBLAS_OP_T, CUBLAS_OP_N,
              M, N, K, dAlpha, dA, M, dB, N, dBeta, dC, N
            );
          });
        }
        else {
          gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
            flow.make_capturer<tf::cublasFlowCapturer>()->gemm(
              CUBLAS_OP_N, CUBLAS_OP_T,
              N, M, K, dAlpha, dB, N, dA, M, dBeta, dC, N
            );
          });
        }
      }
      else if(!tranA && !tranB) {  // C = A * B (r-major)
        if (row_major) {
          gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
            flow.make_capturer<tf::cublasFlowCapturer>()->c_gemm(
              CUBLAS_OP_N, CUBLAS_OP_N,
              M, N, K, dAlpha, dA, K, dB, N, dBeta, dC, N
            );
          });
        }
        else {
          gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
            flow.make_capturer<tf::cublasFlowCapturer>()->gemm(
              CUBLAS_OP_N, CUBLAS_OP_N,
              N, M, K, dAlpha, dB, N, dA, K, dBeta, dC, N
            );
          });
        }
      }
      else if(!tranA && tranB) {   // C = A * B^T (r-major)
        if(row_major) {
          gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
            flow.make_capturer<tf::cublasFlowCapturer>()->c_gemm(
              CUBLAS_OP_N, CUBLAS_OP_T,
              M, N, K, dAlpha, dA, K, dB, K, dBeta, dC, N
            );
          });
        }
        else {
          gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
            flow.make_capturer<tf::cublasFlowCapturer>()->gemm(
              CUBLAS_OP_T, CUBLAS_OP_N,
              N, M, K, dAlpha, dB, K, dA, K, dBeta, dC, N
            );
          });
        }
      }
      else {                       // C = A^T * B^T (r-major)
        if (row_major) {
          gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
            flow.make_capturer<tf::cublasFlowCapturer>()->c_gemm(
              CUBLAS_OP_T, CUBLAS_OP_T,
              M, N, K, dAlpha, dA, M, dB, K, dBeta, dC, N
            );
          });
        }
        else {
          gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
            flow.make_capturer<tf::cublasFlowCapturer>()->gemm(
              CUBLAS_OP_T, CUBLAS_OP_T,
              N, M, K, dAlpha, dB, K, dA, M, dBeta, dC, N
            );
          });
        }
      }
      
      auto copyC = cf.copy(hC, dC, M*N);

      gemm.precede(copyC)
          .succeed(copyA, copyB, alpha, beta);
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
template <typename T>
void gemm_tn(bool row_major) {

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
  
  gemm<T>(row_major, M, N, K, hA, hB, golden, true, false);
}

// C = A * B
template <typename T>
void gemm_nn(bool row_major) {

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

  gemm<T>(row_major, M, N, K, hA, hB, golden, false, false);
}

// C = A * B^T
template <typename T>
void gemm_nt(bool row_major) {

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

  gemm<T>(row_major, M, N, K, hA, hB, golden, false, true);
}

// C = A^T * B^T
template <typename T>
void gemm_tt(bool row_major) {

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

  gemm<T>(row_major, M, N, K, hA, hB, golden, true, true);
}

// gemm (column-major)
TEST_CASE("gemm_nn.float" * doctest::timeout(300)) {
  gemm_nn<float>(false);
}

TEST_CASE("gemm_nn.double" * doctest::timeout(300)) {
  gemm_nn<double>(false);
}

TEST_CASE("gemm_tn.float" * doctest::timeout(300)) {
  gemm_tn<float>(false);
}

TEST_CASE("gemm_tn.double" * doctest::timeout(300)) {
  gemm_tn<double>(false);
}

TEST_CASE("gemm_nt.float" * doctest::timeout(300)) {
  gemm_nt<float>(false);
}

TEST_CASE("gemm_nt.double" * doctest::timeout(300)) {
  gemm_nt<double>(false);
}

TEST_CASE("gemm_tt.float" * doctest::timeout(300)) {
  gemm_tt<float>(false);
}

TEST_CASE("gemm_tt.double" * doctest::timeout(300)) {
  gemm_tt<double>(false);
}

// c_gemm (row_major)
TEST_CASE("c_gemm_nn.float" * doctest::timeout(300)) {
  gemm_nn<float>(true);
}

TEST_CASE("c_gemm_nn.double" * doctest::timeout(300)) {
  gemm_nn<double>(true);
}

TEST_CASE("c_gemm_tn.float" * doctest::timeout(300)) {
  gemm_tn<float>(true);
}

TEST_CASE("c_gemm_tn.double" * doctest::timeout(300)) {
  gemm_tn<double>(true);
}

TEST_CASE("c_gemm_nt.float" * doctest::timeout(300)) {
  gemm_nt<float>(true);
}

TEST_CASE("c_gemm_nt.double" * doctest::timeout(300)) {
  gemm_nt<double>(true);
}

TEST_CASE("c_gemm_tt.float" * doctest::timeout(300)) {
  gemm_tt<float>(true);
}

TEST_CASE("c_gemm_tt.double" * doctest::timeout(300)) {
  gemm_tt<double>(true);
}


// ----------------------------------------------------------------------------
// Testcase: gemm_batched and c_gemm_batched
// ----------------------------------------------------------------------------

constexpr size_t S = 10;

template <typename T>
void gemm_batched(
  bool row_major,
  const int M, 
  const int N, 
  const int K,
  const T* hA[],
  const T* hB[],
  const std::vector<T>& golden,
  bool tranA,
  bool tranB
) {
  
  tf::Taskflow taskflow;
  tf::Executor executor;

  int d = 0;
  
  auto dA = tf::cuda_malloc_device<T>(S*K*M, d);
  auto dB = tf::cuda_malloc_device<T>(S*K*N, d);
  auto dC = tf::cuda_malloc_device<T>(S*M*N, d);
  auto dAlpha = tf::cuda_malloc_device<T>(1, d);
  auto dBeta  = tf::cuda_malloc_device<T>(1, d);
  auto hC = new T[S*M*N];

  auto dAs = tf::cuda_malloc_device<T*>(S, d);
  auto dBs = tf::cuda_malloc_device<T*>(S, d);
  auto dCs = tf::cuda_malloc_device<T*>(S, d);

  auto cudaflow = taskflow.emplace([&](tf::cudaFlow& cf){

    tf::cudaTask copyA[S], copyB[S];

    for(size_t s=0; s<S; s++) {
      copyA[s] = cf.copy(dA + s*K*M, hA[s], K*M);
      copyB[s] = cf.copy(dB + s*K*N, hB[s], K*N);
    }

    auto alpha = cf.single_task([=] __device__ () { *dAlpha = 1; });
    auto beta  = cf.single_task([=] __device__ () { *dBeta  = 0; });
    auto array = cf.single_task([=] __device__ () {
      for(size_t s=0; s<S; s++) {
        dAs[s] = dA + s*K*M;
        dBs[s] = dB + s*K*N;
        dCs[s] = dC + s*M*N;
      }
    });

    tf::cudaTask gemm; 
    
    if(!tranA && !tranB) {  // C = A * B (r-major)
      if (row_major) {
        gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
          flow.make_capturer<tf::cublasFlowCapturer>()->c_gemm_batched(CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, dAlpha, (const T**)dAs, K, (const T**)dBs, N, dBeta, dCs, N, S
          );
        });
      }
      else {
        gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
          flow.make_capturer<tf::cublasFlowCapturer>()->gemm_batched(CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, dAlpha, (const T**)dBs, N, (const T**)dAs, K, dBeta, dCs, N, S
          );
        });
      }
    }
    else if(tranA && !tranB) {        // C = A^T * B (r-major)
      if (row_major) {
        gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
          flow.make_capturer<tf::cublasFlowCapturer>()->c_gemm_batched(CUBLAS_OP_T, CUBLAS_OP_N,
            M, N, K, dAlpha, (const T**)dAs, M, (const T**)dBs, N, dBeta, dCs, N, S
          );
        });
      }
      else {
        gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
          flow.make_capturer<tf::cublasFlowCapturer>()->gemm_batched(CUBLAS_OP_N, CUBLAS_OP_T,
            N, M, K, dAlpha, (const T**)dBs, N, (const T**)dAs, M, dBeta, dCs, N, S
          );
        });
      }
    }
    else if(!tranA && tranB) {   // C = A * B^T (r-major)
      if(row_major) {
        gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
          flow.make_capturer<tf::cublasFlowCapturer>()->c_gemm_batched(CUBLAS_OP_N, CUBLAS_OP_T,
            M, N, K, dAlpha, (const T**)dAs, K, (const T**)dBs, K, dBeta, dCs, N, S
          );
        });
      }
      else {
        gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
          flow.make_capturer<tf::cublasFlowCapturer>()->gemm_batched(CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K, dAlpha, (const T**)dBs, K, (const T**)dAs, K, dBeta, dCs, N, S
          );
        });
      }
    }
    else {                       // C = A^T * B^T (r-major)
      if (row_major) {
        gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
          flow.make_capturer<tf::cublasFlowCapturer>()->c_gemm_batched(CUBLAS_OP_T, CUBLAS_OP_T,
            M, N, K, dAlpha, (const T**)dAs, M, (const T**)dBs, K, dBeta, dCs, N, S
          );
        });
      }
      else {
        gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
          flow.make_capturer<tf::cublasFlowCapturer>()->gemm_batched(CUBLAS_OP_T, CUBLAS_OP_T,
            N, M, K, dAlpha, (const T**)dBs, K, (const T**)dAs, M, dBeta, dCs, N, S
          );
        });
      }
    }
    
    gemm.succeed(alpha, beta, array);

    for(size_t s=0; s<S; s++) {
      auto copyC = cf.copy(hC, dC, S*M*N);
      gemm.succeed(copyA[s], copyB[s])
          .precede(copyC);
    }
  });

  auto verify = taskflow.emplace([&](){
    for(size_t s=0; s<S; s++) {
      auto p = hC + s*M*N;
      for(size_t i=0; i<golden.size(); i++) {
        REQUIRE(std::fabs(p[i]-golden[i]) < 0.0001);
      }
    }
    tf::cuda_free(dA);
    tf::cuda_free(dB);
    tf::cuda_free(dC);
    tf::cuda_free(dAlpha);
    tf::cuda_free(dBeta);
    tf::cuda_free(dAs);
    tf::cuda_free(dBs);
    tf::cuda_free(dCs);
    delete [] hC;
  });
  
  cudaflow.precede(verify);
  

  executor.run(taskflow).wait();
}

// C = A * B
template <typename T>
void gemm_batched_nn(bool row_major) {

  const int N = 4, M = 2, K = 3;

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

  const T* hAs[S];
  const T* hBs[S];

  for(size_t s=0; s<S; s++) {
    hAs[s] = hA.data();
    hBs[s] = hB.data();
  }

  gemm_batched<T>(row_major, M, N, K, hAs, hBs, golden, false, false);
}

// C = A^T * B
template <typename T>
void gemm_batched_tn(bool row_major) {

  const int N = 4, M = 2, K = 3;

  const std::vector<T> hA = {
    11, 14,
    12, 15,
    13, 16
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

  const T* hAs[S];
  const T* hBs[S];

  for(size_t s=0; s<S; s++) {
    hAs[s] = hA.data();
    hBs[s] = hB.data();
  }

  gemm_batched<T>(row_major, M, N, K, hAs, hBs, golden, true, false);
}

// C = A * B^T
template <typename T>
void gemm_batched_nt(bool row_major) {

  const int N = 4, M = 2, K = 3;

  const std::vector<T> hA = {
    11, 12, 13, 
    14, 15, 16
  };

  const std::vector<T> hB = {
    11, 15, 19,
    12, 16, 20,
    13, 17, 21,
    14, 18, 22
  };

  const std::vector<T> golden = {
    548, 584, 620, 656,
    683, 728, 773, 818 
  }; //  MxN

  const T* hAs[S];
  const T* hBs[S];

  for(size_t s=0; s<S; s++) {
    hAs[s] = hA.data();
    hBs[s] = hB.data();
  }

  gemm_batched<T>(row_major, M, N, K, hAs, hBs, golden, false, true);
}

// C = A^T * B^T
template <typename T>
void gemm_batched_tt(bool row_major) {

  const int N = 4, M = 2, K = 3;

  const std::vector<T> hA = {
    11, 14,
    12, 15,
    13, 16
  };

  const std::vector<T> hB = {
    11, 15, 19,
    12, 16, 20,
    13, 17, 21,
    14, 18, 22
  };

  const std::vector<T> golden = {
    548, 584, 620, 656,
    683, 728, 773, 818 
  }; //  MxN

  const T* hAs[S];
  const T* hBs[S];

  for(size_t s=0; s<S; s++) {
    hAs[s] = hA.data();
    hBs[s] = hB.data();
  }

  gemm_batched<T>(row_major, M, N, K, hAs, hBs, golden, true, true);
}

// gemm_batched (column-major)
TEST_CASE("gemm_batched_nn.float" * doctest::timeout(300)) {
  gemm_batched_nn<float>(false);
}

TEST_CASE("gemm_batched_tn.float" * doctest::timeout(300)) {
  gemm_batched_tn<float>(false);
}

TEST_CASE("gemm_batched_nt.float" * doctest::timeout(300)) {
  gemm_batched_nt<float>(false);
}

TEST_CASE("gemm_batched_tt.float" * doctest::timeout(300)) {
  gemm_batched_tt<float>(false);
}

TEST_CASE("gemm_batched_nn.double" * doctest::timeout(300)) {
  gemm_batched_nn<double>(false);
}

TEST_CASE("gemm_batched_tn.double" * doctest::timeout(300)) {
  gemm_batched_tn<double>(false);
}

TEST_CASE("gemm_batched_nt.double" * doctest::timeout(300)) {
  gemm_batched_nt<double>(false);
}

TEST_CASE("gemm_batched_tt.double" * doctest::timeout(300)) {
  gemm_batched_tt<double>(false);
}
    
// c_gemm_batched (row-major)
TEST_CASE("c_gemm_batched_nn.float" * doctest::timeout(300)) {
  gemm_batched_nn<float>(true);
}

TEST_CASE("c_gemm_batched_tn.float" * doctest::timeout(300)) {
  gemm_batched_tn<float>(true);
}

TEST_CASE("c_gemm_batched_nt.float" * doctest::timeout(300)) {
  gemm_batched_nt<float>(true);
}

TEST_CASE("c_gemm_batched_tt.float" * doctest::timeout(300)) {
  gemm_batched_tt<float>(true);
}

TEST_CASE("c_gemm_batched_nn.double" * doctest::timeout(300)) {
  gemm_batched_nn<double>(true);
}

TEST_CASE("c_gemm_batched_tn.double" * doctest::timeout(300)) {
  gemm_batched_tn<double>(true);
}

TEST_CASE("c_gemm_batched_nt.double" * doctest::timeout(300)) {
  gemm_batched_nt<double>(true);
}

TEST_CASE("c_gemm_batched_tt.double" * doctest::timeout(300)) {
  gemm_batched_tt<double>(true);
}

// ----------------------------------------------------------------------------
// Testcase: gemm_strided_batched
// ----------------------------------------------------------------------------

template <typename T>
void gemm_strided_batched(
  bool row_major,
  const int M, 
  const int N, 
  const int K,
  const T* hA,
  const T* hB,
  const std::vector<T>& golden,
  bool tranA,
  bool tranB
) {
  
  tf::Taskflow taskflow;
  tf::Executor executor;

  int d = 0;
  
  auto dA = tf::cuda_malloc_device<T>(S*K*M, d);
  auto dB = tf::cuda_malloc_device<T>(S*K*N, d);
  auto dC = tf::cuda_malloc_device<T>(S*M*N, d);
  auto dAlpha = tf::cuda_malloc_device<T>(1, d);
  auto dBeta  = tf::cuda_malloc_device<T>(1, d);
  auto hC = new T[S*M*N];

  int sA = K*M;
  int sB = K*N;
  int sC = M*N;

  auto cudaflow = taskflow.emplace([&](tf::cudaFlow& cf){

    auto copyA = cf.copy(dA, hA, S*K*M);
    auto copyB = cf.copy(dB, hB, S*K*N);

    auto alpha = cf.single_task([=] __device__ () { *dAlpha = 1; });
    auto beta  = cf.single_task([=] __device__ () { *dBeta  = 0; });

    tf::cudaTask gemm; 
    
    if(!tranA && !tranB) {  // C = A * B (r-major)
      if (row_major) {
        gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
          flow.make_capturer<tf::cublasFlowCapturer>()->c_gemm_sbatched(
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, dAlpha, dA, K, sA, dB, N, sB, dBeta, dC, N, sC, S
          );
        });
      }
      else {
        gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
          flow.make_capturer<tf::cublasFlowCapturer>()->gemm_sbatched(
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, dAlpha, dB, N, sB, dA, K, sA, dBeta, dC, N, sC, S
          );
        });
      }
    }
    else if(tranA && !tranB) {        // C = A^T * B (r-major)
      if (row_major) {
        gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
          flow.make_capturer<tf::cublasFlowCapturer>()->c_gemm_sbatched(
            CUBLAS_OP_T, CUBLAS_OP_N,
            M, N, K, dAlpha, dA, M, sA, dB, N, sB, dBeta, dC, N, sC, S
          );
        });
      }
      else {
        gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
          flow.make_capturer<tf::cublasFlowCapturer>()->gemm_sbatched(
            CUBLAS_OP_N, CUBLAS_OP_T,
            N, M, K, dAlpha, dB, N, sB, dA, M, sA, dBeta, dC, N, sC, S
          );
        });
      }
    }
    else if(!tranA && tranB) {   // C = A * B^T (r-major)
      if(row_major) {
        gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
          flow.make_capturer<tf::cublasFlowCapturer>()->c_gemm_sbatched(
            CUBLAS_OP_N, CUBLAS_OP_T,
            M, N, K, dAlpha, dA, K, sA, dB, K, sB, dBeta, dC, N, sC, S
          );
        });
      }
      else {
        gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
          flow.make_capturer<tf::cublasFlowCapturer>()->gemm_sbatched(
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K, dAlpha, dB, K, sB, dA, K, sA, dBeta, dC, N, sC, S
          );
        });
      }
    }
    else {                       // C = A^T * B^T (r-major)
      if (row_major) {
        gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
          flow.make_capturer<tf::cublasFlowCapturer>()->c_gemm_sbatched(
            CUBLAS_OP_T, CUBLAS_OP_T,
            M, N, K, dAlpha, dA, M, sA, dB, K, sB, dBeta, dC, N, sC, S
          );
        });
      }
      else {
        gemm = cf.capture([&](tf::cudaFlowCapturer& flow){
          flow.make_capturer<tf::cublasFlowCapturer>()->gemm_sbatched(
            CUBLAS_OP_T, CUBLAS_OP_T,
            N, M, K, dAlpha, dB, K, sB, dA, M, sA, dBeta, dC, N, sC, S
          );
        });
      }
    }
      
    auto copyC = cf.copy(hC, dC, S*M*N);
    
    gemm.succeed(alpha, beta, copyA, copyB)
        .precede(copyC);
  });

  auto verify = taskflow.emplace([&](){
    for(size_t s=0; s<S; s++) {
      auto p = hC + s*M*N;
      for(size_t i=0; i<golden.size(); i++) {
        REQUIRE(std::fabs(p[i]-golden[i]) < 0.0001);
      }
    }
    tf::cuda_free(dA);
    tf::cuda_free(dB);
    tf::cuda_free(dC);
    tf::cuda_free(dAlpha);
    tf::cuda_free(dBeta);
    delete [] hC;
  });
  
  cudaflow.precede(verify);

  executor.run(taskflow).wait();
}

// C = A * B
template <typename T>
void gemm_strided_batched_nn(bool row_major) {

  const int N = 4, M = 2, K = 3;

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

  std::vector<T> hAs, hBs;

  for(size_t s=0; s<S; s++) {
    for(auto a : hA) hAs.push_back(a);
    for(auto b : hB) hBs.push_back(b);
  }

  gemm_strided_batched<T>(
    row_major, M, N, K, hAs.data(), hBs.data(), golden, false, false
  );
}

// C = A^T * B
template <typename T>
void gemm_strided_batched_tn(bool row_major) {

  const int N = 4, M = 2, K = 3;

  const std::vector<T> hA = {
    11, 14,
    12, 15,
    13, 16
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

  std::vector<T> hAs, hBs;

  for(size_t s=0; s<S; s++) {
    for(auto a : hA) hAs.push_back(a);
    for(auto b : hB) hBs.push_back(b);
  }

  gemm_strided_batched<T>(
    row_major, M, N, K, hAs.data(), hBs.data(), golden, true, false
  );
}

// C = A * B^T
template <typename T>
void gemm_strided_batched_nt(bool row_major) {

  const int N = 4, M = 2, K = 3;

  const std::vector<T> hA = {
    11, 12, 13, 
    14, 15, 16
  };

  const std::vector<T> hB = {
    11, 15, 19,
    12, 16, 20,
    13, 17, 21,
    14, 18, 22
  };

  const std::vector<T> golden = {
    548, 584, 620, 656,
    683, 728, 773, 818 
  }; //  MxN

  std::vector<T> hAs, hBs;

  for(size_t s=0; s<S; s++) {
    for(auto a : hA) hAs.push_back(a);
    for(auto b : hB) hBs.push_back(b);
  }

  gemm_strided_batched<T>(
    row_major, M, N, K, hAs.data(), hBs.data(), golden, false, true
  );
}

// C = A^T * B^T
template <typename T>
void gemm_strided_batched_tt(bool row_major) {

  const int N = 4, M = 2, K = 3;

  const std::vector<T> hA = {
    11, 14,
    12, 15,
    13, 16
  };

  const std::vector<T> hB = {
    11, 15, 19,
    12, 16, 20,
    13, 17, 21,
    14, 18, 22
  };

  const std::vector<T> golden = {
    548, 584, 620, 656,
    683, 728, 773, 818 
  }; //  MxN

  std::vector<T> hAs, hBs;

  for(size_t s=0; s<S; s++) {
    for(auto a : hA) hAs.push_back(a);
    for(auto b : hB) hBs.push_back(b);
  }

  gemm_strided_batched<T>(
    row_major, M, N, K, hAs.data(), hBs.data(), golden, true, true
  );
}

// gemm_strided_batched (column-major)
TEST_CASE("gemm_strided_batched_nn.float" * doctest::timeout(300)) {
  gemm_strided_batched_nn<float>(false);
}

TEST_CASE("gemm_strided_batched_tn.float" * doctest::timeout(300)) {
  gemm_strided_batched_tn<float>(false);
}

TEST_CASE("gemm_strided_batched_nt.float" * doctest::timeout(300)) {
  gemm_strided_batched_nt<float>(false);
}

TEST_CASE("gemm_strided_batched_tt.float" * doctest::timeout(300)) {
  gemm_strided_batched_tt<float>(false);
}

TEST_CASE("gemm_strided_batched_nn.double" * doctest::timeout(300)) {
  gemm_strided_batched_nn<double>(false);
}

TEST_CASE("gemm_strided_batched_tn.double" * doctest::timeout(300)) {
  gemm_strided_batched_tn<double>(false);
}

TEST_CASE("gemm_strided_batched_nt.double" * doctest::timeout(300)) {
  gemm_strided_batched_nt<double>(false);
}

TEST_CASE("gemm_strided_batched_tt.double" * doctest::timeout(300)) {
  gemm_strided_batched_tt<double>(false);
}

// gemm_strided_batched (row-major)
TEST_CASE("c_gemm_strided_batched_nn.float" * doctest::timeout(300)) {
  gemm_strided_batched_nn<float>(true);
}

TEST_CASE("c_gemm_strided_batched_tn.float" * doctest::timeout(300)) {
  gemm_strided_batched_tn<float>(true);
}

TEST_CASE("c_gemm_strided_batched_nt.float" * doctest::timeout(300)) {
  gemm_strided_batched_nt<float>(true);
}

TEST_CASE("c_gemm_strided_batched_tt.float" * doctest::timeout(300)) {
  gemm_strided_batched_tt<float>(true);
}

TEST_CASE("c_gemm_strided_batched_nn.double" * doctest::timeout(300)) {
  gemm_strided_batched_nn<double>(true);
}

TEST_CASE("c_gemm_strided_batched_tn.double" * doctest::timeout(300)) {
  gemm_strided_batched_tn<double>(true);
}

TEST_CASE("c_gemm_strided_batched_nt.double" * doctest::timeout(300)) {
  gemm_strided_batched_nt<double>(true);
}

TEST_CASE("c_gemm_strided_batched_tt.double" * doctest::timeout(300)) {
  gemm_strided_batched_tt<double>(true);
} 

// ----------------------------------------------------------------------------
// symm
// ----------------------------------------------------------------------------

template <typename T>
void symm_test() {

  int M = 3;
  int N = 4;
  int LA = 6, LB = 6, LC = N;

  const std::vector<T> hA = {
   -1, -1, -1, -1, -1, -1,
   -1,  2,  0,  0, -1, -1,
   -1,  1,  2,  0, -1, -1,
   -1,  1,  1,  2, -1, -1
  };
  
  const std::vector<T> hB = {
   -1, -1, -1, -1, -1, -1,
   -1,  1,  1,  3,  1, -1,
   -1,  1,  4,  1,  1, -1,
   -1,  1,  1,  7,  1, -1
  };

  const std::vector<T> gold = {
    4, 7,  14, 4, 
    4, 10, 12, 4, 
    4, 7,  18, 4 
  };

  std::vector<T> hC(M*N);
  
  tf::Taskflow taskflow;
  tf::Executor executor;

  auto dA = tf::cuda_malloc_device<T>(hA.size());
  auto dB = tf::cuda_malloc_device<T>(hB.size());
  auto dC = tf::cuda_malloc_device<T>(hC.size());
  auto dalpha = tf::cuda_malloc_device<T>(1);
  auto dbeta  = tf::cuda_malloc_device<T>(1);

  taskflow.emplace([&](tf::cudaFlowCapturer& capturer){
    auto blas = capturer.make_capturer<tf::cublasFlowCapturer>();
    auto alpha = capturer.single_task([=] __device__ () { *dalpha = 1; });
    auto beta  = capturer.single_task([=] __device__ () { *dbeta = 0; });
    auto h2dA  = capturer.copy(dA, hA.data(), hA.size());
    auto h2dB  = capturer.copy(dB, hB.data(), hB.size());
    auto symm  = blas->c_symm(
      CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, 
      M, N, dalpha, dA + 7, LA, dB + 7, LB, dbeta, dC, LC
    );
    auto d2hC = capturer.copy(hC.data(), dC, hC.size());

    symm.succeed(h2dA, h2dB, alpha, beta)
        .precede(d2hC);
  });

  executor.run(taskflow).wait();

  for(size_t i=0; i<hC.size(); i++) {
    REQUIRE(std::fabs(hC[i] - gold[i]) < 0.0001);
  }

}

TEST_CASE("c_symm.float" * doctest::timeout(300)) {
  symm_test<float>();
}

TEST_CASE("c_symm.double" * doctest::timeout(300)) {
  symm_test<double>();
}
  
// ----------------------------------------------------------------------------
// syrk
// ----------------------------------------------------------------------------

template <typename T>
void syrk_test() {

  int N = 3;
  int K = 4;
  int LA = 6, LC = 6;

  std::vector<T> hC = {
   -1, -1, -1, -1, -1, -1,
   -1,  2,  0,  0, -1, -1,
   -1,  1,  2,  0, -1, -1,
   -1,  1,  1,  2, -1, -1
  };
  
  const std::vector<T> hA = {
   -1, -1, -1, -1, -1, -1,
   -1,  1,  1,  3,  1, -1,
   -1,  1,  4,  1,  1, -1,
   -1,  1,  1,  7,  1, -1
  };

  const std::vector<T> gold = {
   -1, -1, -1, -1, -1, -1,
   -1, 14,  0,  0, -1, -1,
   -1, 10, 21,  0, -1, -1,
   -1, 25, 14, 54, -1, -1
  };

  tf::Taskflow taskflow;
  tf::Executor executor;

  auto dA = tf::cuda_malloc_device<T>(hA.size());
  auto dC = tf::cuda_malloc_device<T>(hC.size());
  auto dalpha = tf::cuda_malloc_device<T>(1);
  auto dbeta  = tf::cuda_malloc_device<T>(1);

  taskflow.emplace([&](tf::cudaFlowCapturer& capturer){
    auto blas = capturer.make_capturer<tf::cublasFlowCapturer>();
    auto alpha = capturer.single_task([=] __device__ () { *dalpha = 1; });
    auto beta  = capturer.single_task([=] __device__ () { *dbeta = 1; });
    auto h2dA  = capturer.copy(dA, hA.data(), hA.size());
    auto h2dC  = capturer.copy(dC, hC.data(), hC.size());
    auto syrk  = blas->c_syrk(
      CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
      N, K, dalpha, dA + 7, LA, dbeta, dC + 7, LC
    );
    auto d2hC = capturer.copy(hC.data(), dC, hC.size());

    syrk.succeed(h2dA, h2dC, alpha, beta)
        .precede(d2hC);
  });

  executor.run(taskflow).wait();

  //print_matrix(4, 6, hC);

  for(size_t i=0; i<hC.size(); i++) {
    REQUIRE(std::fabs(hC[i] - gold[i]) < 0.0001);
  }

}

TEST_CASE("c_syrk.float" * doctest::timeout(300)) {
  syrk_test<float>();
}

TEST_CASE("c_syrk.double" * doctest::timeout(300)) {
  syrk_test<double>();
}
  
// ----------------------------------------------------------------------------
// syr2k
// ----------------------------------------------------------------------------

template <typename T>
void syr2k_test() {

  int N = 3;
  int K = 4;
  int LA = 6, LC = 6, LB = 6;

  std::vector<T> hC = {
   -1, -1, -1, -1, -1, -1,
   -1,  2,  0,  0, -1, -1,
   -1,  1,  2,  0, -1, -1,
   -1,  1,  1,  2, -1, -1
  };
  
  const std::vector<T> hA = {
   -1, -1, -1, -1, -1, -1,
   -1,  1,  1,  3,  1, -1,
   -1,  1,  4,  1,  1, -1,
   -1,  1,  1,  7,  1, -1
  };
  
  const std::vector<T> hB = {
   -1, -1, -1, -1, -1, -1,
   -1,  1, 10,  2,  9, -1,
   -1,  8, 14,  2,  1, -1,
   -1, 13,  3,  1,  4, -1
  };

  const std::vector<T> gold = {
   -1, -1, -1, -1, -1, -1,
   -1, 54,  0,  0, -1, -1,
   -1, 82, 136, 0, -1, -1,
   -1, 58, 68, 56, -1, -1
  };

  tf::Taskflow taskflow;
  tf::Executor executor;

  auto dA = tf::cuda_malloc_device<T>(hA.size());
  auto dB = tf::cuda_malloc_device<T>(hB.size());
  auto dC = tf::cuda_malloc_device<T>(hC.size());
  auto dalpha = tf::cuda_malloc_device<T>(1);
  auto dbeta  = tf::cuda_malloc_device<T>(1);

  taskflow.emplace([&](tf::cudaFlowCapturer& capturer){
    auto blas = capturer.make_capturer<tf::cublasFlowCapturer>();
    auto alpha = capturer.single_task([=] __device__ () { *dalpha = 1; });
    auto beta  = capturer.single_task([=] __device__ () { *dbeta = 1; });
    auto h2dA  = capturer.copy(dA, hA.data(), hA.size());
    auto h2dB  = capturer.copy(dB, hB.data(), hB.size());
    auto h2dC  = capturer.copy(dC, hC.data(), hC.size());
    auto syr2k  = blas->c_syr2k(
      CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
      N, K, dalpha, dA + 7, LA, dB + 7, LB, dbeta, dC + 7, LC
    );
    auto d2hC = capturer.copy(hC.data(), dC, hC.size());

    syr2k.succeed(h2dA, h2dC, h2dB, alpha, beta)
         .precede(d2hC);
  });

  executor.run(taskflow).wait();

  //print_matrix(4, 6, hC);

  for(size_t i=0; i<hC.size(); i++) {
    REQUIRE(std::fabs(hC[i] - gold[i]) < 0.0001);
  }

}

TEST_CASE("c_syr2k.float" * doctest::timeout(300)) {
  syr2k_test<float>();
}

TEST_CASE("c_syr2k.double" * doctest::timeout(300)) {
  syr2k_test<double>();
}

// ----------------------------------------------------------------------------
// trmm
// ----------------------------------------------------------------------------

template <typename T>
void trmm_test() {

  int N = 4;
  int M = 3;
  int LA = 6, LC = 6, LB = 6;

  std::vector<T> hC = {
   -1,  1,  1,  1,  1, -1,
   -1,  2,  0,  0, -1, -1,
   -1,  1,  2,  0, -1, -1,
   -1,  1,  1,  2, -1, -1
  };
  
  const std::vector<T> hA = {
   -1, -1, -1, -1, -1, -1,
   -1,  1,  0,  0, -1, -1,
   -1,  1,  4,  0, -1, -1,
   -1,  1,  1,  7, -1, -1
  };
  
  const std::vector<T> hB = {
   -1, -1, -1, -1, -1, -1,
   -1,  1, 10,  2,  9, -1,
   -1,  8, 14,  2,  1, -1,
   -1, 13,  3,  1,  4, -1
  };

  const std::vector<T> gold = {
    -1, -1, -1, -1, -1, -1,
    -1, 1, 10, 2, 9, -1,
    -1, 33, 66, 10, 13, -1,
    -1, 100, 45, 11, 38, -1
  };

  tf::Taskflow taskflow;
  tf::Executor executor;

  auto dA = tf::cuda_malloc_device<T>(hA.size());
  auto dB = tf::cuda_malloc_device<T>(hB.size());
  auto dC = tf::cuda_malloc_device<T>(hC.size());
  auto dalpha = tf::cuda_malloc_device<T>(1);

  taskflow.emplace([&](tf::cudaFlowCapturer& capturer){
    auto blas = capturer.make_capturer<tf::cublasFlowCapturer>();
    auto alpha = capturer.single_task([=] __device__ () { *dalpha = 1; });
    auto h2dA  = capturer.copy(dA, hA.data(), hA.size());
    auto h2dB  = capturer.copy(dB, hB.data(), hB.size());
    auto setC  = capturer.for_each(dC, dC + hC.size(), 
      []__device__(T& v) { v = -1; });
    auto trmm  = blas->c_trmm(
      CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
      M, N, dalpha, dA + 7, LA, dB + 7, LB, dC + 7, LC
    );
    auto d2hC = capturer.copy(hC.data(), dC, hC.size());

    trmm.succeed(h2dA, h2dB, alpha, setC)
        .precede(d2hC);
  });

  executor.run(taskflow).wait();

  //print_matrix(4, 6, hC);

  for(size_t i=0; i<hC.size(); i++) {
    REQUIRE(std::fabs(hC[i] - gold[i]) < 0.0001);
  }
}

TEST_CASE("c_trmm.float" * doctest::timeout(300)) {
  trmm_test<float>();
}

TEST_CASE("c_trmm.double" * doctest::timeout(300)) {
  trmm_test<double>();
}

// ----------------------------------------------------------------------------
// trsm
// ----------------------------------------------------------------------------

template <typename T>
void trsm_test() {

  int N = 2;
  int M = 3;
  int LA = 6;
  int LB = 2;

  const std::vector<T> hA = {
   -1, -1, -1, -1, -1, -1,
   -1,  2,  0,  0, -1, -1,
   -1,  1,  2,  0, -1, -1,
   -1,  1,  1,  2, -1, -1
  };

  std::vector<T> hB = {
    5, 10,
    4, 8,
    7, 14
  };

  const std::vector<T> sol = {
    2.5, 5, 
    0.75, 1.5,
    1.875, 3.75
  };

  tf::Taskflow taskflow;
  tf::Executor executor;

  auto dA = tf::cuda_malloc_device<T>(hA.size());
  auto dB = tf::cuda_malloc_device<T>(hB.size());
  auto dAlpha = tf::cuda_malloc_device<T>(1);

  taskflow.emplace([&](tf::cudaFlowCapturer& capturer){
    auto blas = capturer.make_capturer<tf::cublasFlowCapturer>();
    auto alpha = capturer.single_task([=] __device__ () { *dAlpha = 1; });
    auto h2dA = capturer.copy(dA, hA.data(), hA.size());
    auto h2dB = capturer.copy(dB, hB.data(), hB.size());
    auto trsm = blas->c_trsm(
      CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, 
      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
      M, N, dAlpha, dA + 7, LA, dB, LB
    );
    auto d2h = capturer.copy(hB.data(), dB, hB.size());

    trsm.succeed(h2dA, h2dB, alpha)
        .precede(d2h);
  });

  executor.run(taskflow).wait();

  //print_matrix(3, 2, hB);
  
  for(size_t i=0; i<hB.size(); ++i) {
    //std::cout << res[i] << '\n';
    REQUIRE(std::fabs(hB[i] - sol[i]) < 0.0001);
  }

}

TEST_CASE("c_trsm.float" * doctest::timeout(300)) {
  trsm_test<float>();
}

TEST_CASE("c_trsm.double" * doctest::timeout(300)) {
  trsm_test<double>();
}
  
