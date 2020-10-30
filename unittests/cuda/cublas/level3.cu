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
          geam = cf.cublas([&](tf::cublasFlow& flow){
            flow.c_geam(CUBLAS_OP_T, CUBLAS_OP_N,
              M, N, dAlpha, dA, M, dBeta, dB, N, dC, N
            );
          });
        }
        else {
          geam = cf.cublas([&](tf::cublasFlow& flow){
            flow.geam(CUBLAS_OP_T, CUBLAS_OP_N,
              N, M, dAlpha, dA, M, dBeta, dB, N, dC, N
            );
          });
        }
      }
      else if(!tranA && !tranB) {  // C = A + B (r-major)
        if (row_major) {
          geam = cf.cublas([&](tf::cublasFlow& flow){
            flow.c_geam(CUBLAS_OP_N, CUBLAS_OP_N,
              M, N, dAlpha, dA, N, dBeta, dB, N, dC, N
            );
          });
        }
        else {
          geam = cf.cublas([&](tf::cublasFlow& flow){
            flow.geam(CUBLAS_OP_N, CUBLAS_OP_N,
              N, M, dAlpha, dA, N, dBeta, dB, N, dC, N
            );
          });
        }
      }
      else if(!tranA && tranB) {   // C = A + B^T (r-major)
        if(row_major) {
          geam = cf.cublas([&](tf::cublasFlow& flow){
            flow.c_geam(CUBLAS_OP_N, CUBLAS_OP_T,
              M, N, dAlpha, dA, N, dBeta, dB, M, dC, N
            );
          });
        }
        else {
          geam = cf.cublas([&](tf::cublasFlow& flow){
            flow.geam(CUBLAS_OP_N, CUBLAS_OP_T,
              N, M, dAlpha, dA, N, dBeta, dB, M, dC, N
            );
          });
        }
      }
      else {                       // C = A^T * B^T (r-major)
        if (row_major) {
          geam = cf.cublas([&](tf::cublasFlow& flow){
            flow.c_geam(CUBLAS_OP_T, CUBLAS_OP_T,
              M, N, dAlpha, dA, M, dBeta, dB, M, dC, N
            );
          });
        }
        else {
          geam = cf.cublas([&](tf::cublasFlow& flow){
            flow.geam(CUBLAS_OP_T, CUBLAS_OP_T,
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
TEST_CASE("geam_tn.float") {
  geam_tn<float>(false);
}

TEST_CASE("geam_nn.float") {
  geam_nn<float>(false);
}

TEST_CASE("geam_nt.float") {
  geam_nt<float>(false);
}

TEST_CASE("geam_tt.float") {
  geam_tt<float>(false);
}

TEST_CASE("geam_tn.double") {
  geam_tn<double>(false);
}

TEST_CASE("geam_nn.double") {
  geam_nn<double>(false);
}

TEST_CASE("geam_nt.double") {
  geam_nt<double>(false);
}

TEST_CASE("geam_tt.double") {
  geam_tt<double>(false);
}

// row major
TEST_CASE("c_geam_tn.float") {
  geam_tn<float>(true);
}

TEST_CASE("c_geam_nn.float") {
  geam_nn<float>(true);
}

TEST_CASE("c_geam_nt.float") {
  geam_nt<float>(true);
}

TEST_CASE("c_geam_tt.float") {
  geam_tt<float>(true);
}

TEST_CASE("c_geam_tn.double") {
  geam_tn<double>(true);
}

TEST_CASE("c_geam_nn.double") {
  geam_nn<double>(true);
}

TEST_CASE("c_geam_nt.double") {
  geam_nt<double>(true);
}

TEST_CASE("c_geam_tt.double") {
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
          gemm = cf.cublas([&](tf::cublasFlow& flow){
            flow.c_gemm(
              CUBLAS_OP_T, CUBLAS_OP_N,
              M, N, K, dAlpha, dA, M, dB, N, dBeta, dC, N
            );
          });
        }
        else {
          gemm = cf.cublas([&](tf::cublasFlow& flow){
            flow.gemm(
              CUBLAS_OP_N, CUBLAS_OP_T,
              N, M, K, dAlpha, dB, N, dA, M, dBeta, dC, N
            );
          });
        }
      }
      else if(!tranA && !tranB) {  // C = A * B (r-major)
        if (row_major) {
          gemm = cf.cublas([&](tf::cublasFlow& flow){
            flow.c_gemm(
              CUBLAS_OP_N, CUBLAS_OP_N,
              M, N, K, dAlpha, dA, K, dB, N, dBeta, dC, N
            );
          });
        }
        else {
          gemm = cf.cublas([&](tf::cublasFlow& flow){
            flow.gemm(
              CUBLAS_OP_N, CUBLAS_OP_N,
              N, M, K, dAlpha, dB, N, dA, K, dBeta, dC, N
            );
          });
        }
      }
      else if(!tranA && tranB) {   // C = A * B^T (r-major)
        if(row_major) {
          gemm = cf.cublas([&](tf::cublasFlow& flow){
            flow.c_gemm(
              CUBLAS_OP_N, CUBLAS_OP_T,
              M, N, K, dAlpha, dA, K, dB, K, dBeta, dC, N
            );
          });
        }
        else {
          gemm = cf.cublas([&](tf::cublasFlow& flow){
            flow.gemm(
              CUBLAS_OP_T, CUBLAS_OP_N,
              N, M, K, dAlpha, dB, K, dA, K, dBeta, dC, N
            );
          });
        }
      }
      else {                       // C = A^T * B^T (r-major)
        if (row_major) {
          gemm = cf.cublas([&](tf::cublasFlow& flow){
            flow.c_gemm(
              CUBLAS_OP_T, CUBLAS_OP_T,
              M, N, K, dAlpha, dA, M, dB, K, dBeta, dC, N
            );
          });
        }
        else {
          gemm = cf.cublas([&](tf::cublasFlow& flow){
            flow.gemm(
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
TEST_CASE("gemm_nn.float") {
  gemm_nn<float>(false);
}

TEST_CASE("gemm_nn.double") {
  gemm_nn<double>(false);
}

TEST_CASE("gemm_tn.float") {
  gemm_tn<float>(false);
}

TEST_CASE("gemm_tn.double") {
  gemm_tn<double>(false);
}

TEST_CASE("gemm_nt.float") {
  gemm_nt<float>(false);
}

TEST_CASE("gemm_nt.double") {
  gemm_nt<double>(false);
}

TEST_CASE("gemm_tt.float") {
  gemm_tt<float>(false);
}

TEST_CASE("gemm_tt.double") {
  gemm_tt<double>(false);
}

// c_gemm (row_major)
TEST_CASE("c_gemm_nn.float") {
  gemm_nn<float>(true);
}

TEST_CASE("c_gemm_nn.double") {
  gemm_nn<double>(true);
}

TEST_CASE("c_gemm_tn.float") {
  gemm_tn<float>(true);
}

TEST_CASE("c_gemm_tn.double") {
  gemm_tn<double>(true);
}

TEST_CASE("c_gemm_nt.float") {
  gemm_nt<float>(true);
}

TEST_CASE("c_gemm_nt.double") {
  gemm_nt<double>(true);
}

TEST_CASE("c_gemm_tt.float") {
  gemm_tt<float>(true);
}

TEST_CASE("c_gemm_tt.double") {
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
        gemm = cf.cublas([&](tf::cublasFlow& flow){
          flow.c_gemm_batched(CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, dAlpha, (const T**)dAs, K, (const T**)dBs, N, dBeta, dCs, N, S
          );
        });
      }
      else {
        gemm = cf.cublas([&](tf::cublasFlow& flow){
          flow.gemm_batched(CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, dAlpha, (const T**)dBs, N, (const T**)dAs, K, dBeta, dCs, N, S
          );
        });
      }
    }
    else if(tranA && !tranB) {        // C = A^T * B (r-major)
      if (row_major) {
        gemm = cf.cublas([&](tf::cublasFlow& flow){
          flow.c_gemm_batched(CUBLAS_OP_T, CUBLAS_OP_N,
            M, N, K, dAlpha, (const T**)dAs, M, (const T**)dBs, N, dBeta, dCs, N, S
          );
        });
      }
      else {
        gemm = cf.cublas([&](tf::cublasFlow& flow){
          flow.gemm_batched(CUBLAS_OP_N, CUBLAS_OP_T,
            N, M, K, dAlpha, (const T**)dBs, N, (const T**)dAs, M, dBeta, dCs, N, S
          );
        });
      }
    }
    else if(!tranA && tranB) {   // C = A * B^T (r-major)
      if(row_major) {
        gemm = cf.cublas([&](tf::cublasFlow& flow){
          flow.c_gemm_batched(CUBLAS_OP_N, CUBLAS_OP_T,
            M, N, K, dAlpha, (const T**)dAs, K, (const T**)dBs, K, dBeta, dCs, N, S
          );
        });
      }
      else {
        gemm = cf.cublas([&](tf::cublasFlow& flow){
          flow.gemm_batched(CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K, dAlpha, (const T**)dBs, K, (const T**)dAs, K, dBeta, dCs, N, S
          );
        });
      }
    }
    else {                       // C = A^T * B^T (r-major)
      if (row_major) {
        gemm = cf.cublas([&](tf::cublasFlow& flow){
          flow.c_gemm_batched(CUBLAS_OP_T, CUBLAS_OP_T,
            M, N, K, dAlpha, (const T**)dAs, M, (const T**)dBs, K, dBeta, dCs, N, S
          );
        });
      }
      else {
        gemm = cf.cublas([&](tf::cublasFlow& flow){
          flow.gemm_batched(CUBLAS_OP_T, CUBLAS_OP_T,
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
TEST_CASE("gemm_batched_nn.float") {
  gemm_batched_nn<float>(false);
}

TEST_CASE("gemm_batched_tn.float") {
  gemm_batched_tn<float>(false);
}

TEST_CASE("gemm_batched_nt.float") {
  gemm_batched_nt<float>(false);
}

TEST_CASE("gemm_batched_tt.float") {
  gemm_batched_tt<float>(false);
}

TEST_CASE("gemm_batched_nn.double") {
  gemm_batched_nn<double>(false);
}

TEST_CASE("gemm_batched_tn.double") {
  gemm_batched_tn<double>(false);
}

TEST_CASE("gemm_batched_nt.double") {
  gemm_batched_nt<double>(false);
}

TEST_CASE("gemm_batched_tt.double") {
  gemm_batched_tt<double>(false);
}
    
// c_gemm_batched (row-major)
TEST_CASE("c_gemm_batched_nn.float") {
  gemm_batched_nn<float>(true);
}

TEST_CASE("c_gemm_batched_tn.float") {
  gemm_batched_tn<float>(true);
}

TEST_CASE("c_gemm_batched_nt.float") {
  gemm_batched_nt<float>(true);
}

TEST_CASE("c_gemm_batched_tt.float") {
  gemm_batched_tt<float>(true);
}

TEST_CASE("c_gemm_batched_nn.double") {
  gemm_batched_nn<double>(true);
}

TEST_CASE("c_gemm_batched_tn.double") {
  gemm_batched_tn<double>(true);
}

TEST_CASE("c_gemm_batched_nt.double") {
  gemm_batched_nt<double>(true);
}

TEST_CASE("c_gemm_batched_tt.double") {
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
        gemm = cf.cublas([&](tf::cublasFlow& flow){
          flow.c_gemm_sbatched(CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, dAlpha, dA, K, sA, dB, N, sB, dBeta, dC, N, sC, S
          );
        });
      }
      else {
        gemm = cf.cublas([&](tf::cublasFlow& flow){
          flow.gemm_sbatched(CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, dAlpha, dB, N, sB, dA, K, sA, dBeta, dC, N, sC, S
          );
        });
      }
    }
    else if(tranA && !tranB) {        // C = A^T * B (r-major)
      if (row_major) {
        gemm = cf.cublas([&](tf::cublasFlow& flow){
          flow.c_gemm_sbatched(CUBLAS_OP_T, CUBLAS_OP_N,
            M, N, K, dAlpha, dA, M, sA, dB, N, sB, dBeta, dC, N, sC, S
          );
        });
      }
      else {
        gemm = cf.cublas([&](tf::cublasFlow& flow){
          flow.gemm_sbatched(CUBLAS_OP_N, CUBLAS_OP_T,
            N, M, K, dAlpha, dB, N, sB, dA, M, sA, dBeta, dC, N, sC, S
          );
        });
      }
    }
    else if(!tranA && tranB) {   // C = A * B^T (r-major)
      if(row_major) {
        gemm = cf.cublas([&](tf::cublasFlow& flow){
          flow.c_gemm_sbatched(CUBLAS_OP_N, CUBLAS_OP_T,
            M, N, K, dAlpha, dA, K, sA, dB, K, sB, dBeta, dC, N, sC, S
          );
        });
      }
      else {
        gemm = cf.cublas([&](tf::cublasFlow& flow){
          flow.gemm_sbatched(CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K, dAlpha, dB, K, sB, dA, K, sA, dBeta, dC, N, sC, S
          );
        });
      }
    }
    else {                       // C = A^T * B^T (r-major)
      if (row_major) {
        gemm = cf.cublas([&](tf::cublasFlow& flow){
          flow.c_gemm_sbatched(CUBLAS_OP_T, CUBLAS_OP_T,
            M, N, K, dAlpha, dA, M, sA, dB, K, sB, dBeta, dC, N, sC, S
          );
        });
      }
      else {
        gemm = cf.cublas([&](tf::cublasFlow& flow){
          flow.gemm_sbatched(CUBLAS_OP_T, CUBLAS_OP_T,
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
TEST_CASE("gemm_strided_batched_nn.float") {
  gemm_strided_batched_nn<float>(false);
}

TEST_CASE("gemm_strided_batched_tn.float") {
  gemm_strided_batched_tn<float>(false);
}

TEST_CASE("gemm_strided_batched_nt.float") {
  gemm_strided_batched_nt<float>(false);
}

TEST_CASE("gemm_strided_batched_tt.float") {
  gemm_strided_batched_tt<float>(false);
}

TEST_CASE("gemm_strided_batched_nn.double") {
  gemm_strided_batched_nn<double>(false);
}

TEST_CASE("gemm_strided_batched_tn.double") {
  gemm_strided_batched_tn<double>(false);
}

TEST_CASE("gemm_strided_batched_nt.double") {
  gemm_strided_batched_nt<double>(false);
}

TEST_CASE("gemm_strided_batched_tt.double") {
  gemm_strided_batched_tt<double>(false);
}

// gemm_strided_batched (row-major)
TEST_CASE("c_gemm_strided_batched_nn.float") {
  gemm_strided_batched_nn<float>(true);
}

TEST_CASE("c_gemm_strided_batched_tn.float") {
  gemm_strided_batched_tn<float>(true);
}

TEST_CASE("c_gemm_strided_batched_nt.float") {
  gemm_strided_batched_nt<float>(true);
}

TEST_CASE("c_gemm_strided_batched_tt.float") {
  gemm_strided_batched_tt<float>(true);
}

TEST_CASE("c_gemm_strided_batched_nn.double") {
  gemm_strided_batched_nn<double>(true);
}

TEST_CASE("c_gemm_strided_batched_tn.double") {
  gemm_strided_batched_tn<double>(true);
}

TEST_CASE("c_gemm_strided_batched_nt.double") {
  gemm_strided_batched_nt<double>(true);
}

TEST_CASE("c_gemm_strided_batched_tt.double") {
  gemm_strided_batched_tt<double>(true);
}

