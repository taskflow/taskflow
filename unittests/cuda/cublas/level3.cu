#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cublas.hpp>


// --------------------------------------------------------
// Testcase: gemm
// --------------------------------------------------------

template <typename T>
void gemm() {

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
  };
  
  tf::Taskflow taskflow;
  tf::Executor executor;

  for(size_t d=0; d<tf::cuda_get_num_devices(); d++) {

    auto dA = tf::cuda_malloc_device<T>(M*K, d);
    auto dB = tf::cuda_malloc_device<T>(K*N, d);
    auto dC = tf::cuda_malloc_device<T>(M*N, d);
    auto dAlpha = tf::cuda_malloc_shared<T>(1);
    auto dBeta  = tf::cuda_malloc_shared<T>(1);

    *dAlpha = 1;
    *dBeta  = 0;
  
    T* hC = new T[N*M];

    auto cudaflow = taskflow.emplace_on([=, &hA, &hB](tf::cudaFlow& cf){

      REQUIRE(tf::cuda_get_device() == d);
      
      auto copyA = cf.copy(dA, hA.data(), M*K);
      auto copyB = cf.copy(dB, hB.data(), K*N);

      auto gemm  = cf.childflow([&](tf::cublasFlow& flow){
        flow.gemm(
          CUBLAS_OP_N, CUBLAS_OP_N,
          N, M, K,
          dAlpha,
          dB, N, dA, K, 
          dBeta,
          dC, N
        );
      });
      
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

TEST_CASE("gemm.float") {
  gemm<float>();
}

TEST_CASE("gemm.double") {
  gemm<double>();
}

