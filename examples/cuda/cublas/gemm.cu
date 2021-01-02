// This program performs general matrix multiplication on row-major layout
// using tf::cublasFlowCapturer::c_gemm.

#include <taskflow/taskflow.hpp>
#include <taskflow/cudaflow.hpp>
#include <taskflow/cublasflow.hpp>

int main() {

  const int M = 2, N = 4, K = 3;

  const std::vector<float> hA = {
    11, 12, 13, 
    14, 15, 16
  };  // M x K

  const std::vector<float> hB = {
    11, 12, 13, 14,
    15, 16, 17, 18,
    19, 20, 21, 22
  };  // K x N

  const std::vector<float> golden = {
    548, 584, 620, 656,
    683, 728, 773, 818 
  };  //  M x N

  std::vector<float> hC(M*N);
    
  //auto dA = tf::cuda_malloc_device<float>(hA.size());
  //auto dB = tf::cuda_malloc_device<float>(hB.size());
  //auto dC = tf::cuda_malloc_device<float>(hC.size());
  //auto dAlpha = tf::cuda_malloc_device<float>(1);
  //auto dBeta  = tf::cuda_malloc_device<float>(1);
  float *dA, *dB, *dC, *dAlpha, *dBeta;

  tf::Taskflow taskflow("Matrix Multiplication");
  tf::Executor executor;

  auto malloc_dA = taskflow.emplace(
    [&](){ dA = tf::cuda_malloc_device<float>(hA.size()); }
  ).name("malloc_dA");
  
  auto malloc_dB = taskflow.emplace(
    [&](){ dB = tf::cuda_malloc_device<float>(hB.size()); }
  ).name("malloc_dB");
  
  auto malloc_dC = taskflow.emplace(
    [&](){ dC = tf::cuda_malloc_device<float>(hC.size()); }
  ).name("malloc_dC");
  
  auto malloc_dAlpha = taskflow.emplace(
    [&](){ dAlpha = tf::cuda_malloc_device<float>(1); }
  ).name("malloc_dAlpha");
  
  auto malloc_dBeta = taskflow.emplace(
    [&](){ dBeta = tf::cuda_malloc_device<float>(1); }
  ).name("malloc_dBeta");

  auto cublasFlow = taskflow.emplace([&](tf::cudaFlowCapturer& capturer) {
    auto blas  = capturer.make_capturer<tf::cublasFlowCapturer>();

    auto alpha = capturer.single_task([=] __device__ () { *dAlpha = 1; })
                         .name("alpha=1");
    auto beta  = capturer.single_task([=] __device__ () { *dBeta  = 0; })
                         .name("beta=0");
    auto copyA = capturer.copy(dA, hA.data(), hA.size()).name("copyA"); 
    auto copyB = capturer.copy(dB, hB.data(), hB.size()).name("copyB");
    auto gemm  = blas->c_gemm(CUBLAS_OP_N, CUBLAS_OP_N,
      M, N, K, dAlpha, dA, K, dB, N, dBeta, dC, N
    ).name("C = alpha * A * B + beta * C");
    auto copyC = capturer.copy(hC.data(), dC, hC.size()).name("copyC");

    gemm.succeed(alpha, beta, copyA, copyB)
        .precede(copyC);

    capturer.dump(std::cout);  // dump the graph constructed so far.
  }).name("cublasFlow");

  cublasFlow.succeed(
    malloc_dA, malloc_dB, malloc_dC, malloc_dAlpha, malloc_dBeta
  );

  executor.run(taskflow).wait();

  taskflow.dump(std::cout);
  
  std::cout << "Matrix C:\n";
  for(int m=0; m<M; m++) {
    for(int n=0; n<N; n++) {
      std::cout << hC[m*N+n] << ' ';
    }
    std::cout << '\n';
  }

  return 0;
}







