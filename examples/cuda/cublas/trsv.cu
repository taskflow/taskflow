#include <taskflow/taskflow.hpp>
#include <taskflow/cudaflow.hpp>
#include <taskflow/cublasflow.hpp>

int main() {

  int N = 3;

  // A x = b
  const std::vector<float> hA = {
    1, 0, 0, 
    1, 1, 0,
    1, 1, 1
  };

  const std::vector<float> hB = {
    5,
    4,
    7
  };

  std::vector<float> res(N, 0);

  tf::Taskflow taskflow("Ax = b");
  tf::Executor executor;

  auto dA = tf::cuda_malloc_device<float>(hA.size());
  auto dB = tf::cuda_malloc_device<float>(hB.size());

  taskflow.emplace([&](tf::cudaFlowCapturer& capturer){

    auto blas = capturer.make_capturer<tf::cublasFlowCapturer>();
    auto h2dA = capturer.copy(dA, hA.data(), hA.size()).name("copy A");
    auto h2dB = capturer.copy(dB, hB.data(), hB.size()).name("copy B");
    auto trsv = blas->c_trsv(
      CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, 
      N, dA, N, dB, 1
    ).name("trsv");
    auto d2h = capturer.copy(res.data(), dB, res.size()).name("copy result");

    trsv.succeed(h2dA, h2dB)
        .precede(d2h);
  }).name("cublasFlow");

  executor.run(taskflow).wait();
  taskflow.dump(std::cout);
  
  std::cout << "solution of the linear system: \n";
  for(size_t i=0; i<res.size(); ++i) {
    std::cout << "x" << i << ": " << res[i] << '\n';
  }

  return 0;
}

