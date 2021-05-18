#include <taskflow/taskflow.hpp>
#include <taskflow/cudaflow.hpp>
#include <taskflow/cublasflow.hpp>

int main() {

  const int N = 1024;
  
  tf::Executor executor;
  tf::Taskflow taskflow("2-norm");

  std::vector<float> hvec(N, 1);
  float  hres;
  float* gvec = tf::cuda_malloc_device<float>(N);
  float* gres = tf::cuda_malloc_device<float>(1);

  taskflow.emplace([&](tf::cudaFlowCapturer& capturer){
    
    auto blas = capturer.make_capturer<tf::cublasFlowCapturer>();

    tf::cudaTask h2d = capturer.copy(gvec, hvec.data(), N).name("h2d");
    tf::cudaTask nrm = blas->nrm2(N, gvec, 1, gres).name("2-norm");
    tf::cudaTask d2h = capturer.copy(&hres, gres, 1).name("d2h");

    nrm.precede(d2h)
       .succeed(h2d);

  }).name("capturer");

  executor.run(taskflow).wait();

  taskflow.dump(std::cout);

  std::cout << "2-norm of an unity vector of 1024 elements is: " 
            << hres << '\n';  // 32

  return 0;
}

