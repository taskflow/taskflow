#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cublas.hpp>

template <typename T>
void copy_vec() {
  
  tf::Taskflow taskflow;
  tf::Executor executor;

  std::vector<T> host1(1024, -1), host2(1024, 1);

  auto gpu = tf::cuda_malloc_device<T>(1024);

  taskflow.emplace([&](tf::cudaFlow& cf){
    cf.cublas([&](tf::cublasFlow& bf) {
      auto h2d = bf.setvec(1024, host1.data(), 1, gpu, 1);
      auto d2h = bf.getvec(1024, gpu, 1, host2.data(), 1);
      h2d.precede(d2h);
    });
  });

  executor.run(taskflow).wait();

  REQUIRE(host1 == host2);

}

TEST_CASE("copy_vec.float") {
  copy_vec<float>();
}

TEST_CASE("copy_vec.double") {
  copy_vec<double>();
}

// ----------------------------------------------------------------------------
// copy through capture
// ----------------------------------------------------------------------------

TEST_CASE("capture.copy") {
  
  tf::Taskflow taskflow;
  tf::Executor executor;

  std::vector<float> host1(1024, -1), host2(1024, 1);

  auto gpu = tf::cuda_malloc_device<float>(1024);

  taskflow.emplace([&](tf::cudaFlow& cf){
    cf.cublas([&](tf::cublasFlow& bf) {
      bf.on([&](cudaStream_t stream){
        tf::cublas_setvec_async(stream, 1024, host1.data(), 1, gpu, 1);
        tf::cublas_getvec_async(stream, 1024, gpu, 1, host2.data(), 1);
      });
    });
  });

  executor.run(taskflow).wait();

  REQUIRE(host1 == host2);
}









