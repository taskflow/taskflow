#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cudaflow.hpp>
#include <taskflow/cublasflow.hpp>

template <typename T>
void copy_vec() {
  
  tf::Taskflow taskflow;
  tf::Executor executor;

  std::vector<T> host1(1024, -1), host2(1024, 1);

  auto gpu = tf::cuda_malloc_device<T>(1024);

  taskflow.emplace([&](tf::cudaFlow& cf){
    cf.capture([&](tf::cudaFlowCapturer& cap) {
      auto bf = cap.make_capturer<tf::cublasFlowCapturer>();
      auto h2d = bf->vset(1024, host1.data(), 1, gpu, 1);
      auto d2h = bf->vget(1024, gpu, 1, host2.data(), 1);
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








