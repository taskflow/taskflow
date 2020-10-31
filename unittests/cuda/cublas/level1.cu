#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cublas.hpp>

TEST_CASE("amax-amin") {

  int N = 11111;
  float min_v = FLT_MAX, max_v = -1;

  std::vector<float> host(N);

  for(int i=0; i<N; i++) {
    host[i] = rand() % 100 - 50;
    min_v = std::min(min_v, std::abs(host[i]));
    max_v = std::max(max_v, std::abs(host[i]));
  }

  auto gpu = tf::cuda_malloc_device<float>(N);
  auto min_i = tf::cuda_malloc_device<int>(1);
  auto max_i = tf::cuda_malloc_device<int>(1);
  int h_min_i = -1, h_max_i = -1;

  tf::Taskflow taskflow;
  tf::Executor executor;

  taskflow.emplace([&](tf::cudaFlow& cf){
    auto cublas = cf.cublas([&](tf::cublasFlow& capturer){
      auto amax = capturer.amax(N, gpu, 1, max_i);
      auto amin = capturer.amin(N, gpu, 1, min_i);
      auto vset = capturer.vset(N, host.data(), 1, gpu, 1);
      auto back = capturer.single_task([min_i, max_i] __device__ () {
        (*min_i)--;
        (*max_i)--;
      });
      vset.precede(amin, amax);
      back.succeed(amin, amax);
    });
    auto copy_min_i = cf.copy(&h_min_i, min_i, 1);
    auto copy_max_i = cf.copy(&h_max_i, max_i, 1);
    cublas.precede(copy_min_i, copy_max_i);
  });

  executor.run(taskflow).wait();
  
  REQUIRE(std::abs(host[h_min_i]) == min_v);
  REQUIRE(std::abs(host[h_max_i]) == max_v);

  tf::cuda_free(gpu);
  tf::cuda_free(min_i);
  tf::cuda_free(max_i);
}



