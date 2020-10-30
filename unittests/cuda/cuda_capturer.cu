#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

// ----------------------------------------------------------------------------
// Kernel Helpers
// ----------------------------------------------------------------------------

template <typename T>
__global__ void k_add(T* ptr, size_t N, T value) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    ptr[i] += value;
  }
}

// ----------------------------------------------------------------------------
// MemcpyCapturer
// ----------------------------------------------------------------------------

TEST_CASE("MemcpyCapture") {

  tf::Taskflow taskflow;
  tf::Executor executor;

  const size_t N = 1177;
  
  std::vector<int> src(N, 100), tgt(N, -100);
  auto gpu = tf::cuda_malloc_device<int>(N);

  taskflow.emplace([&](tf::cudaFlow& cf){
    cf.capture([&](tf::cudaFlowCapturer& cap){
      auto d2h = cap.memcpy(tgt.data(), gpu, N*sizeof(int));
      auto h2d = cap.memcpy(gpu, src.data(), N*sizeof(int));
      h2d.precede(d2h);
    });
  });

  executor.run(taskflow).wait();
  
  REQUIRE(tgt == src);

  tf::cuda_free(gpu);
}

// ----------------------------------------------------------------------------
// MemsetCapturer
// ----------------------------------------------------------------------------
TEST_CASE("MemsetCapture") {

  tf::Taskflow taskflow;
  tf::Executor executor;

  const size_t N = 1177;
  
  std::vector<int> src(N, 100), tgt(N, -100);
  auto gpu = tf::cuda_malloc_device<int>(N);

  taskflow.emplace([&](tf::cudaFlow& cf){
    cf.capture([&](tf::cudaFlowCapturer& cap){
      auto d2h = cap.memcpy(tgt.data(), gpu, N*sizeof(int));
      auto set = cap.memset(gpu, 0, N*sizeof(int));
      auto h2d = cap.memcpy(gpu, src.data(), N*sizeof(int));
      h2d.precede(set);
      set.precede(d2h);
    });
  });

  executor.run(taskflow).wait();
    
  for(size_t i=0; i<N; i++) {
    REQUIRE(src[i] == 100);
    REQUIRE(tgt[i] == 0);
  }

  tf::cuda_free(gpu);
}

// ----------------------------------------------------------------------------
// KernelCapturer
// ----------------------------------------------------------------------------
TEST_CASE("KernelCapture") {

  tf::Taskflow taskflow;
  tf::Executor executor;

  const size_t N = 1177;
  const size_t B = 256;
  
  std::vector<int> src(N, 100), tgt(N, -100);
  auto gpu = tf::cuda_malloc_device<int>(N);

  taskflow.emplace([&](tf::cudaFlow& cf){
    cf.capture([&](tf::cudaFlowCapturer& cap){
      auto d2h = cap.memcpy(tgt.data(), gpu, N*sizeof(int));
      auto set = cap.memset(gpu, 0, N*sizeof(int));
      auto add = cap.kernel((N+B-1)/B, B, 0, k_add<int>, gpu, N, 17);
      auto h2d = cap.memcpy(gpu, src.data(), N*sizeof(int));
      h2d.precede(set);
      set.precede(add);
      add.precede(d2h);
    });
  });

  executor.run(taskflow).wait();
    
  for(size_t i=0; i<N; i++) {
    REQUIRE(src[i] == 100);
    REQUIRE(tgt[i] == 17);
  }

  tf::cuda_free(gpu);
}


