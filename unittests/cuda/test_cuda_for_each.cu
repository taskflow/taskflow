#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>
#include <taskflow/cuda/algorithm/for_each.hpp>

constexpr float eps = 0.0001f;

void run_and_wait(tf::cudaGraphExec& exec) {
  tf::cudaStream stream;
  stream.run(exec).synchronize();
}

// ----------------------------------------------------------------------------
// for_each
// ----------------------------------------------------------------------------

template <typename T>
void for_each() {
    
  tf::Taskflow taskflow;
  tf::Executor executor;

  for(int n=1; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {
    
    taskflow.emplace([n](){

      auto cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
      
      T* gpu = nullptr;
      REQUIRE(cudaMalloc(&gpu, n*sizeof(T)) == cudaSuccess);

      tf::cudaGraph cg;
      auto d2h = cg.copy(cpu, gpu, n);
      auto h2d = cg.copy(gpu, cpu, n);
      auto kernel = cg.for_each(
        gpu, gpu+n, [] __device__ (T& val) { val = 65536; }
      );
      h2d.precede(kernel);
      d2h.succeed(kernel);

      tf::cudaGraphExec exec(cg);

      run_and_wait(exec);

      for(int i=0; i<n; i++) {
        REQUIRE(std::fabs(cpu[i] - (T)65536) < eps);
      }

      // update the kernel
      exec.for_each(kernel,
        gpu, gpu+n, [] __device__ (T& val) { val = 100; }
      );

      run_and_wait(exec);

      for(int i=0; i<n; i++) {
        REQUIRE(std::fabs(cpu[i] - (T)100) < eps);
      }

      std::free(cpu);
      REQUIRE(cudaFree(gpu) == cudaSuccess); 
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cudaGraph.for_each.int" * doctest::timeout(300)) {
  for_each<int>();
}

TEST_CASE("cudaGraph.for_each.float" * doctest::timeout(300)) {
  for_each<float>();
}

TEST_CASE("cudaGraph.for_each.double" * doctest::timeout(300)) {
  for_each<double>();
}

// ----------------------------------------------------------------------------
// for_each_index
// ----------------------------------------------------------------------------

template <typename T>
void for_each_index() {
    
  tf::Taskflow taskflow;
  tf::Executor executor;

  for(int n=1; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {
    
    taskflow.emplace([n](){

      auto cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
      
      T* gpu = nullptr;
      REQUIRE(cudaMalloc(&gpu, n*sizeof(T)) == cudaSuccess);

      tf::cudaGraph cg;
      auto d2h = cg.copy(cpu, gpu, n);
      auto h2d = cg.copy(gpu, cpu, n);
      auto kernel = cg.for_each_index(
        0, n, 1, [gpu] __device__ (int i) { gpu[i] = 65536; }
      );
      h2d.precede(kernel);
      d2h.succeed(kernel);

      tf::cudaGraphExec exec(cg);

      run_and_wait(exec);

      for(int i=0; i<n; i++) {
        REQUIRE(std::fabs(cpu[i] - (T)65536) < eps);
      }
      
      // update
      exec.for_each_index(kernel,
        0, n, 1, [gpu] __device__ (int i) { gpu[i] = (T)100; }
      );

      run_and_wait(exec);
      
      for(int j=0; j<n; j++) {
        REQUIRE(std::fabs(cpu[j] - (T)100) < eps);
      }

      std::free(cpu);
      REQUIRE(cudaFree(gpu) == cudaSuccess); 
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cudaGraph.for_each_index.int" * doctest::timeout(300)) {
  for_each_index<int>();
}

TEST_CASE("cudaGraph.for_each_index.float" * doctest::timeout(300)) {
  for_each_index<float>();
}

TEST_CASE("cudaGraph.for_each_index.double" * doctest::timeout(300)) {
  for_each_index<double>();
}


