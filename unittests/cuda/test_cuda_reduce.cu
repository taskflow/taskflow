#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>
#include <taskflow/cuda/algorithm/reduce.hpp>

// ----------------------------------------------------------------------------
// cuda_reduce_bufsz
// ----------------------------------------------------------------------------

TEST_CASE("cuda_reduce.BufferSize") {

  using P = tf::cudaExecutionPolicy<32, 3>;
  
  // within one block
  for(unsigned i=0; i<=P::nv; i++) {
    REQUIRE(P::reduce_bufsz<int>(i) == 0);
  }

  // two blocks
  for(unsigned i=P::nv+1; i<=2*P::nv; i++) {
    REQUIRE(P::reduce_bufsz<int>(i) == 2*sizeof(int));
  }
  
  // three blocks
  for(unsigned i=2*P::nv+1; i<=3*P::nv; i++) {
    REQUIRE(P::reduce_bufsz<int>(i) == 3*sizeof(int));
  }

  REQUIRE(
    P::reduce_bufsz<int>(P::nv*P::nv) == P::nv*sizeof(int)
  );

  REQUIRE(
    P::reduce_bufsz<int>(P::nv*P::nv+1) == (P::nv + 3)*sizeof(int)
  );

  REQUIRE(
    P::reduce_bufsz<int>(P::nv*P::nv*2) == (2*P::nv + 2)*sizeof(int)
  );
  
}

// ----------------------------------------------------------------------------
// cuda_reduce
// ----------------------------------------------------------------------------

template <typename T>
void cuda_reduce() {

  tf::Taskflow taskflow;
  tf::Executor executor;
  
  for(int n=0; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {
    taskflow.emplace([n](){
      tf::cudaStream stream;
      tf::cudaDefaultExecutionPolicy policy(stream);
      
      unsigned bufsz = policy.reduce_bufsz<T>(n);

      T gold {1000};

      auto gpu = tf::cuda_malloc_shared<T>(n);
      auto res = tf::cuda_malloc_shared<T>(1);
      auto buf = tf::cuda_malloc_shared<T>(bufsz);
      for(int i=0; i<n; i++) {
        gpu[i] = i;
        gold += i;
      }
      *res = T{1000};  // initial value
      
      // reduce
      tf::cuda_reduce(policy,
        gpu, gpu + n, res, [] __device__ (T a, T b) { return a + b; }, buf
      );
      stream.synchronize();

      REQUIRE(*res == gold);
      
      // uninitialized reduce
      tf::cuda_uninitialized_reduce(policy,
        gpu, gpu + n, res, [] __device__ (T a, T b) { return a + b; }, buf
      );
      stream.synchronize();
      
      if(n == 0) {
        REQUIRE(*res == 1000);
      }
      else {
        REQUIRE(*res == gold - 1000);
      }

      REQUIRE(cudaFree(gpu) == cudaSuccess);
      REQUIRE(cudaFree(res) == cudaSuccess);
      REQUIRE(cudaFree(buf) == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_reduce.int" * doctest::timeout(300)) {
  cuda_reduce<int>();
}

// ----------------------------------------------------------------------------
// cuda_transform_reduce
// ----------------------------------------------------------------------------

template <typename T>
void cuda_transform_reduce() {

  tf::Taskflow taskflow;
  tf::Executor executor;
  
  for(int n=0; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {
    taskflow.emplace([n](){
      tf::cudaStream stream;
      tf::cudaDefaultExecutionPolicy policy(stream);
      
      unsigned bufsz = policy.reduce_bufsz<T>(n);

      T gold {1000};

      auto gpu = tf::cuda_malloc_shared<T>(n);
      auto res = tf::cuda_malloc_shared<T>(1);
      auto buf = tf::cuda_malloc_shared<T>(bufsz);
      for(int i=0; i<n; i++) {
        gpu[i] = i;
        gold += (-i);
      }
      *res = T{1000};  // initial value
      
      // reduce
      tf::cuda_transform_reduce(policy,
        gpu, gpu + n, res, 
        [] __device__ (T a, T b) { return a + b; }, 
        [] __device__ (T a)      { return -a; }, 
        buf
      );
      stream.synchronize();

      REQUIRE(*res == gold);
      
      // uninitialized reduce
      tf::cuda_uninitialized_transform_reduce(policy,
        gpu, gpu + n, res, 
        [] __device__ (T a, T b) { return a + b; }, 
        [] __device__ (T a)      { return -a; },
        buf
      );
      stream.synchronize();
      
      if(n == 0) {
        REQUIRE(*res == 1000);
      }
      else {
        REQUIRE(*res == gold - 1000);
      }

      REQUIRE(cudaFree(gpu) == cudaSuccess);
      REQUIRE(cudaFree(res) == cudaSuccess);
      REQUIRE(cudaFree(buf) == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_transform_reduce.int" * doctest::timeout(300)) {
  cuda_transform_reduce<int>();
}







