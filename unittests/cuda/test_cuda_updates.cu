#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>

template <typename T>
void run_and_wait(T& cf) {
  tf::cudaStream stream;
  cf.run(stream);
  stream.synchronize();
}

//verify
template <typename T>
__global__
void verify(const T* a, const T* b, bool* check, size_t size) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(;tid < size; tid += gridDim.x * blockDim.x) {
    if(a[tid] != b[tid]) {
      *check = false;
      return;
    }
  }
}

template <typename T>
__global__ void k_add(T* ptr, size_t N, T value) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    ptr[i] += value;
  }
}

//add
template <typename T>
__global__
void add(const T* a, const T* b, T* c, size_t size) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(;tid < size; tid += gridDim.x * blockDim.x) {
    c[tid] = a[tid] + b[tid];
  }
}

//multiply
template <typename T>
__global__
void multiply(const T* a, const T* b, T* c, size_t size) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(;tid < size; tid += gridDim.x * blockDim.x) {
    c[tid] = a[tid] * b[tid];
  }
}

// update single_task
TEST_CASE("cudaGraph.Update.SingleTask") {

  tf::cudaGraph cg;
  
  auto var = tf::cuda_malloc_shared<int>(1);
  *var = 1;
  REQUIRE(*var == 1);

  auto task = cg.single_task([=] __device__ () { *var = 2; });

  tf::cudaGraphExec exec(cg);
  tf::cudaStream stream;
  stream.run(exec).synchronize();

  REQUIRE(*var == 2);

  exec.single_task(task, [=] __device__ () { *var = 10; });
  
  stream.run(exec).synchronize();

  REQUIRE(*var == 10);

  tf::cuda_free(var);
}


// update kernel
TEST_CASE("cudaGraph.Update.Kernel") {

  const size_t N = 1024;
  
  tf::cudaGraph cg;

  auto vec = tf::cuda_malloc_shared<int>(N);
  
  auto t1 = cg.zero(vec, N);
  auto t2 = cg.kernel(2, 512, 0, k_add<int>, vec, N, 10);
  t1.precede(t2);

  tf::cudaGraphExec exec(cg);
  tf::cudaStream stream;

  stream.run(exec).synchronize();
  
  for(size_t i=0; i<N; i++) {
    REQUIRE(vec[i] == 10);
  }
  
  exec.zero(t1, vec, N/2);
  exec.kernel(t2, 2, 512, 0, k_add<int>, vec, N, 20);
  
  stream.run(exec).synchronize();
  
  for(size_t i=0; i<N/2; i++) {
    REQUIRE(vec[i] == 20);
  }
  
  for(size_t i=N/2; i<N; i++) {
    REQUIRE(vec[i] == 30);
  }

  tf::cuda_free(vec);
}

// update memset
TEST_CASE("cudaGraph.Update.Memset") {

  const size_t N = 1024;
  
  tf::cudaGraph cg;

  auto vec = tf::cuda_malloc_shared<int>(N);
  
  auto t1 = cg.memset(vec, 0x01, N*sizeof(int));

  tf::cudaGraphExec exec(cg);
  tf::cudaStream stream;

  stream.run(exec).synchronize();
  
  for(size_t i=0; i<N; i++) {
    REQUIRE(vec[i] == 0x01010101);
  }

  exec.memset(t1, vec, 0x0F, N*sizeof(int));

  stream.run(exec).synchronize();
  
  for(size_t i=0; i<N; i++) {
    REQUIRE(vec[i] == 0x0F0F0F0F);
  }

  tf::cuda_free(vec);
}

// update memcpy
TEST_CASE("cudaGraph.Update.Memcpy") {

  const size_t N = 1024;
  
  tf::cudaGraph cg;

  auto vec1 = tf::cuda_malloc_shared<int>(N);
  auto vec2 = tf::cuda_malloc_shared<int>(N);
  auto vec3 = tf::cuda_malloc_shared<int>(N);

  for(size_t i=0; i<N; i++) {
    vec1[i] = 1;
    vec2[i] = 2;
    vec3[i] = 3;
  }
  
  auto t1 = cg.memcpy(vec2, vec1, N*sizeof(int));

  tf::cudaGraphExec exec(cg);
  tf::cudaStream stream;

  stream.run(exec).synchronize();
  
  for(size_t i=0; i<N; i++) {
    REQUIRE(vec2[i] == 1);
  }

  exec.memcpy(t1, vec2, vec3, N*sizeof(int));

  stream.run(exec).synchronize();
  
  for(size_t i=0; i<N; i++) {
    REQUIRE(vec2[i] == 3);
    vec2[i] = 0;
  }

  stream.run(exec).synchronize();

  for(size_t i=0; i<N; ++i) {
    REQUIRE(vec2[i] == 3);
  }

  tf::cuda_free(vec1);
  tf::cuda_free(vec2);
  tf::cuda_free(vec3);
}

