#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/cudaflow.hpp>

// ----------------------------------------------------------------------------
// kernel helper
// ----------------------------------------------------------------------------
template <typename T>
__global__ void k_set(T* ptr, size_t N, T value) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    ptr[i] = value;
  }
}

template <typename T>
__global__ void k_add(T* ptr, size_t N, T value) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    ptr[i] += value;
  }
}

// --------------------------------------------------------
// standalone execution
// --------------------------------------------------------
void run(tf::cudaGraph& G) {
  cudaGraphExec_t graphExec;
  TF_CHECK_CUDA(
    cudaGraphInstantiate(&graphExec, G.native_handle(), nullptr, nullptr, 0),
    "failed to create an executable cudaGraph"
  );
  TF_CHECK_CUDA(cudaGraphLaunch(graphExec, 0), "failed to launch cudaGraph")
  TF_CHECK_CUDA(cudaStreamSynchronize(0), "failed to sync cudaStream");
  TF_CHECK_CUDA(
    cudaGraphExecDestroy(graphExec), "failed to destroy an executable cudaGraph"
  );
}

// --------------------------------------------------------
// Testcase: Builder
// --------------------------------------------------------
TEST_CASE("Builder" * doctest::timeout(300)) {

  tf::cudaGraph G;
  tf::cudaFlow cf(G);

  int source = 1;
  int target = 1;

  auto copy1 = cf.copy(&target, &source, 1).name("copy1");
  auto copy2 = cf.copy(&target, &source, 1).name("copy2");
  auto copy3 = cf.copy(&target, &source, 1).name("copy3");

  REQUIRE(copy1.name() == "copy1");
  REQUIRE(copy2.name() == "copy2");
  REQUIRE(copy3.name() == "copy3");

  REQUIRE(!copy1.empty());
  REQUIRE(!copy2.empty());
  REQUIRE(!copy3.empty());
  
  copy1.precede(copy2);
  copy2.succeed(copy3);

  REQUIRE(copy1.num_successors() == 1);
  REQUIRE(copy2.num_successors() == 0);
  REQUIRE(copy3.num_successors() == 1);
}

// --------------------------------------------------------
// Testcase: Set
// --------------------------------------------------------
template <typename T>
void set() {

  for(unsigned n=1; n<=1345678; n = n*2 + 1) {
    tf::cudaGraph G;
    tf::cudaFlow cf(G);
    
    T* cpu = nullptr;
    T* gpu = nullptr;

    cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
    REQUIRE(cudaMalloc(&gpu, n*sizeof(T)) == cudaSuccess);

    dim3 g = {(n+255)/256, 1, 1};
    dim3 b = {256, 1, 1};
    auto h2d = cf.copy(gpu, cpu, n);
    auto kernel = cf.kernel(g, b, 0, k_set<T>, gpu, n, (T)17);
    auto d2h = cf.copy(cpu, gpu, n);
    h2d.precede(kernel);
    kernel.precede(d2h);
    
    run(G);

    for(unsigned i=0; i<n; ++i) {
      REQUIRE(cpu[i] == (T)17);
    }

    std::free(cpu);
    REQUIRE(cudaFree(gpu) == cudaSuccess);
  }
}

TEST_CASE("Set.Int8" * doctest::timeout(300)) {
  set<int8_t>();
}

TEST_CASE("Set.Int16" * doctest::timeout(300)) {
  set<int16_t>();
}

TEST_CASE("Set.Int32" * doctest::timeout(300)) {
  set<int32_t>();
}

TEST_CASE("Set.Float" * doctest::timeout(300)) {
  set<float>();
}

// --------------------------------------------------------
// Testcase: Add
// --------------------------------------------------------
template <typename T>
void add() {

  for(unsigned n=1; n<=1345678; n = n*2 + 1) {
    tf::cudaGraph G;
    tf::cudaFlow cf(G);
    
    T* cpu = nullptr;
    T* gpu = nullptr;

    cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
    REQUIRE(cudaMalloc(&gpu, n*sizeof(T)) == cudaSuccess);

    dim3 g = {(n+255)/256, 1, 1};
    dim3 b = {256, 1, 1};
    auto h2d = cf.copy(gpu, cpu, n);
    auto ad1 = cf.kernel(g, b, 0, k_add<T>, gpu, n, 1);
    auto ad2 = cf.kernel(g, b, 0, k_add<T>, gpu, n, 2);
    auto ad3 = cf.kernel(g, b, 0, k_add<T>, gpu, n, 3);
    auto ad4 = cf.kernel(g, b, 0, k_add<T>, gpu, n, 4);
    auto d2h = cf.copy(cpu, gpu, n);
    h2d.precede(ad1);
    ad1.precede(ad2);
    ad2.precede(ad3);
    ad3.precede(ad4);
    ad4.precede(d2h);
    
    run(G);

    for(unsigned i=0; i<n; ++i) {
      REQUIRE(cpu[i] == 10);
    }

    std::free(cpu);
    REQUIRE(cudaFree(gpu) == cudaSuccess);
  }
}

TEST_CASE("Add.Int8" * doctest::timeout(300)) {
  add<int8_t>();
}

TEST_CASE("Add.Int16" * doctest::timeout(300)) {
  add<int16_t>();
}

TEST_CASE("Add.Int32" * doctest::timeout(300)) {
  add<int32_t>();
}






