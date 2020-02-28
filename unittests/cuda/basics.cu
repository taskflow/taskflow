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
__global__ void k_single_set(T* ptr, int i, T value) {
  ptr[i] = value;
}

template <typename T>
__global__ void k_add(T* ptr, size_t N, T value) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    ptr[i] += value;
  }
}

template <typename T>
__global__ void k_single_add(T* ptr, int i, T value) {
  ptr[i] += value;
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

TEST_CASE("Set.i8" * doctest::timeout(300)) {
  set<int8_t>();
}

TEST_CASE("Set.i16" * doctest::timeout(300)) {
  set<int16_t>();
}

TEST_CASE("Set.i32" * doctest::timeout(300)) {
  set<int32_t>();
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

TEST_CASE("Add.i8" * doctest::timeout(300)) {
  add<int8_t>();
}

TEST_CASE("Add.i16" * doctest::timeout(300)) {
  add<int16_t>();
}

TEST_CASE("Add.i32" * doctest::timeout(300)) {
  add<int32_t>();
}

// TODO: 64-bit fail?
//TEST_CASE("Add.i64" * doctest::timeout(300)) {
//  add<int64_t>();
//}


// --------------------------------------------------------
// Testcase: Binary Set
// --------------------------------------------------------
template <typename T>
void bset() {

  const unsigned n = 10000;

  tf::cudaGraph G;
  tf::cudaFlow cf(G);
  
  T* cpu = nullptr;
  T* gpu = nullptr;

  cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
  REQUIRE(cudaMalloc(&gpu, n*sizeof(T)) == cudaSuccess);

  dim3 g = {1, 1, 1};
  dim3 b = {1, 1, 1};
  auto h2d = cf.copy(gpu, cpu, n);
  auto d2h = cf.copy(cpu, gpu, n);

  std::vector<tf::cudaTask> tasks(n+1);

  for(unsigned i=1; i<=n; ++i) {
    tasks[i] = cf.kernel(g, b, 0, k_single_set<T>, gpu, i-1, (T)17);

    auto p = i/2;
    if(p != 0) {
      tasks[p].precede(tasks[i]);
    }

    tasks[i].precede(d2h);
    h2d.precede(tasks[i]);
  }
  
  run(G);

  for(unsigned i=0; i<n; ++i) {
    REQUIRE(cpu[i] == (T)17);
  }

  std::free(cpu);
  REQUIRE(cudaFree(gpu) == cudaSuccess);
}

TEST_CASE("BSet.i8" * doctest::timeout(300)) {
  bset<int8_t>();
}

TEST_CASE("BSet.i16" * doctest::timeout(300)) {
  bset<int16_t>();
}

TEST_CASE("BSet.i32" * doctest::timeout(300)) {
  bset<int32_t>();
}

// --------------------------------------------------------
// Testcase: Barrier
// --------------------------------------------------------
template <typename T>
void barrier() {

  const unsigned n = 1000;

  tf::cudaGraph G;
  tf::cudaFlow cf(G);
  
  T* cpu = nullptr;
  T* gpu = nullptr;

  cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
  REQUIRE(cudaMalloc(&gpu, n*sizeof(T)) == cudaSuccess);

  dim3 g = {1, 1, 1};
  dim3 b = {1, 1, 1};
  auto br1 = cf.noop();
  auto br2 = cf.noop();
  auto br3 = cf.noop();
  auto h2d = cf.copy(gpu, cpu, n);
  auto d2h = cf.copy(cpu, gpu, n);

  h2d.precede(br1);

  for(unsigned i=0; i<n; ++i) {
    auto k1 = cf.kernel(g, b, 0, k_single_set<T>, gpu, i, (T)17);
    k1.succeed(br1)
      .precede(br2);

    auto k2 = cf.kernel(g, b, 0, k_single_add<T>, gpu, i, (T)3);
    k2.succeed(br2)
      .precede(br3);
  }

  br3.precede(d2h);
  
  run(G);

  for(unsigned i=0; i<n; ++i) {
    REQUIRE(cpu[i] == (T)20);
  }

  std::free(cpu);
  REQUIRE(cudaFree(gpu) == cudaSuccess);
}

TEST_CASE("Barrier.i8" * doctest::timeout(300)) {
  barrier<int8_t>();
}

TEST_CASE("Barrier.i16" * doctest::timeout(300)) {
  barrier<int16_t>();
}

TEST_CASE("Barrier.i32" * doctest::timeout(300)) {
  barrier<int32_t>();
}

