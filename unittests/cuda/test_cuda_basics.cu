#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>

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

void run_and_wait(tf::cudaGraph& cg) {
  tf::cudaStream stream;
  tf::cudaGraphExec exec(cg);
  stream.run(exec).synchronize();
}

// ----------------------------------------------------------------------------
// standalone add
// ----------------------------------------------------------------------------
TEST_CASE("cudaGraph.Standalone") {

  tf::cudaGraph cg;
  tf::cudaStream stream;
  REQUIRE(cg.empty());

  unsigned N = 1024;
    
  auto cpu = static_cast<int*>(std::calloc(N, sizeof(int)));
  auto gpu = tf::cuda_malloc_device<int>(N);

  dim3 g = {(N+255)/256, 1, 1};
  dim3 b = {256, 1, 1};
  auto h2d = cg.copy(gpu, cpu, N);
  auto kernel = cg.kernel(g, b, 0, k_add<int>, gpu, N, 17);
  auto d2h = cg.copy(cpu, gpu, N);
  h2d.precede(kernel);
  kernel.precede(d2h);
    
  for(unsigned i=0; i<N; ++i) {
    REQUIRE(cpu[i] == 0);
  }

  tf::cudaGraphExec exec(cg);

  stream.run(exec).synchronize();
  for(unsigned i=0; i<N; ++i) {
    REQUIRE(cpu[i] == 17);
  }
  
  for(size_t i=0; i<9; i++) {
    stream.run(exec);
  }
  stream.synchronize();

  for(unsigned i=0; i<N; ++i) {
    REQUIRE(cpu[i] == 170);
  }
  
  std::free(cpu);
  tf::cuda_free(gpu);
}

// --------------------------------------------------------
// Testcase: Set
// --------------------------------------------------------
template <typename T>
void set() {
    
  tf::Executor executor;
  tf::Taskflow taskflow;

  for(unsigned n=1; n<=123456; n = n*2 + 1) {

    taskflow.clear();
    
    T* cpu = nullptr;
    T* gpu = nullptr;

    auto cputask = taskflow.emplace([&](){
      cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
      REQUIRE(cudaMalloc(&gpu, n*sizeof(T)) == cudaSuccess);
    });

    auto gputask = taskflow.emplace([&]() {
      tf::cudaGraph cg;
      auto h2d = cg.copy(gpu, cpu, n);
      auto kernel = cg.kernel((n+255)/256, 256, 0, k_set<T>, gpu, n, (T)17);
      auto d2h = cg.copy(cpu, gpu, n);
      h2d.precede(kernel);
      kernel.precede(d2h);
      run_and_wait(cg);

      REQUIRE(cg.num_nodes() == 3);
    });

    cputask.precede(gputask);
    
    executor.run(taskflow).wait();

    for(unsigned i=0; i<n; ++i) {
      REQUIRE(cpu[i] == (T)17);
    }

    std::free(cpu);
    REQUIRE(cudaFree(gpu) == cudaSuccess);
  }
}

TEST_CASE("cudaGraph.Set.i8" * doctest::timeout(300)) {
  set<int8_t>();
}

TEST_CASE("cudaGraph.Set.i16" * doctest::timeout(300)) {
  set<int16_t>();
}

TEST_CASE("cudaGraph.Set.i32" * doctest::timeout(300)) {
  set<int32_t>();
}

// --------------------------------------------------------
// Testcase: Add
// --------------------------------------------------------
template <typename T>
void add() {

  for(unsigned n=1; n<=123456; n = n*2 + 1) {
   
    tf::Taskflow taskflow;
    tf::Executor executor;
    
    T* cpu = nullptr;
    T* gpu = nullptr;
    
    auto cputask = taskflow.emplace([&](){
      cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
      REQUIRE(cudaMalloc(&gpu, n*sizeof(T)) == cudaSuccess);
    });
    
    auto gputask = taskflow.emplace([&](){
      tf::cudaGraph cg;
      dim3 g = {(n+255)/256, 1, 1};
      dim3 b = {256, 1, 1};
      auto h2d = cg.copy(gpu, cpu, n);
      auto ad1 = cg.kernel(g, b, 0, k_add<T>, gpu, n, 1);
      auto ad2 = cg.kernel(g, b, 0, k_add<T>, gpu, n, 2);
      auto ad3 = cg.kernel(g, b, 0, k_add<T>, gpu, n, 3);
      auto ad4 = cg.kernel(g, b, 0, k_add<T>, gpu, n, 4);
      auto d2h = cg.copy(cpu, gpu, n);
      h2d.precede(ad1);
      ad1.precede(ad2);
      ad2.precede(ad3);
      ad3.precede(ad4);
      ad4.precede(d2h);
      run_and_wait(cg);
      REQUIRE(cg.num_nodes() == 6);
    });

    cputask.precede(gputask);
    
    executor.run(taskflow).wait();

    for(unsigned i=0; i<n; ++i) {
      REQUIRE(cpu[i] == 10);
    }

    std::free(cpu);
    REQUIRE(cudaFree(gpu) == cudaSuccess);
  }
}

TEST_CASE("cudaGraph.Add.i8" * doctest::timeout(300)) {
  add<int8_t>();
}

TEST_CASE("cudaGraph.Add.i16" * doctest::timeout(300)) {
  add<int16_t>();
}

TEST_CASE("cudaGraph.Add.i32" * doctest::timeout(300)) {
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

  tf::Taskflow taskflow;
  tf::Executor executor;

  T* cpu = nullptr;
  T* gpu = nullptr;
  
  auto cputask = taskflow.emplace([&](){
    cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
    REQUIRE(cudaMalloc(&gpu, n*sizeof(T)) == cudaSuccess);
  });

  auto gputask = taskflow.emplace([&]() {
    tf::cudaGraph cg;
    dim3 g = {1, 1, 1};
    dim3 b = {1, 1, 1};
    auto h2d = cg.copy(gpu, cpu, n);
    auto d2h = cg.copy(cpu, gpu, n);

    std::vector<tf::cudaTask> tasks(n+1);

    for(unsigned i=1; i<=n; ++i) {
      tasks[i] = cg.kernel(g, b, 0, k_single_set<T>, gpu, i-1, (T)17);

      auto p = i/2;
      if(p != 0) {
        tasks[p].precede(tasks[i]);
      }

      tasks[i].precede(d2h);
      h2d.precede(tasks[i]);
    }

    run_and_wait(cg);
    REQUIRE(cg.num_nodes() == n + 2);
  });

  cputask.precede(gputask);
  
  executor.run(taskflow).wait();

  for(unsigned i=0; i<n; ++i) {
    REQUIRE(cpu[i] == (T)17);
  }

  std::free(cpu);
  REQUIRE(cudaFree(gpu) == cudaSuccess);
}

TEST_CASE("cudaGraph.BSet.i8" * doctest::timeout(300)) {
  bset<int8_t>();
}

TEST_CASE("cudaGraph.BSet.i16" * doctest::timeout(300)) {
  bset<int16_t>();
}

TEST_CASE("cudaGraph.BSet.i32" * doctest::timeout(300)) {
  bset<int32_t>();
}

// --------------------------------------------------------
// Testcase: Memset
// --------------------------------------------------------

TEST_CASE("cudaGraph.Memset" * doctest::timeout(300)) {

  tf::Taskflow taskflow;
  tf::Executor executor;
  
  const int N = 100;

  int* cpu = new int [N];
  int* gpu = nullptr;
    
  REQUIRE(cudaMalloc(&gpu, N*sizeof(int)) == cudaSuccess);

  for(int r=1; r<=100; ++r) {

    int start = ::rand() % N;

    for(int i=0; i<N; ++i) {
      cpu[i] = 999;
    }
    
    taskflow.emplace([&](){
      tf::cudaGraph cg;
      dim3 g = {(unsigned)(N+255)/256, 1, 1};
      dim3 b = {256, 1, 1};
      auto kset = cg.kernel(g, b, 0, k_set<int>, gpu, N, 123);
      auto copy = cg.copy(cpu, gpu, N);
      auto zero = cg.memset(gpu+start, 0x3f, (N-start)*sizeof(int));
      kset.precede(zero);
      zero.precede(copy);
      run_and_wait(cg);
      REQUIRE(cg.num_nodes() == 3);
    });
    
    executor.run(taskflow).wait();

    for(int i=0; i<start; ++i) {
      REQUIRE(cpu[i] == 123);
    }
    for(int i=start; i<N; ++i) {
      REQUIRE(cpu[i] == 0x3f3f3f3f);
    }
  }
  
  delete [] cpu;
  REQUIRE(cudaFree(gpu) == cudaSuccess);
}

// --------------------------------------------------------
// Testcase: Memcpy
// --------------------------------------------------------
template <typename T>
void memcpy() {
  
  tf::Taskflow taskflow;
  tf::Executor executor;
  
  const int N = 97;

  T* cpu = new T [N];
  T* gpu = nullptr;
    
  REQUIRE(cudaMalloc(&gpu, N*sizeof(T)) == cudaSuccess);

  for(int r=1; r<=100; ++r) {

    int start = ::rand() % N;

    for(int i=0; i<N; ++i) {
      cpu[i] = (T)999;
    }
    
    taskflow.emplace([&](){
      tf::cudaGraph cg;
      dim3 g = {(unsigned)(N+255)/256, 1, 1};
      dim3 b = {256, 1, 1};
      auto kset = cg.kernel(g, b, 0, k_set<T>, gpu, N, (T)123);
      auto zero = cg.memset(gpu+start, (T)0, (N-start)*sizeof(T));
      auto copy = cg.memcpy(cpu, gpu, N*sizeof(T));
      kset.precede(zero);
      zero.precede(copy);
      run_and_wait(cg);
      REQUIRE(cg.num_nodes() == 3);
    });
    
    executor.run(taskflow).wait();

    for(int i=0; i<start; ++i) {
      REQUIRE(std::fabs(cpu[i] - (T)123) < 1e-4);
    }
    for(int i=start; i<N; ++i) {
      REQUIRE(std::fabs(cpu[i] - (T)0) < 1e-4);
    }
  }
  
  delete [] cpu;
  REQUIRE(cudaFree(gpu) == cudaSuccess);
}

TEST_CASE("cudaGraph.Memcpy.i8") {
  memcpy<int8_t>();
}

TEST_CASE("cudaGraph.Memcpy.i16") {
  memcpy<int16_t>();
}

TEST_CASE("cudaGraph.Memcpy.i32") {
  memcpy<int32_t>();
}

TEST_CASE("cudaGraph.Memcpy.f32") {
  memcpy<float>();
}

TEST_CASE("cudaGraph.Memcpy.f64") {
  memcpy<double>();
}

// --------------------------------------------------------
// Testcase: fill
// --------------------------------------------------------
template <typename T>
void fill(T value) {
  
  tf::Taskflow taskflow;
  tf::Executor executor;
  
  const int N = 107;

  T* cpu = new T [N];
  T* gpu = nullptr;
    
  REQUIRE(cudaMalloc(&gpu, N*sizeof(T)) == cudaSuccess);

  for(int r=1; r<=100; ++r) {

    int start = ::rand() % N;

    for(int i=0; i<N; ++i) {
      cpu[i] = (T)999;
    }
    
    taskflow.emplace([&](){

      tf::cudaGraph cg;

      dim3 g = {(unsigned)(N+255)/256, 1, 1};
      dim3 b = {256, 1, 1};
      auto kset = cg.kernel(g, b, 0, k_set<T>, gpu, N, (T)123);
      auto fill = cg.fill(gpu+start, value, (N-start));
      auto copy = cg.copy(cpu, gpu, N);
      kset.precede(fill);
      fill.precede(copy);

      run_and_wait(cg);
      REQUIRE(cg.num_nodes() == 3);
    });
    
    executor.run(taskflow).wait();

    for(int i=0; i<start; ++i) {
      REQUIRE(std::fabs(cpu[i] - (T)123) < 1e-4);
    }
    for(int i=start; i<N; ++i) {
      REQUIRE(std::fabs(cpu[i] - value) < 1e-4);
    }
  }

  delete [] cpu;
  REQUIRE(cudaFree(gpu) == cudaSuccess);
}

TEST_CASE("cudaGraph.Fill.i8") {
  fill<int8_t>(+123);
  fill<int8_t>(-123);
}

TEST_CASE("cudaGraph.Fill.i16") {
  fill<int16_t>(+12345);
  fill<int16_t>(-12345);
}

TEST_CASE("cudaGraph.Fill.i32") {
  fill<int32_t>(+123456789);
  fill<int32_t>(-123456789);
}

TEST_CASE("cudaGraph.Fill.f32") {
  fill<float>(+123456789.0f);
  fill<float>(-123456789.0f);
}

// --------------------------------------------------------
// Testcase: Zero
// --------------------------------------------------------
template <typename T>
void zero() {
  
  tf::Taskflow taskflow;
  tf::Executor executor;
  
  const int N = 100;

  T* cpu = new T [N];
  T* gpu = nullptr;
    
  REQUIRE(cudaMalloc(&gpu, N*sizeof(T)) == cudaSuccess);

  for(int r=1; r<=100; ++r) {

    int start = ::rand() % N;

    for(int i=0; i<N; ++i) {
      cpu[i] = (T)999;
    }
    
    taskflow.emplace([&](){

      tf::cudaGraph cg;

      dim3 g = {(unsigned)(N+255)/256, 1, 1};
      dim3 b = {256, 1, 1};
      auto kset = cg.kernel(g, b, 0, k_set<T>, gpu, N, (T)123);
      auto zero = cg.zero(gpu+start, (N-start));
      auto copy = cg.copy(cpu, gpu, N);
      kset.precede(zero);
      zero.precede(copy);

      run_and_wait(cg);
      REQUIRE(cg.num_nodes() == 3);
    });
    
    executor.run(taskflow).wait();

    for(int i=0; i<start; ++i) {
      REQUIRE(std::fabs(cpu[i] - (T)123) < 1e-4);
    }
    for(int i=start; i<N; ++i) {
      REQUIRE(std::fabs(cpu[i] - (T)0) < 1e-4);
    }
  }

  delete [] cpu;
  REQUIRE(cudaFree(gpu) == cudaSuccess);
}

TEST_CASE("cudaGraph.Zero.i8") {
  zero<int8_t>();
}

TEST_CASE("cudaGraph.Zero.i16") {
  zero<int16_t>();
}

TEST_CASE("cudaGraph.Zero.i32") {
  zero<int32_t>();
}

TEST_CASE("cudaGraph.Zero.f32") {
  zero<float>();
}

// --------------------------------------------------------
// Testcase: Barrier
// --------------------------------------------------------
template <typename T>
void barrier() {

  const unsigned n = 1000;
  
  tf::Taskflow taskflow;
  tf::Executor executor;
  
  T* cpu = nullptr;
  T* gpu = nullptr;

  auto cputask = taskflow.emplace([&](){
    cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
    REQUIRE(cudaMalloc(&gpu, n*sizeof(T)) == cudaSuccess);
  });

  auto gputask = taskflow.emplace([&]() {
    
    tf::cudaGraph cg;

    dim3 g = {1, 1, 1};
    dim3 b = {1, 1, 1};
    auto br1 = cg.noop();
    auto br2 = cg.noop();
    auto br3 = cg.noop();
    auto h2d = cg.copy(gpu, cpu, n);
    auto d2h = cg.copy(cpu, gpu, n);

    h2d.precede(br1);

    for(unsigned i=0; i<n; ++i) {
      auto k1 = cg.kernel(g, b, 0, k_single_set<T>, gpu, i, (T)17);
      k1.succeed(br1)
        .precede(br2);

      auto k2 = cg.kernel(g, b, 0, k_single_add<T>, gpu, i, (T)3);
      k2.succeed(br2)
        .precede(br3);
    }

    br3.precede(d2h);

    run_and_wait(cg);
    REQUIRE(cg.num_nodes() == 5 + 2*n);
  });

  cputask.precede(gputask);
  
  executor.run(taskflow).wait();

  for(unsigned i=0; i<n; ++i) {
    REQUIRE(cpu[i] == (T)20);
  }

  std::free(cpu);
  REQUIRE(cudaFree(gpu) == cudaSuccess);
}

TEST_CASE("cudaGraph.Barrier.i8" * doctest::timeout(300)) {
  barrier<int8_t>();
}

TEST_CASE("cudaGraph.Barrier.i16" * doctest::timeout(300)) {
  barrier<int16_t>();
}

TEST_CASE("cudaGraph.Barrier.i32" * doctest::timeout(300)) {
  barrier<int32_t>();
}

// ----------------------------------------------------------------------------
// Conditional GPU tasking
// ----------------------------------------------------------------------------

TEST_CASE("cudaGraph.ConditionTask" * doctest::timeout(300)) {

  tf::Taskflow taskflow;
  tf::Executor executor;

  const unsigned n = 1000;
    
  int* cpu = nullptr;
  int* gpu = nullptr;

  auto cputask = taskflow.emplace([&](){
    cpu = static_cast<int*>(std::calloc(n, sizeof(int)));
    REQUIRE(cudaMalloc(&gpu, n*sizeof(int)) == cudaSuccess);
  });

  auto gputask = taskflow.emplace([&]() {
    tf::cudaGraph cg;
    dim3 g = {(n+255)/256, 1, 1};
    dim3 b = {256, 1, 1};
    auto h2d = cg.copy(gpu, cpu, n);
    auto kernel = cg.kernel(g, b, 0, k_add<int>, gpu, n, 1);
    auto d2h = cg.copy(cpu, gpu, n);
    h2d.precede(kernel);
    kernel.precede(d2h);
    run_and_wait(cg);
    REQUIRE(cg.num_nodes() == 3);
  });

  auto condition = taskflow.emplace([&cpu, round=0] () mutable {
    ++round;
    for(unsigned i=0; i<n; ++i) {
      REQUIRE(cpu[i] == round);
    }
    return round >= 100;
  });

  auto freetask = taskflow.emplace([&](){
    REQUIRE(cudaFree(gpu) == cudaSuccess);
    std::free(cpu);
  });

  cputask.precede(gputask);
  gputask.precede(condition);
  condition.precede(gputask, freetask);
  
  executor.run(taskflow).wait();
}


// ----------------------------------------------------------------------------
// Predicate
// ----------------------------------------------------------------------------

TEST_CASE("cudaGraph.Loop") {

  tf::Taskflow taskflow;
  tf::Executor executor;

  const unsigned n = 1000;
    
  int* cpu = nullptr;
  int* gpu = nullptr;

  auto cputask = taskflow.emplace([&](){
    cpu = static_cast<int*>(std::calloc(n, sizeof(int)));
    REQUIRE(cudaMalloc(&gpu, n*sizeof(int)) == cudaSuccess);
    REQUIRE(cudaMemcpy(gpu, cpu, n*sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
  });

  auto gputask = taskflow.emplace([&]() {
    tf::cudaGraph cg;
    dim3 g = {(n+255)/256, 1, 1};
    dim3 b = {256, 1, 1};
    auto kernel = cg.kernel(g, b, 0, k_add<int>, gpu, n, 1);
    auto copy = cg.copy(cpu, gpu, n);
    kernel.precede(copy);

    tf::cudaStream stream;
    tf::cudaGraphExec exec(cg);
    for(int i=0; i<100; i++) {
      stream.run(exec);
    }
    stream.synchronize();
  });

  auto freetask = taskflow.emplace([&](){
    for(unsigned i=0; i<n; ++i) {
      REQUIRE(cpu[i] == 100);
    }
    REQUIRE(cudaFree(gpu) == cudaSuccess);
    std::free(cpu);
  });

  cputask.precede(gputask);
  gputask.precede(freetask);
  
  executor.run(taskflow).wait();
}
