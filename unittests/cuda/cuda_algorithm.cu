#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

constexpr float eps = 0.0001f;

// --------------------------------------------------------
// Testcase: add2
// --------------------------------------------------------
template <typename T>
void add2() {

  //const unsigned N = 1<<20;

  for(size_t N=1; N<=(1<<20); N <<= 1) {

    tf::Taskflow taskflow;
    tf::Executor executor;

    T v1 = ::rand() % 100;
    T v2 = ::rand() % 100;
    T v3 = v1 + v2;

    std::vector<T> hx, hy;

    T* dx {nullptr};
    T* dy {nullptr};
    
    // allocate x
    auto allocate_x = taskflow.emplace([&]() {
      hx.resize(N, v1);
      REQUIRE(cudaMalloc(&dx, N*sizeof(T)) == cudaSuccess);
    }).name("allocate_x");

    // allocate y
    auto allocate_y = taskflow.emplace([&]() {
      hy.resize(N, v2);
      REQUIRE(cudaMalloc(&dy, N*sizeof(T)) == cudaSuccess);
    }).name("allocate_y");
    
    // saxpy
    auto cudaflow = taskflow.emplace([&](tf::cudaFlow& cf) {
      auto h2d_x = cf.copy(dx, hx.data(), N).name("h2d_x");
      auto h2d_y = cf.copy(dy, hy.data(), N).name("h2d_y");
      auto d2h_x = cf.copy(hx.data(), dx, N).name("d2h_x");
      auto d2h_y = cf.copy(hy.data(), dy, N).name("d2h_y");

      //auto kernel = cf.add(dx, N, dx, dy);
      auto kernel = cf.transform(
        dx, N, [] __device__ (T& v1, T& v2) { return v1 + v2; }, 
        dx, dy
      );
      kernel.succeed(h2d_x, h2d_y)
            .precede(d2h_x, d2h_y);
    }).name("saxpy");

    cudaflow.succeed(allocate_x, allocate_y);

    // Add a verification task
    auto verifier = taskflow.emplace([&](){
      for (size_t i = 0; i < N; i++) {
        REQUIRE(std::fabs(hx[i] - v3) < eps);
      }
    }).succeed(cudaflow).name("verify");

    // free memory
    auto deallocate_x = taskflow.emplace([&](){
      REQUIRE(cudaFree(dx) == cudaSuccess);
    }).name("deallocate_x");
    
    auto deallocate_y = taskflow.emplace([&](){
      REQUIRE(cudaFree(dy) == cudaSuccess);
    }).name("deallocate_y");

    verifier.precede(deallocate_x, deallocate_y);

    executor.run(taskflow).wait();
  }
}

TEST_CASE("add2.int" * doctest::timeout(300)) {
  add2<int>();
}

TEST_CASE("add2.float" * doctest::timeout(300)) {
  add2<float>();
}

TEST_CASE("add2.double" * doctest::timeout(300)) {
  add2<double>();
}

// --------------------------------------------------------
// Testcase: add3
// --------------------------------------------------------
template <typename T>
void add3() {

  //const unsigned N = 1<<20;

  for(size_t N=1; N<=(1<<20); N <<= 1) {

    tf::Taskflow taskflow;
    tf::Executor executor;

    T v1 = ::rand() % 100;
    T v2 = ::rand() % 100;
    T v3 = ::rand() % 100;
    T v4 = v1 + v2 + v3;

    std::vector<T> hx, hy, hz;

    T* dx {nullptr};
    T* dy {nullptr};
    T* dz {nullptr};
    
    // allocate x
    auto allocate_x = taskflow.emplace([&]() {
      hx.resize(N, v1);
      REQUIRE(cudaMalloc(&dx, N*sizeof(T)) == cudaSuccess);
    }).name("allocate_x");

    // allocate y
    auto allocate_y = taskflow.emplace([&]() {
      hy.resize(N, v2);
      REQUIRE(cudaMalloc(&dy, N*sizeof(T)) == cudaSuccess);
    }).name("allocate_y");
    
    // allocate z
    auto allocate_z = taskflow.emplace([&]() {
      hz.resize(N, v3);
      REQUIRE(cudaMalloc(&dz, N*sizeof(T)) == cudaSuccess);
    }).name("allocate_y");
    
    // saxpy
    auto cudaflow = taskflow.emplace([&](tf::cudaFlow& cf) {
      auto h2d_x = cf.copy(dx, hx.data(), N).name("h2d_x");
      auto h2d_y = cf.copy(dy, hy.data(), N).name("h2d_y");
      auto h2d_z = cf.copy(dz, hz.data(), N).name("h2d_z");
      auto d2h_x = cf.copy(hx.data(), dx, N).name("d2h_x");
      auto d2h_y = cf.copy(hy.data(), dy, N).name("d2h_y");
      auto d2h_z = cf.copy(hz.data(), dz, N).name("d2h_z");

      //auto kernel = cf.add(dx, N, dx, dy, dz);
      auto kernel = cf.transform(
        dx, N, [] __device__ (T& v1, T& v2, T& v3) { return v1 + v2 + v3; }, 
        dx, dy, dz
      );
      kernel.succeed(h2d_x, h2d_y, h2d_z)
            .precede(d2h_x, d2h_y, d2h_z);
    }).name("saxpy");

    cudaflow.succeed(allocate_x, allocate_y, allocate_z);

    // Add a verification task
    auto verifier = taskflow.emplace([&](){
      for (size_t i = 0; i < N; i++) {
        REQUIRE(std::fabs(hx[i] - v4) < eps);
      }
    }).succeed(cudaflow).name("verify");

    // free memory
    auto deallocate_x = taskflow.emplace([&](){
      REQUIRE(cudaFree(dx) == cudaSuccess);
    }).name("deallocate_x");
    
    auto deallocate_y = taskflow.emplace([&](){
      REQUIRE(cudaFree(dy) == cudaSuccess);
    }).name("deallocate_y");
    
    auto deallocate_z = taskflow.emplace([&](){
      REQUIRE(cudaFree(dz) == cudaSuccess);
    }).name("deallocate_z");

    verifier.precede(deallocate_x, deallocate_y, deallocate_z);

    executor.run(taskflow).wait();
  }
}

TEST_CASE("add3.int" * doctest::timeout(300)) {
  add3<int>();
}

TEST_CASE("add3.float" * doctest::timeout(300)) {
  add3<float>();
}

TEST_CASE("add3.double" * doctest::timeout(300)) {
  add3<double>();
}

// --------------------------------------------------------
// Testcase: multiply2
// --------------------------------------------------------
template <typename T>
void multiply2() {

  //const unsigned N = 1<<20;

  for(size_t N=1; N<=(1<<20); N <<= 1) {

    tf::Taskflow taskflow;
    tf::Executor executor;

    T v1 = ::rand() % 100;
    T v2 = ::rand() % 100;
    T v3 = v1 * v2;

    std::vector<T> hx, hy;

    T* dx {nullptr};
    T* dy {nullptr};
    
    // allocate x
    auto allocate_x = taskflow.emplace([&]() {
      hx.resize(N, v1);
      REQUIRE(cudaMalloc(&dx, N*sizeof(T)) == cudaSuccess);
    }).name("allocate_x");

    // allocate y
    auto allocate_y = taskflow.emplace([&]() {
      hy.resize(N, v2);
      REQUIRE(cudaMalloc(&dy, N*sizeof(T)) == cudaSuccess);
    }).name("allocate_y");
    
    // saxpy
    auto cudaflow = taskflow.emplace([&](tf::cudaFlow& cf) {
      auto h2d_x = cf.copy(dx, hx.data(), N).name("h2d_x");
      auto h2d_y = cf.copy(dy, hy.data(), N).name("h2d_y");
      auto d2h_x = cf.copy(hx.data(), dx, N).name("d2h_x");
      auto d2h_y = cf.copy(hy.data(), dy, N).name("d2h_y");

      //auto kernel = cf.multiply(dx, N, dx, dy);
      auto kernel = cf.transform(
        dx, N, [] __device__ (T& v1, T& v2) { return v1 * v2; }, 
        dx, dy
      );
      kernel.succeed(h2d_x, h2d_y)
            .precede(d2h_x, d2h_y);
    }).name("saxpy");

    cudaflow.succeed(allocate_x, allocate_y);

    // Add a verification task
    auto verifier = taskflow.emplace([&](){
      for (size_t i = 0; i < N; i++) {
        REQUIRE(std::fabs(hx[i] - v3) < eps);
      }
    }).succeed(cudaflow).name("verify");

    // free memory
    auto deallocate_x = taskflow.emplace([&](){
      REQUIRE(cudaFree(dx) == cudaSuccess);
    }).name("deallocate_x");
    
    auto deallocate_y = taskflow.emplace([&](){
      REQUIRE(cudaFree(dy) == cudaSuccess);
    }).name("deallocate_y");

    verifier.precede(deallocate_x, deallocate_y);

    executor.run(taskflow).wait();
  }
}

TEST_CASE("multiply2.int" * doctest::timeout(300)) {
  multiply2<int>();
}

TEST_CASE("multiply2.float" * doctest::timeout(300)) {
  multiply2<float>();
}

TEST_CASE("multiply2.double" * doctest::timeout(300)) {
  multiply2<double>();
}

// ----------------------------------------------------------------------------
// for_each
// ----------------------------------------------------------------------------

template <typename T>
void for_each() {

  for(int n=1; n<=123456; n = n*2 + 1) {

    tf::Taskflow taskflow;
    tf::Executor executor;
    
    T* cpu = nullptr;
    T* gpu = nullptr;

    auto cputask = taskflow.emplace([&](){
      cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
      REQUIRE(cudaMalloc(&gpu, n*sizeof(T)) == cudaSuccess);
    });

    auto gputask = taskflow.emplace([&](tf::cudaFlow& cf) {
      auto h2d = cf.copy(gpu, cpu, n);
      auto kernel = cf.for_each(
        gpu, n, [] __device__ (T& val) { val = 65536; }
      );
      auto d2h = cf.copy(cpu, gpu, n);
      h2d.precede(kernel);
      d2h.succeed(kernel);
    });

    cputask.precede(gputask);
    
    executor.run(taskflow).wait();

    for(int i=0; i<n; i++) {
      REQUIRE(std::fabs(cpu[i] - (T)65536) < eps);
    }

    std::free(cpu);
    REQUIRE(cudaFree(gpu) == cudaSuccess);
  }
}

TEST_CASE("for_each.int" * doctest::timeout(300)) {
  for_each<int>();
}

TEST_CASE("for_each.float" * doctest::timeout(300)) {
  for_each<float>();
}

TEST_CASE("for_each.double" * doctest::timeout(300)) {
  for_each<double>();
}

// --------------------------------------------------------
// Testcase: for_each_index
// --------------------------------------------------------

//template <typename T>
//struct Reset {
//  __device__ void operator () (T& value) {
//    value = 17;
//  }
//};

template <typename T>
void for_each_index() {

  for(int n=1; n<=123456; n = n*2 + 1) {

    tf::Taskflow taskflow;
    tf::Executor executor;
    
    T* cpu = nullptr;
    T* gpu = nullptr;

    auto cputask = taskflow.emplace([&](){
      cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
      REQUIRE(cudaMalloc(&gpu, n*sizeof(T)) == cudaSuccess);
    });

    auto gputask = taskflow.emplace([&](tf::cudaFlow& cf) {
      auto h2d = cf.copy(gpu, cpu, n);
      //auto kernel = cf.for_each_index(gpu, n, [] __device__ (T& value){ value = 17; });
      auto kernel1 = cf.for_each_index(
        0, n, 2, 
        [gpu] __device__ (int i) { gpu[i] = 17; }
      );
      auto kernel2 = cf.for_each_index(
        1, n, 2, 
        [=] __device__ (int i) { gpu[i] = -17; }
      );
      auto d2h = cf.copy(cpu, gpu, n);
      h2d.precede(kernel1, kernel2);
      d2h.succeed(kernel1, kernel2);
    });

    cputask.precede(gputask);
    
    executor.run(taskflow).wait();

    for(int i=0; i<n; i++) {
      if(i % 2 == 0) {
        REQUIRE(std::fabs(cpu[i] - (T)17) < eps);
      }
      else {
        REQUIRE(std::fabs(cpu[i] - (T)(-17)) < eps);
      }
    }

    std::free(cpu);
    REQUIRE(cudaFree(gpu) == cudaSuccess);
  }
}

TEST_CASE("for_each_index.int" * doctest::timeout(300)) {
  for_each_index<int>();
}

TEST_CASE("for_each_index.float" * doctest::timeout(300)) {
  for_each_index<float>();
}

TEST_CASE("for_each_index.double" * doctest::timeout(300)) {
  for_each_index<double>();
}

// ----------------------------------------------------------------------------
// transform
// ----------------------------------------------------------------------------

struct TransformFunc {
  __device__ int operator () (int& v1, float& v2, double& v3) {
    v1 = 1;
    v2 = 3.0f;
    v3 = 5.0;
    return 17;
  }
};

TEST_CASE("transform" * doctest::timeout(300) ) {

  for(unsigned n=1; n<=123456; n = n*2 + 1) {

    tf::Taskflow taskflow;
    tf::Executor executor;
    
    int* htgt = nullptr;
    int* tgt = nullptr;
    int* hsrc1 = nullptr;
    int* src1 = nullptr;
    float* hsrc2 = nullptr;
    float* src2 = nullptr;
    double* hsrc3 = nullptr;
    double* src3 = nullptr;

    auto htgttask = taskflow.emplace([&](){
      htgt = static_cast<int*>(std::calloc(n, sizeof(int)));
      hsrc1 = static_cast<int*>(std::calloc(n, sizeof(int)));
      hsrc2 = static_cast<float*>(std::calloc(n, sizeof(float)));
      hsrc3 = static_cast<double*>(std::calloc(n, sizeof(double)));
      REQUIRE(cudaMalloc(&tgt, n*sizeof(int)) == cudaSuccess);
      REQUIRE(cudaMalloc(&src1, n*sizeof(int)) == cudaSuccess);
      REQUIRE(cudaMalloc(&src2, n*sizeof(float)) == cudaSuccess);
      REQUIRE(cudaMalloc(&src3, n*sizeof(double)) == cudaSuccess);
    });

    auto gputask = taskflow.emplace([&](tf::cudaFlow& cf) {
      auto h2d = cf.copy(tgt, htgt, n);
      auto kernel = cf.transform(
        tgt, 
        n, 
        [] __device__ (int& v1, float& v2, double& v3) -> int {
          v1 = 1;
          v2 = 3.0f;
          v3 = 5.0;
          return 17;
        }, 
        src1, 
        src2, 
        src3
      );
      auto d2h = cf.copy(htgt, tgt, n);
      auto d2h1 = cf.copy(hsrc1, src1, n);
      auto d2h2 = cf.copy(hsrc2, src2, n);
      auto d2h3 = cf.copy(hsrc3, src3, n);
      h2d.precede(kernel);
      kernel.precede(d2h, d2h1, d2h2, d2h3);
    });

    htgttask.precede(gputask);
    
    executor.run(taskflow).wait();

    for(unsigned i=0; i<n; ++i) {
      REQUIRE(htgt[i] == 17);
      REQUIRE(hsrc1[i] == 1);
      REQUIRE(std::fabs(hsrc2[i] - 3.0f) < eps);
      REQUIRE(std::fabs(hsrc3[i] - 5.0) < eps);
    }

    std::free(htgt);
    std::free(hsrc1);
    std::free(hsrc2);
    std::free(hsrc3);
    REQUIRE(cudaFree(tgt) == cudaSuccess);
    REQUIRE(cudaFree(src1) == cudaSuccess);
    REQUIRE(cudaFree(src2) == cudaSuccess);
    REQUIRE(cudaFree(src3) == cudaSuccess);
  }
}


