#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taskflow/syclflow.hpp>

constexpr float eps = 0.0001f;

// ----------------------------------------------------------------------------
// for_each
// ----------------------------------------------------------------------------

template <typename T>
void for_each() {

  tf::Taskflow taskflow;
  tf::Executor executor;
  sycl::queue queue;

  for(int n=1; n<=123456; n = n*2 + 1) {

    taskflow.clear();

    T* cpu = nullptr;
    T* gpu = nullptr;

    auto cputask = taskflow.emplace([&](){
      cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
      gpu = sycl::malloc_device<T>(n, queue);
    });

    tf::Task gputask;

    gputask = taskflow.emplace_on([&](tf::syclFlow& cf) {
      auto d2h = cf.copy(cpu, gpu, n);
      auto h2d = cf.copy(gpu, cpu, n);
      auto kernel = cf.for_each(
        gpu, gpu+n, [] (T& val) { val = 65536; }
      );
      h2d.precede(kernel);
      d2h.succeed(kernel);
    }, queue);

    cputask.precede(gputask);

    executor.run(taskflow).wait();

    for(int i=0; i<n; i++) {
      REQUIRE(std::fabs(cpu[i] - (T)65536) < eps);
    }

    std::free(cpu);
    sycl::free(gpu, queue);
  }
}

TEST_CASE("syclFlow.for_each.int" * doctest::timeout(300)) {
  for_each<int>();
}

TEST_CASE("syclFlow.for_each.float" * doctest::timeout(300)) {
  for_each<float>();
}

TEST_CASE("syclFlow.for_each.double" * doctest::timeout(300)) {
  for_each<double>();
}

// --------------------------------------------------------
// Testcase: for_each_index
// --------------------------------------------------------

template <typename T>
void for_each_index() {

  tf::Taskflow taskflow;
  tf::Executor executor;

  sycl::queue queue;

  for(int n=10; n<=123456; n = n*2 + 1) {

    taskflow.clear();

    T* cpu = nullptr;
    T* gpu = nullptr;

    auto cputask = taskflow.emplace([&](){
      cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
      gpu = sycl::malloc_device<T>(n, queue);
    });

    auto gputask = taskflow.emplace_on([&](tf::syclFlow& cf) {
      auto d2h = cf.copy(cpu, gpu, n);
      auto h2d = cf.copy(gpu, cpu, n);
      auto kernel1 = cf.for_each_index(
        0, n, 2,
        [gpu] (int i) { gpu[i] = 17; }
      );
      auto kernel2 = cf.for_each_index(
        1, n, 2,
        [=] (int i) { gpu[i] = -17; }
      );
      h2d.precede(kernel1, kernel2);
      d2h.succeed(kernel1, kernel2);
    }, queue);

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
    sycl::free(gpu, queue);
  }
}

TEST_CASE("syclFlow.for_each_index.int" * doctest::timeout(300)) {
  for_each_index<int>();
}

TEST_CASE("syclFlow.for_each_index.float" * doctest::timeout(300)) {
  for_each_index<float>();
}

TEST_CASE("syclFlow.for_each_index.double" * doctest::timeout(300)) {
  for_each_index<double>();
}

// ----------------------------------------------------------------------------
// reduce
// ----------------------------------------------------------------------------

template <typename T>
void reduce() {

  sycl::queue queue;
  tf::Taskflow taskflow;
  tf::Executor executor;

  for(int N=1; N<=1000000; N += (N/10+1)) {

    taskflow.clear();

    T sum = 0;

    std::vector<T> cpu(N);
    for(auto& i : cpu) {
      i = ::rand()%100-50;
      sum += i;
    }

    T sol;

    T* gpu = nullptr;
    T* res = nullptr;

    auto cputask = taskflow.emplace([&](){
      gpu = sycl::malloc_shared<T>(N, queue);
      res = sycl::malloc_shared<T>(1, queue);
    });

    tf::Task gputask;

    gputask = taskflow.emplace_on([&](tf::syclFlow& cf) {
      auto d2h = cf.copy(&sol, res, 1);
      auto h2d = cf.copy(gpu, cpu.data(), N);
      auto set = cf.single_task([res] () {
        *res = 1000;
      });
      auto kernel = cf.reduce(gpu, gpu+N, res, std::plus<T>());
      kernel.succeed(h2d, set);
      d2h.succeed(kernel);
    }, queue);

    cputask.precede(gputask);

    executor.run(taskflow).wait();

    REQUIRE(std::fabs(sum-sol+1000) < 0.0001);

    // ------------------------------------------------------------------------
    // standard algorithms
    // ------------------------------------------------------------------------
    tf::syclDefaultExecutionPolicy p{queue};

    *res = 1000;
    tf::sycl_reduce(p, gpu, gpu+N, res, std::plus<T>{});

    REQUIRE(std::fabs(sum-sol+1000) < 0.0001);

    sycl::free(gpu, queue);
    sycl::free(res, queue);
  }
}

TEST_CASE("syclFlow.reduce.int" * doctest::timeout(300)) {
  reduce<int>();
}

TEST_CASE("syclFlow.reduce.float" * doctest::timeout(300)) {
  reduce<float>();
}

TEST_CASE("syclFlow.reduce.double" * doctest::timeout(300)) {
  reduce<double>();
}

// ----------------------------------------------------------------------------
// uninitialized_reduce
// ----------------------------------------------------------------------------

template <typename T>
void uninitialized_reduce() {

  sycl::queue queue;
  tf::Taskflow taskflow;
  tf::Executor executor;

  for(int N=1; N<=1000000; N += (N/10+1)) {

    taskflow.clear();

    T sum = 0;

    std::vector<T> cpu(N);
    for(auto& i : cpu) {
      i = ::rand()%100-50;
      sum += i;
    }

    T sol;

    T* gpu = nullptr;
    T* res = nullptr;

    auto cputask = taskflow.emplace([&](){
      gpu = sycl::malloc_shared<T>(N, queue);
      res = sycl::malloc_shared<T>(1, queue);
    });

    tf::Task gputask;

    gputask = taskflow.emplace_on([&](tf::syclFlow& cf) {
      auto d2h = cf.copy(&sol, res, 1);
      auto h2d = cf.copy(gpu, cpu.data(), N);
      auto set = cf.single_task([res] () {
        *res = 1000;
      });
      auto kernel = cf.uninitialized_reduce(gpu, gpu+N, res, std::plus<T>());
      kernel.succeed(h2d, set);
      d2h.succeed(kernel);
    }, queue);

    cputask.precede(gputask);

    executor.run(taskflow).wait();

    REQUIRE(std::fabs(sum-sol) < 0.0001);

    // ------------------------------------------------------------------------
    // standard algorithms
    // ------------------------------------------------------------------------
    tf::syclDefaultExecutionPolicy p{queue};

    *res = 1000;
    tf::sycl_reduce(p, gpu, gpu+N, res, std::plus<T>{});

    REQUIRE(std::fabs(sum-sol) < 0.0001);

    sycl::free(gpu, queue);
    sycl::free(res, queue);
  }
}

TEST_CASE("syclFlow.uninitialized_reduce.int" * doctest::timeout(300)) {
  uninitialized_reduce<int>();
}

TEST_CASE("syclFlow.uninitialized_reduce.float" * doctest::timeout(300)) {
  uninitialized_reduce<float>();
}

TEST_CASE("syclFlow.uninitialized_reduce.double" * doctest::timeout(300)) {
  uninitialized_reduce<double>();
}

// ----------------------------------------------------------------------------
// transform
// ----------------------------------------------------------------------------

void transform() {

  tf::Taskflow taskflow;
  tf::Executor executor;

  sycl::queue queue;

  for(unsigned n=1; n<=123456; n = n*2 + 1) {

    taskflow.clear();

    int* htgt = nullptr;
    int* tgt = nullptr;
    int* hsrc1 = nullptr;
    int* src1 = nullptr;
    float* hsrc2 = nullptr;
    float* src2 = nullptr;
    double* hsrc3 = nullptr;
    double* src3 = nullptr;

    auto htgttask = taskflow.emplace([&](){
      htgt  = static_cast<int*>(std::calloc(n, sizeof(int)));
      hsrc1 = static_cast<int*>(std::calloc(n, sizeof(int)));
      hsrc2 = static_cast<float*>(std::calloc(n, sizeof(float)));
      hsrc3 = static_cast<double*>(std::calloc(n, sizeof(double)));
      tgt   = sycl::malloc_device<int>(n, queue);
      src1  = sycl::malloc_device<int>(n, queue);
      src2  = sycl::malloc_device<float>(n, queue);
      src3  = sycl::malloc_device<double>(n, queue);
    });

    auto gputask = taskflow.emplace_on([&](tf::syclFlow& cf) {
      auto d2h = cf.copy(htgt, tgt, n);
      auto d2h3 = cf.copy(hsrc3, src3, n);
      auto d2h2 = cf.copy(hsrc2, src2, n);
      auto d2h1 = cf.copy(hsrc1, src1, n);
      auto kernel = cf.transform(
        tgt, tgt+n,
        [] (int& v1, float& v2, double& v3) -> int {
          v1 = 1;
          v2 = 3.0f;
          v3 = 5.0;
          return 17;
        },
        src1, src2, src3
      );
      auto h2d = cf.copy(tgt, htgt, n);
      h2d.precede(kernel);
      kernel.precede(d2h, d2h1, d2h2, d2h3);
    }, queue);

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
    sycl::free(tgt, queue);
    sycl::free(src1, queue);
    sycl::free(src2, queue);
    sycl::free(src3, queue);
  }
}

TEST_CASE("syclFlow.transform" * doctest::timeout(300)) {
  transform();
}
