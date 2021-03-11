#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taskflow/syclflow.hpp>

// ----------------------------------------------------------------------------
// reduce
// ----------------------------------------------------------------------------

template <typename T>
void reduce() {
    
  sycl::queue queue;
  tf::Taskflow taskflow;
  tf::Executor executor;

  for(int n=1; n<=123456; n = n*2 + 1) {

    taskflow.clear();

    T sum = 0;

    std::vector<T> cpu(n);
    for(auto& i : cpu) {
      i = ::rand()%100-50;
      sum += i;
    }

    T sol;
    
    T* gpu = nullptr;
    T* res = nullptr;

    auto cputask = taskflow.emplace([&](){
      gpu = sycl::malloc_device<T>(n, queue);
      res = sycl::malloc_device<T>(1, queue);
    });

    tf::Task gputask;
    
    gputask = taskflow.emplace_on([&](tf::syclFlow& cf) {
      auto d2h = cf.copy(&sol, res, 1);
      auto h2d = cf.copy(gpu, cpu.data(), n);
      auto set = cf.single_task([res] () {
        *res = 1000;
      });
      auto kernel = cf.reduce(
        gpu, gpu+n, res, [] (T a, T b) { 
          return a + b;
        }
      );
      kernel.succeed(h2d, set);
      d2h.succeed(kernel);
    }, queue);

    cputask.precede(gputask);
    
    executor.run(taskflow).wait();

    REQUIRE(std::fabs(sum-sol+1000) < 0.0001);

    sycl::free(gpu, queue);
    sycl::free(res, queue);
  }
}

TEST_CASE("reduce.int" * doctest::timeout(300)) {
  reduce<int>();
}

TEST_CASE("reduce.float" * doctest::timeout(300)) {
  reduce<float>();
}

TEST_CASE("reduce.double" * doctest::timeout(300)) {
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

  for(int n=1; n<=123456; n = n*2 + 1) {

    taskflow.clear();

    T sum = 0;

    std::vector<T> cpu(n);
    for(auto& i : cpu) {
      i = ::rand()%100-50;
      sum += i;
    }

    T sol;
    
    T* gpu = nullptr;
    T* res = nullptr;

    auto cputask = taskflow.emplace([&](){
      gpu = sycl::malloc_device<T>(n, queue);
      res = sycl::malloc_device<T>(1, queue);
    });

    tf::Task gputask;
    
    gputask = taskflow.emplace_on([&](tf::syclFlow& cf) {
      auto d2h = cf.copy(&sol, res, 1);
      auto h2d = cf.copy(gpu, cpu.data(), n);
      auto set = cf.single_task([res] () {
        *res = 1000;
      });
      auto kernel = cf.uninitialized_reduce(
        gpu, gpu+n, res, [] (T a, T b) { 
          return a + b;
        }
      );
      kernel.succeed(h2d, set);
      d2h.succeed(kernel);
    }, queue);

    cputask.precede(gputask);
    
    executor.run(taskflow).wait();

    REQUIRE(std::fabs(sum-sol) < 0.0001);

    sycl::free(gpu, queue);
    sycl::free(res, queue);
  }
}

TEST_CASE("uninitialized_reduce.int" * doctest::timeout(300)) {
  uninitialized_reduce<int>();
}

TEST_CASE("uninitialized_reduce.float" * doctest::timeout(300)) {
  uninitialized_reduce<float>();
}

TEST_CASE("uninitialized_reduce.double" * doctest::timeout(300)) {
  uninitialized_reduce<double>();
}
