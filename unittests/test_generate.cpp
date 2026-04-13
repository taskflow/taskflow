#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/algorithm/generate.hpp>
#include <taskflow/taskflow.hpp>

template <typename T> void generate_sync(size_t W, size_t N, T val) {
  auto gen = [val](){
    return val;
  };
  std::vector<T> a(N);

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  taskflow.generate(a.begin(), a.end(), gen);
  executor.run(taskflow).wait();

  std::vector<T> std_a(N);
  std::generate(std_a.begin(), std_a.end(), gen);

  REQUIRE(a == std_a);
}

TEST_CASE("ParallelGenerate.int.1.10000" * doctest::timeout(300)){
  generate_sync<int>(1, 10000, 12);
}

TEST_CASE("ParallelGenerate.int.2.10000" * doctest::timeout(300)){
  generate_sync<int>(2, 10000, 12);
}

TEST_CASE("ParallelGenerate.int.3.10000" * doctest::timeout(300)){
  generate_sync<int>(3, 10000, 12);
}

TEST_CASE("ParallelGenerate.int.4.10000" * doctest::timeout(300)){
  generate_sync<int>(4, 10000, 12);
}

TEST_CASE("ParallelGenerate.ldouble.1.10000" * doctest::timeout(300)){
  generate_sync<long double>(1, 10000, 12.);
}

TEST_CASE("ParallelGenerate.ldouble.2.10000" * doctest::timeout(300)){
  generate_sync<long double>(2, 10000, 12.);
}

TEST_CASE("ParallelGenerate.ldouble.3.10000" * doctest::timeout(300)){
  generate_sync<long double>(3, 10000, 12.);
}

TEST_CASE("ParallelGenerate.ldouble.4.10000" * doctest::timeout(300)){
  generate_sync<long double>(4, 10000, 12.);
}

template <typename T> void generate_async(size_t W, size_t N, T val) {
  auto gen = [val](){
    return val;
  };
  std::vector<T> a(N);

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  executor.async(tf::make_generate_task(a.begin(), a.end(), gen));

  executor.wait_for_all();

  std::vector<T> std_a(N);
  std::generate(std_a.begin(), std_a.end(), gen);

  REQUIRE(a == std_a);
}

TEST_CASE("ParallelGenerateAsync.int.1.10000" * doctest::timeout(300)){
  generate_async<int>(1, 10000, 12);
}

TEST_CASE("ParallelGenerateAsync.int.2.10000" * doctest::timeout(300)){
  generate_async<int>(2, 10000, 12);
}

TEST_CASE("ParallelGenerateAsync.int.3.10000" * doctest::timeout(300)){
  generate_async<int>(3, 10000, 12);
}

TEST_CASE("ParallelGenerateAsync.int.4.10000" * doctest::timeout(300)){
  generate_async<int>(4, 10000, 12);
}

TEST_CASE("ParallelGenerateAsync.ldouble.1.10000" * doctest::timeout(300)){
  generate_async<long double>(1, 10000, 12.);
}

TEST_CASE("ParallelGenerateAsync.ldouble.2.10000" * doctest::timeout(300)){
  generate_async<long double>(2, 10000, 12.);
}

TEST_CASE("ParallelGenerateAsync.ldouble.3.10000" * doctest::timeout(300)){
  generate_async<long double>(3, 10000, 12.);
}

TEST_CASE("ParallelGenerateAsync.ldouble.4.10000" * doctest::timeout(300)){
  generate_async<long double>(4, 10000, 12.);
}

template <typename T> void generate_n_sync(size_t W, size_t N, size_t C, T val) {
  auto gen = [val](){
    return val;
  };
  std::vector<T> a(N, T(0));

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  taskflow.generate_n(a.begin(), C, gen);
  executor.run(taskflow).wait();

  std::vector<T> std_a(N, T(0));
  std::generate_n(std_a.begin(), C, gen);

  REQUIRE(a == std_a);
}

TEST_CASE("ParallelGenerate.int.1.10000" * doctest::timeout(300)){
  generate_n_sync<int>(1, 10000, 5000, 12);
}

TEST_CASE("ParallelGenerate.int.2.10000" * doctest::timeout(300)){
  generate_n_sync<int>(2, 10000, 5000, 12);
}

TEST_CASE("ParallelGenerate.int.3.10000" * doctest::timeout(300)){
  generate_n_sync<int>(3, 10000, 5000, 12);
}

TEST_CASE("ParallelGenerate.int.4.10000" * doctest::timeout(300)){
  generate_n_sync<int>(4, 10000, 5000, 12);
}

TEST_CASE("ParallelGenerate.ldouble.1.10000" * doctest::timeout(300)){
  generate_n_sync<long double>(1, 10000, 5000, 12.);
}

TEST_CASE("ParallelGenerate.ldouble.2.10000" * doctest::timeout(300)){
  generate_n_sync<long double>(2, 10000, 5000, 12.);
}

TEST_CASE("ParallelGenerate.ldouble.3.10000" * doctest::timeout(300)){
  generate_n_sync<long double>(3, 10000, 5000, 12.);
}

TEST_CASE("ParallelGenerate.ldouble.4.10000" * doctest::timeout(300)){
  generate_n_sync<long double>(4, 10000, 5000, 12.);
}

template <typename T> void generate_n_async(size_t W, size_t N, size_t C, T val) {
  auto gen = [val](){
    return val;
  };
  std::vector<T> a(N, T(0));

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  executor.async(tf::make_generate_n_task(a.begin(), C, gen));

  executor.wait_for_all();

  std::vector<T> std_a(N, T(0));
  std::generate_n(std_a.begin(), C, gen);

  REQUIRE(a == std_a);
}

TEST_CASE("ParallelGenerateAsync.int.1.10000" * doctest::timeout(300)){
  generate_n_async<int>(1, 10000, 5000, 12);
}

TEST_CASE("ParallelGenerateAsync.int.2.10000" * doctest::timeout(300)){
  generate_n_async<int>(2, 10000, 5000, 12);
}

TEST_CASE("ParallelGenerateAsync.int.3.10000" * doctest::timeout(300)){
  generate_n_async<int>(3, 10000, 5000, 12);
}

TEST_CASE("ParallelGenerateAsync.int.4.10000" * doctest::timeout(300)){
  generate_n_async<int>(4, 10000, 5000, 12);
}

TEST_CASE("ParallelGenerateAsync.ldouble.1.10000" * doctest::timeout(300)){
  generate_n_async<long double>(1, 10000, 5000, 12.);
}

TEST_CASE("ParallelGenerateAsync.ldouble.2.10000" * doctest::timeout(300)){
  generate_n_async<long double>(2, 10000, 5000, 12.);
}

TEST_CASE("ParallelGenerateAsync.ldouble.3.10000" * doctest::timeout(300)){
  generate_n_async<long double>(3, 10000, 5000, 12.);
}

TEST_CASE("ParallelGenerateAsync.ldouble.4.10000" * doctest::timeout(300)){
  generate_n_async<long double>(4, 10000, 5000, 12.);
}