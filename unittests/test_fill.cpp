#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/algorithm/fill.hpp>
#include <taskflow/taskflow.hpp>

template <typename T> void fill_sync(size_t W, size_t N, T val) {
  std::vector<T> a(N);

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  taskflow.fill(a.begin(), a.end(), val);
  executor.run(taskflow).wait();

  std::vector<T> std_a(N);
  std::fill(std_a.begin(), std_a.end(), val);

  REQUIRE(a == std_a);
}

TEST_CASE("ParallelFill.int.1.10000" * doctest::timeout(300)) {
  fill_sync<int>(1, 10000, 12);
}

TEST_CASE("ParallelFill.int.2.10000" * doctest::timeout(300)) {
  fill_sync<int>(2, 10000, 12);
}

TEST_CASE("ParallelFill.int.3.10000" * doctest::timeout(300)) {
  fill_sync<int>(3, 10000, 12);
}

TEST_CASE("ParallelFill.int.4.10000" * doctest::timeout(300)) {
  fill_sync<int>(4, 10000, 12);
}

TEST_CASE("ParallelFill.ldouble.1.10000" * doctest::timeout(300)) {
  fill_sync<long double>(1, 10000, 12.);
}

TEST_CASE("ParallelFill.ldouble.2.10000" * doctest::timeout(300)) {
  fill_sync<long double>(2, 10000, 12.);
}

TEST_CASE("ParallelFill.ldouble.3.10000" * doctest::timeout(300)) {
  fill_sync<long double>(3, 10000, 12.);
}

TEST_CASE("ParallelFill.ldouble.4.10000" * doctest::timeout(300)) {
  fill_sync<long double>(4, 10000, 12.);
}

template <typename T> void fill_async(size_t W, size_t N, T val) {
  std::vector<T> a(N);

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  executor.async(tf::make_fill_task(a.begin(), a.end(), val));

  executor.wait_for_all();

  std::vector<T> std_a(N);
  std::fill(std_a.begin(), std_a.end(), val);

  REQUIRE(a == std_a);
}

TEST_CASE("ParallelFillAsync.int.1.10000" * doctest::timeout(300)) {
  fill_async<int>(1, 10000, 12);
}

TEST_CASE("ParallelFillAsync.int.2.10000" * doctest::timeout(300)) {
  fill_async<int>(2, 10000, 12);
}

TEST_CASE("ParallelFillAsync.int.3.10000" * doctest::timeout(300)) {
  fill_async<int>(3, 10000, 12);
}

TEST_CASE("ParallelFillAsync.int.4.10000" * doctest::timeout(300)) {
  fill_async<int>(4, 10000, 12);
}

TEST_CASE("ParallelFillAsync.ldouble.1.10000" * doctest::timeout(300)) {
  fill_async<long double>(1, 10000, 12.);
}

TEST_CASE("ParallelFillAsync.ldouble.2.10000" * doctest::timeout(300)) {
  fill_async<long double>(2, 10000, 12.);
}

TEST_CASE("ParallelFillAsync.ldouble.3.10000" * doctest::timeout(300)) {
  fill_async<long double>(3, 10000, 12.);
}

TEST_CASE("ParallelFillAsync.ldouble.4.10000" * doctest::timeout(300)) {
  fill_async<long double>(4, 10000, 12.);
}

template <typename T> void fill_n_sync(size_t W, size_t N, size_t C, T val) {
  std::vector<T> a(N, T(0));

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  taskflow.fill_n(a.begin(), C, val);
  executor.run(taskflow).wait();

  std::vector<T> std_a(N, T(0));
  std::fill_n(std_a.begin(), C, val);

  REQUIRE(a == std_a);
}

TEST_CASE("ParallelFillN.int.1.10000" * doctest::timeout(300)) {
  fill_n_sync<int>(1, 10000, 5000, 12);
}

TEST_CASE("ParallelFillN.int.2.10000" * doctest::timeout(300)) {
  fill_n_sync<int>(2, 10000, 5000, 12);
}

TEST_CASE("ParallelFillN.int.3.10000" * doctest::timeout(300)) {
  fill_n_sync<int>(3, 10000, 5000, 12);
}

TEST_CASE("ParallelFillN.int.4.10000" * doctest::timeout(300)) {
  fill_n_sync<int>(4, 10000, 5000, 12);
}

TEST_CASE("ParallelFillN.ldouble.1.10000" * doctest::timeout(300)) {
  fill_n_sync<long double>(1, 10000, 5000, 12.);
}

TEST_CASE("ParallelFillN.ldouble.2.10000" * doctest::timeout(300)) {
  fill_n_sync<long double>(2, 10000, 5000, 12.);
}

TEST_CASE("ParallelFillN.ldouble.3.10000" * doctest::timeout(300)) {
  fill_n_sync<long double>(3, 10000, 5000, 12.);
}

TEST_CASE("ParallelFillN.ldouble.4.10000" * doctest::timeout(300)) {
  fill_n_sync<long double>(4, 10000, 5000, 12.);
}

template <typename T> void fill_n_async(size_t W, size_t N, size_t C, T val) {
  std::vector<T> a(N, T(0));

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  executor.async(tf::make_fill_n_task(a.begin(), C, val));

  executor.wait_for_all();

  std::vector<T> std_a(N, T(0));
  std::fill_n(std_a.begin(), C, val);

  REQUIRE(a == std_a);
}

TEST_CASE("ParallelFillNAsync.int.1.10000" * doctest::timeout(300)) {
  fill_n_async<int>(1, 10000, 5000, 12);
}

TEST_CASE("ParallelFillNAsync.int.2.10000" * doctest::timeout(300)) {
  fill_n_async<int>(2, 10000, 5000, 12);
}

TEST_CASE("ParallelFillNAsync.int.3.10000" * doctest::timeout(300)) {
  fill_n_async<int>(3, 10000, 5000, 12);
}

TEST_CASE("ParallelFillNAsync.int.4.10000" * doctest::timeout(300)) {
  fill_n_async<int>(4, 10000, 5000, 12);
}

TEST_CASE("ParallelFillNAsync.ldouble.1.10000" * doctest::timeout(300)) {
  fill_n_async<long double>(1, 10000, 5000, 12.);
}

TEST_CASE("ParallelFillNAsync.ldouble.2.10000" * doctest::timeout(300)) {
  fill_n_async<long double>(2, 10000, 5000, 12.);
}

TEST_CASE("ParallelFillNAsync.ldouble.3.10000" * doctest::timeout(300)) {
  fill_n_async<long double>(3, 10000, 5000, 12.);
}

TEST_CASE("ParallelFillNAsync.ldouble.4.10000" * doctest::timeout(300)) {
  fill_n_async<long double>(4, 10000, 5000, 12.);
}

TEST_CASE("ParallelFillStateful.4" * doctest::timeout(300)) {
  std::vector<int> a;

  tf::Executor executor(4);
  tf::Taskflow taskflow;

  std::vector<int>::iterator beg;
  std::vector<int>::iterator end;
  auto init = taskflow.emplace([&]() {
    a.resize(10000, 0);
    beg = a.begin();
    end = a.end();
  });
  auto fill_task = taskflow.fill(std::ref(beg), std::ref(end), 12);
  init.precede(fill_task);

  executor.run(taskflow).wait();

  std::vector<int> std_a(10000, 0);
  std::fill(std_a.begin(), std_a.end(), 12);

  REQUIRE(a == std_a);
}

TEST_CASE("ParallelFillStatefulMixed.4" * doctest::timeout(300)) {
  std::vector<int> a(10000, 0);

  tf::Executor executor(1);
  tf::Taskflow taskflow;

  std::vector<int>::iterator beg;
  auto init = taskflow.emplace([&]() { beg = a.begin(); });
  auto fill_task = taskflow.fill(std::ref(beg), a.end(), 12);
  init.precede(fill_task);

  executor.run(taskflow).wait();

  std::vector<int> std_a(10000, 0);
  std::fill(std_a.begin(), std_a.end(), 12);

  REQUIRE(a == std_a);
}

TEST_CASE("ParallelFillNStateful.4" * doctest::timeout(300)) {
  std::vector<int> a;

  tf::Executor executor(1);
  tf::Taskflow taskflow;

  std::vector<int>::iterator beg;
  auto init = taskflow.emplace([&]() {
    a.resize(10000, 0);
    beg = a.begin();
  });
  auto fill_task = taskflow.fill_n(std::ref(beg), 5000, 12);
  init.precede(fill_task);

  executor.run(taskflow).wait();

  std::vector<int> std_a(10000, 0);
  std::fill_n(std_a.begin(), 5000, 12);

  REQUIRE(a == std_a);
}

template <typename P> void fill_partioners(size_t W, size_t N, P part) {
  std::vector<int> a(N);

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  taskflow.fill(a.begin(), a.end(), 12, part);
  executor.run(taskflow).wait();

  std::vector<int> std_a(N);
  std::fill(std_a.begin(), std_a.end(), 12);

  REQUIRE(a == std_a);

  std::vector<int> b(N);

  executor.async(tf::make_fill_task(b.begin(), b.end(), 12, part));

  executor.wait_for_all();
  std::vector<int> std_b(N);
  std::fill(std_b.begin(), std_b.end(), 12);

  REQUIRE(b == std_b);
}

TEST_CASE("ParallelFill.Static.int.1.10000" * doctest::timeout(300)) {
  fill_partioners(1, 10000, tf::StaticPartitioner<>());
}

TEST_CASE("ParallelFill.Static.int.2.10000" * doctest::timeout(300)) {
  fill_partioners(2, 10000, tf::StaticPartitioner<>());
}

TEST_CASE("ParallelFill.Static.int.3.10000" * doctest::timeout(300)) {
  fill_partioners(3, 10000, tf::StaticPartitioner<>());
}

TEST_CASE("ParallelFill.Static.int.4.10000" * doctest::timeout(300)) {
  fill_partioners(4, 10000, tf::StaticPartitioner<>());
}


TEST_CASE("ParallelFill.Dynamic.int.1.10000" * doctest::timeout(300)) {
  fill_partioners(1, 10000, tf::DynamicPartitioner<>());
}

TEST_CASE("ParallelFill.Dynamic.int.2.10000" * doctest::timeout(300)) {
  fill_partioners(2, 10000, tf::DynamicPartitioner<>());
}

TEST_CASE("ParallelFill.Dynamic.int.3.10000" * doctest::timeout(300)) {
  fill_partioners(3, 10000, tf::DynamicPartitioner<>());
}

TEST_CASE("ParallelFill.Dynamic.int.4.10000" * doctest::timeout(300)) {
  fill_partioners(4, 10000, tf::DynamicPartitioner<>());
}


TEST_CASE("ParallelFill.Random.int.1.10000" * doctest::timeout(300)) {
  fill_partioners(1, 10000, tf::RandomPartitioner<>());
}

TEST_CASE("ParallelFill.Random.int.2.10000" * doctest::timeout(300)) {
  fill_partioners(2, 10000, tf::RandomPartitioner<>());
}

TEST_CASE("ParallelFill.Random.int.3.10000" * doctest::timeout(300)) {
  fill_partioners(3, 10000, tf::RandomPartitioner<>());
}

TEST_CASE("ParallelFill.Random.int.4.10000" * doctest::timeout(300)) {
  fill_partioners(4, 10000, tf::RandomPartitioner<>());
}


TEST_CASE("ParallelFill.Default.int.1.10000" * doctest::timeout(300)) {
  fill_partioners(1, 10000, tf::DefaultPartitioner());
}

TEST_CASE("ParallelFill.Default.int.2.10000" * doctest::timeout(300)) {
  fill_partioners(2, 10000, tf::DefaultPartitioner());
}

TEST_CASE("ParallelFill.Default.int.3.10000" * doctest::timeout(300)) {
  fill_partioners(3, 10000, tf::DefaultPartitioner());
}

TEST_CASE("ParallelFill.Default.int.4.10000" * doctest::timeout(300)) {
  fill_partioners(4, 10000, tf::DefaultPartitioner());
}

template <typename P> void fill_n_partioners(size_t W, size_t N, P part) {
  std::vector<int> a(N);

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  taskflow.fill_n(a.begin(), N / 2, 12, part);
  executor.run(taskflow).wait();

  std::vector<int> std_a(N);
  std::fill_n(std_a.begin(), N / 2, 12);

  REQUIRE(a == std_a);

  std::vector<int> b(N, 0);

  executor.async(tf::make_fill_n_task(b.begin(), N / 2, 12, part));

  executor.wait_for_all();
  std::vector<int> std_b(N, 0);
  std::fill_n(std_b.begin(), N / 2, 12);

  REQUIRE(b == std_b);
}

TEST_CASE("ParallelFillN.Static.int.1.10000" * doctest::timeout(300)) {
  fill_n_partioners(1, 10000, tf::StaticPartitioner<>());
}

TEST_CASE("ParallelFillN.Static.int.2.10000" * doctest::timeout(300)) {
  fill_n_partioners(2, 10000, tf::StaticPartitioner<>());
}

TEST_CASE("ParallelFillN.Static.int.3.10000" * doctest::timeout(300)) {
  fill_n_partioners(3, 10000, tf::StaticPartitioner<>());
}

TEST_CASE("ParallelFillN.Static.int.4.10000" * doctest::timeout(300)) {
  fill_n_partioners(4, 10000, tf::StaticPartitioner<>());
}


TEST_CASE("ParallelFillN.Dynamic.int.1.10000" * doctest::timeout(300)) {
  fill_n_partioners(1, 10000, tf::DynamicPartitioner<>());
}

TEST_CASE("ParallelFillN.Dynamic.int.2.10000" * doctest::timeout(300)) {
  fill_n_partioners(2, 10000, tf::DynamicPartitioner<>());
}

TEST_CASE("ParallelFillN.Dynamic.int.3.10000" * doctest::timeout(300)) {
  fill_n_partioners(3, 10000, tf::DynamicPartitioner<>());
}

TEST_CASE("ParallelFillN.Dynamic.int.4.10000" * doctest::timeout(300)) {
  fill_n_partioners(4, 10000, tf::DynamicPartitioner<>());
}


TEST_CASE("ParallelFillN.Random.int.1.10000" * doctest::timeout(300)) {
  fill_n_partioners(1, 10000, tf::RandomPartitioner<>());
}

TEST_CASE("ParallelFillN.Random.int.2.10000" * doctest::timeout(300)) {
  fill_n_partioners(2, 10000, tf::RandomPartitioner<>());
}

TEST_CASE("ParallelFillN.Random.int.3.10000" * doctest::timeout(300)) {
  fill_n_partioners(3, 10000, tf::RandomPartitioner<>());
}

TEST_CASE("ParallelFillN.Random.int.4.10000" * doctest::timeout(300)) {
  fill_n_partioners(4, 10000, tf::RandomPartitioner<>());
}


TEST_CASE("ParallelFillN.Default.int.1.10000" * doctest::timeout(300)) {
  fill_n_partioners(1, 10000, tf::DefaultPartitioner());
}

TEST_CASE("ParallelFillN.Default.int.2.10000" * doctest::timeout(300)) {
  fill_n_partioners(2, 10000, tf::DefaultPartitioner());
}

TEST_CASE("ParallelFillN.Default.int.3.10000" * doctest::timeout(300)) {
  fill_n_partioners(3, 10000, tf::DefaultPartitioner());
}

TEST_CASE("ParallelFillN.Default.int.4.10000" * doctest::timeout(300)) {
  fill_n_partioners(4, 10000, tf::DefaultPartitioner());
}
