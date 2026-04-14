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

TEST_CASE("ParallelGenerateN.int.1.10000" * doctest::timeout(300)){
  generate_n_sync<int>(1, 10000, 5000, 12);
}

TEST_CASE("ParallelGenerateN.int.2.10000" * doctest::timeout(300)){
  generate_n_sync<int>(2, 10000, 5000, 12);
}

TEST_CASE("ParallelGenerateN.int.3.10000" * doctest::timeout(300)){
  generate_n_sync<int>(3, 10000, 5000, 12);
}

TEST_CASE("ParallelGenerateN.int.4.10000" * doctest::timeout(300)){
  generate_n_sync<int>(4, 10000, 5000, 12);
}

TEST_CASE("ParallelGenerateN.ldouble.1.10000" * doctest::timeout(300)){
  generate_n_sync<long double>(1, 10000, 5000, 12.);
}

TEST_CASE("ParallelGenerateN.ldouble.2.10000" * doctest::timeout(300)){
  generate_n_sync<long double>(2, 10000, 5000, 12.);
}

TEST_CASE("ParallelGenerateN.ldouble.3.10000" * doctest::timeout(300)){
  generate_n_sync<long double>(3, 10000, 5000, 12.);
}

TEST_CASE("ParallelGenerateN.ldouble.4.10000" * doctest::timeout(300)){
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

TEST_CASE("ParallelGenerateNAsync.int.1.10000" * doctest::timeout(300)){
  generate_n_async<int>(1, 10000, 5000, 12);
}

TEST_CASE("ParallelGenerateNAsync.int.2.10000" * doctest::timeout(300)){
  generate_n_async<int>(2, 10000, 5000, 12);
}

TEST_CASE("ParallelGenerateNAsync.int.3.10000" * doctest::timeout(300)){
  generate_n_async<int>(3, 10000, 5000, 12);
}

TEST_CASE("ParallelGenerateNAsync.int.4.10000" * doctest::timeout(300)){
  generate_n_async<int>(4, 10000, 5000, 12);
}

TEST_CASE("ParallelGenerateNAsync.ldouble.1.10000" * doctest::timeout(300)){
  generate_n_async<long double>(1, 10000, 5000, 12.);
}

TEST_CASE("ParallelGenerateNAsync.ldouble.2.10000" * doctest::timeout(300)){
  generate_n_async<long double>(2, 10000, 5000, 12.);
}

TEST_CASE("ParallelGenerateNAsync.ldouble.3.10000" * doctest::timeout(300)){
  generate_n_async<long double>(3, 10000, 5000, 12.);
}

TEST_CASE("ParallelGenerateNAsync.ldouble.4.10000" * doctest::timeout(300)){
  generate_n_async<long double>(4, 10000, 5000, 12.);
}

TEST_CASE("ParallelGenerateStateful.4" * doctest::timeout(300)) {
  auto gen = [](){
    return 12;
  };
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
  auto generate_task = taskflow.generate(std::ref(beg), std::ref(end), gen);
  init.precede(generate_task);

  executor.run(taskflow).wait();

  std::vector<int> std_a(10000, 0);
  std::generate(std_a.begin(), std_a.end(), gen);

  REQUIRE(a == std_a);
}

TEST_CASE("ParallelGenerateStatefulMixed.4" * doctest::timeout(300)) {
  auto gen = [](){
    return 12;
  };
  std::vector<int> a(10000, 0);

  tf::Executor executor(4);
  tf::Taskflow taskflow;

  std::vector<int>::iterator beg;
  auto init = taskflow.emplace([&]() {
    beg = a.begin();
  });
  auto generate_task = taskflow.generate(std::ref(beg), a.end(), gen);
  init.precede(generate_task);

  executor.run(taskflow).wait();

  std::vector<int> std_a(10000, 0);
  std::generate(std_a.begin(), std_a.end(), gen);

  REQUIRE(a == std_a);
}

TEST_CASE("ParallelGenerateNStateful.1" * doctest::timeout(300)) {
  auto gen = [](){
    return 12;
  };
  std::vector<int> a;

  tf::Executor executor(4);
  tf::Taskflow taskflow;

  std::vector<int>::iterator beg;
  auto init = taskflow.emplace([&]() {
    a.resize(10000, 0);
    beg = a.begin();
  });
  auto generate_task = taskflow.generate_n(std::ref(beg), 5000, gen);
  init.precede(generate_task);

  executor.run(taskflow).wait();

  std::vector<int> std_a(10000, 0);
  std::generate_n(std_a.begin(), 5000, gen);

  REQUIRE(a == std_a);
}

template <typename P> void generate_partioners(size_t W, size_t N, P part) {
   auto gen = [](){
    return 12;
  };
  std::vector<int> a(N);

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  taskflow.generate(a.begin(), a.end(), gen, part);
  executor.run(taskflow).wait();

  std::vector<int> std_a(N);
  std::generate(std_a.begin(), std_a.end(), gen);

  REQUIRE(a == std_a);

  std::vector<int> b(N);

  executor.async(tf::make_generate_task(b.begin(), b.end(), gen, part));

  executor.wait_for_all();
  std::vector<int> std_b(N);
  std::generate(std_b.begin(), std_b.end(), gen);

  REQUIRE(b == std_b);
}

TEST_CASE("ParallelGenerate.Static.int.1.10000" * doctest::timeout(300)) {
  generate_partioners(1, 10000, tf::StaticPartitioner<>());
}

TEST_CASE("ParallelGenerate.Static.int.2.10000" * doctest::timeout(300)) {
  generate_partioners(2, 10000, tf::StaticPartitioner<>());
}

TEST_CASE("ParallelGenerate.Static.int.3.10000" * doctest::timeout(300)) {
  generate_partioners(3, 10000, tf::StaticPartitioner<>());
}

TEST_CASE("ParallelGenerate.Static.int.4.10000" * doctest::timeout(300)) {
  generate_partioners(4, 10000, tf::StaticPartitioner<>());
}


TEST_CASE("ParallelGenerate.Dynamic.int.1.10000" * doctest::timeout(300)) {
  generate_partioners(1, 10000, tf::DynamicPartitioner<>());
}

TEST_CASE("ParallelGenerate.Dynamic.int.2.10000" * doctest::timeout(300)) {
  generate_partioners(2, 10000, tf::DynamicPartitioner<>());
}

TEST_CASE("ParallelGenerate.Dynamic.int.3.10000" * doctest::timeout(300)) {
  generate_partioners(3, 10000, tf::DynamicPartitioner<>());
}

TEST_CASE("ParallelGenerate.Dynamic.int.4.10000" * doctest::timeout(300)) {
  generate_partioners(4, 10000, tf::DynamicPartitioner<>());
}


TEST_CASE("ParallelGenerate.Random.int.1.10000" * doctest::timeout(300)) {
  generate_partioners(1, 10000, tf::RandomPartitioner<>());
}

TEST_CASE("ParallelGenerate.Random.int.2.10000" * doctest::timeout(300)) {
  generate_partioners(2, 10000, tf::RandomPartitioner<>());
}

TEST_CASE("ParallelGenerate.Random.int.3.10000" * doctest::timeout(300)) {
  generate_partioners(3, 10000, tf::RandomPartitioner<>());
}

TEST_CASE("ParallelGenerate.Random.int.4.10000" * doctest::timeout(300)) {
  generate_partioners(4, 10000, tf::RandomPartitioner<>());
}


TEST_CASE("ParallelGenerate.Default.int.1.10000" * doctest::timeout(300)) {
  generate_partioners(1, 10000, tf::DefaultPartitioner());
}

TEST_CASE("ParallelGenerate.Default.int.2.10000" * doctest::timeout(300)) {
  generate_partioners(2, 10000, tf::DefaultPartitioner());
}

TEST_CASE("ParallelGenerate.Default.int.3.10000" * doctest::timeout(300)) {
  generate_partioners(3, 10000, tf::DefaultPartitioner());
}

TEST_CASE("ParallelGenerate.Default.int.4.10000" * doctest::timeout(300)) {
  generate_partioners(4, 10000, tf::DefaultPartitioner());
}

template <typename P> void generate_n_partioners(size_t W, size_t N, P part) {
   auto gen = [](){
    return 12;
  };
  std::vector<int> a(N);

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  taskflow.generate_n(a.begin(), N / 2, gen, part);
  executor.run(taskflow).wait();

  std::vector<int> std_a(N);
  std::generate_n(std_a.begin(), N / 2, gen);

  REQUIRE(a == std_a);

  std::vector<int> b(N, 0);

  executor.async(tf::make_generate_n_task(b.begin(), N / 2, gen, part));

  executor.wait_for_all();
  std::vector<int> std_b(N, 0);
  std::generate_n(std_b.begin(), N / 2, gen);

  REQUIRE(b == std_b);
}

TEST_CASE("ParallelGenerateN.Static.int.1.10000" * doctest::timeout(300)) {
  generate_n_partioners(1, 10000, tf::StaticPartitioner<>());
}

TEST_CASE("ParallelGenerateN.Static.int.2.10000" * doctest::timeout(300)) {
  generate_n_partioners(2, 10000, tf::StaticPartitioner<>());
}

TEST_CASE("ParallelGenerateN.Static.int.3.10000" * doctest::timeout(300)) {
  generate_n_partioners(3, 10000, tf::StaticPartitioner<>());
}

TEST_CASE("ParallelGenerateN.Static.int.4.10000" * doctest::timeout(300)) {
  generate_n_partioners(4, 10000, tf::StaticPartitioner<>());
}


TEST_CASE("ParallelGenerateN.Dynamic.int.1.10000" * doctest::timeout(300)) {
  generate_n_partioners(1, 10000, tf::DynamicPartitioner<>());
}

TEST_CASE("ParallelGenerateN.Dynamic.int.2.10000" * doctest::timeout(300)) {
  generate_n_partioners(2, 10000, tf::DynamicPartitioner<>());
}

TEST_CASE("ParallelGenerateN.Dynamic.int.3.10000" * doctest::timeout(300)) {
  generate_n_partioners(3, 10000, tf::DynamicPartitioner<>());
}

TEST_CASE("ParallelGenerateN.Dynamic.int.4.10000" * doctest::timeout(300)) {
  generate_n_partioners(4, 10000, tf::DynamicPartitioner<>());
}


TEST_CASE("ParallelGenerateN.Random.int.1.10000" * doctest::timeout(300)) {
  generate_n_partioners(1, 10000, tf::RandomPartitioner<>());
}

TEST_CASE("ParallelGenerateN.Random.int.2.10000" * doctest::timeout(300)) {
  generate_n_partioners(2, 10000, tf::RandomPartitioner<>());
}

TEST_CASE("ParallelGenerateN.Random.int.3.10000" * doctest::timeout(300)) {
  generate_n_partioners(3, 10000, tf::RandomPartitioner<>());
}

TEST_CASE("ParallelGenerateN.Random.int.4.10000" * doctest::timeout(300)) {
  generate_n_partioners(4, 10000, tf::RandomPartitioner<>());
}


TEST_CASE("ParallelGenerateN.Default.int.1.10000" * doctest::timeout(300)) {
  generate_n_partioners(1, 10000, tf::DefaultPartitioner());
}

TEST_CASE("ParallelGenerateN.Default.int.2.10000" * doctest::timeout(300)) {
  generate_n_partioners(2, 10000, tf::DefaultPartitioner());
}

TEST_CASE("ParallelGenerateN.Default.int.3.10000" * doctest::timeout(300)) {
  generate_n_partioners(3, 10000, tf::DefaultPartitioner());
}

TEST_CASE("ParallelGenerateN.Default.int.4.10000" * doctest::timeout(300)) {
  generate_n_partioners(4, 10000, tf::DefaultPartitioner());
}
