#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

import tf;

// --------------------------------------------------------
// Testcase: CxxModule.Basic
// Verify that tf::Taskflow and tf::Executor are accessible
// via C++20 module import.
// --------------------------------------------------------
void cxx_module_basic(unsigned W) {
  tf::Executor executor(W);
  tf::Taskflow taskflow;

  int counter = 0;

  auto A = taskflow.emplace([&counter]{ ++counter; }).name("A");
  auto B = taskflow.emplace([&counter]{ ++counter; }).name("B");
  auto C = taskflow.emplace([&counter]{ ++counter; }).name("C");

  A.precede(B);
  B.precede(C);

  executor.run(taskflow).get();
  REQUIRE(counter == 3);

  counter = 0;
  executor.run_n(taskflow, 10).get();
  REQUIRE(counter == 30);
}

TEST_CASE("CxxModule.Basic.1thread" * doctest::timeout(300)) {
  cxx_module_basic(1);
}

TEST_CASE("CxxModule.Basic.2threads" * doctest::timeout(300)) {
  cxx_module_basic(2);
}

TEST_CASE("CxxModule.Basic.4threads" * doctest::timeout(300)) {
  cxx_module_basic(4);
}

TEST_CASE("CxxModule.Basic.8threads" * doctest::timeout(300)) {
  cxx_module_basic(8);
}

// --------------------------------------------------------
// Testcase: CxxModule.ComposedOf
// Verify that composed_of (module task) works when
// imported via C++20 module.
// --------------------------------------------------------
void cxx_module_composed_of(unsigned W) {
  tf::Executor executor(W);

  tf::Taskflow sub;
  int sub_cnt = 0;
  auto s1 = sub.emplace([&sub_cnt]{ ++sub_cnt; });
  auto s2 = sub.emplace([&sub_cnt]{ ++sub_cnt; });
  s1.precede(s2);

  tf::Taskflow top;
  int top_cnt = 0;
  auto t1 = top.emplace([&top_cnt]{ ++top_cnt; });
  auto m  = top.composed_of(sub);
  t1.precede(m);

  executor.run(top).get();
  REQUIRE(top_cnt == 1);
  REQUIRE(sub_cnt == 2);
}

TEST_CASE("CxxModule.ComposedOf.1thread" * doctest::timeout(300)) {
  cxx_module_composed_of(1);
}

TEST_CASE("CxxModule.ComposedOf.2threads" * doctest::timeout(300)) {
  cxx_module_composed_of(2);
}

TEST_CASE("CxxModule.ComposedOf.4threads" * doctest::timeout(300)) {
  cxx_module_composed_of(4);
}

TEST_CASE("CxxModule.ComposedOf.8threads" * doctest::timeout(300)) {
  cxx_module_composed_of(8);
}

// --------------------------------------------------------
// Testcase: CxxModule.Async
// Verify that async task submission works when
// imported via C++20 module.
// --------------------------------------------------------
void cxx_module_async(unsigned W) {
  tf::Executor executor(W);

  std::atomic<int> counter{0};

  std::vector<tf::Future<void>> futures;
  for (int i = 0; i < 100; ++i) {
    futures.emplace_back(executor.async([&counter]{ counter.fetch_add(1); }));
  }

  for (auto& f : futures) {
    f.get();
  }

  REQUIRE(counter.load() == 100);
}

TEST_CASE("CxxModule.Async.1thread" * doctest::timeout(300)) {
  cxx_module_async(1);
}

TEST_CASE("CxxModule.Async.2threads" * doctest::timeout(300)) {
  cxx_module_async(2);
}

TEST_CASE("CxxModule.Async.4threads" * doctest::timeout(300)) {
  cxx_module_async(4);
}

TEST_CASE("CxxModule.Async.8threads" * doctest::timeout(300)) {
  cxx_module_async(8);
}

// --------------------------------------------------------
// Testcase: CxxModule.Semaphore
// Verify that tf::Semaphore is accessible via module import
// and correctly limits concurrency.
// --------------------------------------------------------
void cxx_module_semaphore(unsigned W) {
  tf::Executor executor(W);
  tf::Taskflow taskflow;
  tf::Semaphore semaphore(1);

  std::atomic<int> concurrent{0};
  std::atomic<int> max_concurrent{0};

  for (int i = 0; i < 16; ++i) {
    taskflow.emplace([&concurrent, &max_concurrent]{
      int c = concurrent.fetch_add(1) + 1;
      int expected = max_concurrent.load();
      while (c > expected && !max_concurrent.compare_exchange_weak(expected, c)) {}
      concurrent.fetch_sub(1);
    }).acquire(semaphore).release(semaphore);
  }

  executor.run(taskflow).get();
  REQUIRE(max_concurrent.load() <= 1);
}

TEST_CASE("CxxModule.Semaphore.1thread" * doctest::timeout(300)) {
  cxx_module_semaphore(1);
}

TEST_CASE("CxxModule.Semaphore.2threads" * doctest::timeout(300)) {
  cxx_module_semaphore(2);
}

TEST_CASE("CxxModule.Semaphore.4threads" * doctest::timeout(300)) {
  cxx_module_semaphore(4);
}

TEST_CASE("CxxModule.Semaphore.8threads" * doctest::timeout(300)) {
  cxx_module_semaphore(8);
}
