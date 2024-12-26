#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/module.hpp>

// --------------------------------------------------------
// Testcase: Module
// --------------------------------------------------------
void module1(unsigned W) {

  tf::Executor executor(W);

  tf::Taskflow f0;

  int cnt {0};

  auto A = f0.emplace([&cnt](){ ++cnt; });
  auto B = f0.emplace([&cnt](){ ++cnt; });
  auto C = f0.emplace([&cnt](){ ++cnt; });
  auto D = f0.emplace([&cnt](){ ++cnt; });
  auto E = f0.emplace([&cnt](){ ++cnt; });

  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);

  tf::Taskflow f1;

  // module 1
  std::tie(A, B, C, D, E) = f1.emplace(
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; }
  );
  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);
  auto m1_1 = f1.composed_of(f0);
  E.precede(m1_1);

  executor.run(f1).get();
  REQUIRE(cnt == 10);

  cnt = 0;
  executor.run_n(f1, 100).get();
  REQUIRE(cnt == 10 * 100);

  auto m1_2 = f1.composed_of(f0);
  m1_1.precede(m1_2);

  for(int n=0; n<100; n++) {
    cnt = 0;
    executor.run_n(f1, n).get();
    REQUIRE(cnt == 15*n);
  }

  cnt = 0;
  for(int n=0; n<100; n++) {
    executor.run(f1);
  }

  executor.wait_for_all();

  REQUIRE(cnt == 1500);
}

TEST_CASE("Module1.1thread" * doctest::timeout(300)) {
  module1(1);
}

TEST_CASE("Module1.2threads" * doctest::timeout(300)) {
  module1(2);
}

TEST_CASE("Module1.3threads" * doctest::timeout(300)) {
  module1(3);
}

TEST_CASE("Module1.4threads" * doctest::timeout(300)) {
  module1(4);
}

TEST_CASE("Module1.5threads" * doctest::timeout(300)) {
  module1(5);
}

TEST_CASE("Module1.6threads" * doctest::timeout(300)) {
  module1(6);
}

TEST_CASE("Module1.7threads" * doctest::timeout(300)) {
  module1(7);
}

TEST_CASE("Module1.8threads" * doctest::timeout(300)) {
  module1(8);
}

// ----------------------------------------------------------------------------
// Module 2
// ----------------------------------------------------------------------------

// TESTCASE: module-2
void module2(unsigned W) {

  tf::Executor executor(W);

  int cnt {0};

  // level 0 (+5)
  tf::Taskflow f0;

  auto A = f0.emplace([&cnt](){ ++cnt; }).name("f0A");
  auto B = f0.emplace([&cnt](){ ++cnt; }).name("f0B");
  auto C = f0.emplace([&cnt](){ ++cnt; }).name("f0C");
  auto D = f0.emplace([&cnt](){ ++cnt; }).name("f0D");
  auto E = f0.emplace([&cnt](){ ++cnt; }).name("f0E");

  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);

  // level 1 (+10)
  tf::Taskflow f1;
  auto m1_1 = f1.composed_of(f0).name("m1_1");
  auto m1_2 = f1.composed_of(f0).name("m1_2");
  m1_1.precede(m1_2);

  // level 2 (+20)
  tf::Taskflow f2;
  auto m2_1 = f2.composed_of(f1).name("m2_1");
  auto m2_2 = f2.composed_of(f1).name("m2_2");
  m2_1.precede(m2_2);

  //f2.dump(std::cout);

  // synchronous run
  for(int n=0; n<100; n++) {
    cnt = 0;
    executor.run_n(f2, n).get();
    REQUIRE(cnt == 20*n);
  }

  // asynchronous run
  cnt = 0;
  for(int n=0; n<100; n++) {
    executor.run(f2);
  }
  executor.wait_for_all();
  REQUIRE(cnt == 100*20);
 
}

TEST_CASE("Module2.1thread" * doctest::timeout(300)) {
  module2(1);
}

TEST_CASE("Module2.2threads" * doctest::timeout(300)) {
  module2(2);
}

TEST_CASE("Module2.3threads" * doctest::timeout(300)) {
  module2(3);
}

TEST_CASE("Module2.4threads" * doctest::timeout(300)) {
  module2(4);
}

TEST_CASE("Module2.5threads" * doctest::timeout(300)) {
  module2(5);
}

TEST_CASE("Module2.6threads" * doctest::timeout(300)) {
  module2(6);
}

TEST_CASE("Module2.7threads" * doctest::timeout(300)) {
  module2(7);
}

TEST_CASE("Module2.8threads" * doctest::timeout(300)) {
  module2(8);
}

// ----------------------------------------------------------------------------
// Module 3
// ----------------------------------------------------------------------------

// TESTCASE: module-3
void module3(unsigned W) {

  tf::Executor executor(W);

  int cnt {0};

  // level 0 (+2)
  tf::Taskflow f0;

  auto A = f0.emplace([&cnt](){ ++cnt; });
  auto B = f0.emplace([&cnt](){ ++cnt; });

  A.precede(B);

  // level 1 (+4)
  tf::Taskflow f1;
  auto m1_1 = f1.composed_of(f0);
  auto m1_2 = f1.composed_of(f0);
  m1_1.precede(m1_2);

  // level 2 (+8)
  tf::Taskflow f2;
  auto m2_1 = f2.composed_of(f1);
  auto m2_2 = f2.composed_of(f1);
  m2_1.precede(m2_2);

  // level 3 (+16)
  tf::Taskflow f3;
  auto m3_1 = f3.composed_of(f2);
  auto m3_2 = f3.composed_of(f2);
  m3_1.precede(m3_2);

  // synchronous run
  for(int n=0; n<100; n++) {
    cnt = 0;
    executor.run_n(f3, n).get();
    REQUIRE(cnt == 16*n);
  }

  // asynchronous run
  cnt = 0;
  for(int n=0; n<100; n++) {
    executor.run(f3);
  }
  executor.wait_for_all();
  REQUIRE(cnt == 16*100);
 
}

TEST_CASE("Module3.1thread" * doctest::timeout(300)) {
  module3(1);
}

TEST_CASE("Module3.2threads" * doctest::timeout(300)) {
  module3(2);
}

TEST_CASE("Module3.3threads" * doctest::timeout(300)) {
  module3(3);
}

TEST_CASE("Module3.4threads" * doctest::timeout(300)) {
  module3(4);
}

TEST_CASE("Module3.5threads" * doctest::timeout(300)) {
  module3(5);
}

TEST_CASE("Module3.6threads" * doctest::timeout(300)) {
  module3(6);
}

TEST_CASE("Module3.7threads" * doctest::timeout(300)) {
  module3(7);
}

TEST_CASE("Module3.8threads" * doctest::timeout(300)) {
  module3(8);
}

// ----------------------------------------------------------------------------
// Module Algorithm with Taskflow Launch
// ----------------------------------------------------------------------------

void module4(unsigned W) {

  tf::Executor executor(W);

  tf::Taskflow f0;

  int cnt {0};

  auto A = f0.emplace([&cnt](){ ++cnt; });
  auto B = f0.emplace([&cnt](){ ++cnt; });
  auto C = f0.emplace([&cnt](){ ++cnt; });
  auto D = f0.emplace([&cnt](){ ++cnt; });
  auto E = f0.emplace([&cnt](){ ++cnt; });

  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);

  tf::Taskflow f1;

  // module 1
  std::tie(A, B, C, D, E) = f1.emplace(
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; }
  );
  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);
  auto m1_1 = f1.emplace(tf::make_module_task(f0));
  E.precede(m1_1);

  executor.run(f1).get();
  REQUIRE(cnt == 10);

  cnt = 0;
  executor.run_n(f1, 100).get();
  REQUIRE(cnt == 10 * 100);

  auto m1_2 = f1.emplace(tf::make_module_task(f0));
  m1_1.precede(m1_2);

  for(int n=0; n<100; n++) {
    cnt = 0;
    executor.run_n(f1, n).get();
    REQUIRE(cnt == 15*n);
  }

  cnt = 0;
  for(int n=0; n<100; n++) {
    executor.run(f1);
  }

  executor.wait_for_all();

  REQUIRE(cnt == 1500);
}

TEST_CASE("Module4.1thread" * doctest::timeout(300)) {
  module4(1);
}

TEST_CASE("Module4.2threads" * doctest::timeout(300)) {
  module4(2);
}

TEST_CASE("Module4.3threads" * doctest::timeout(300)) {
  module4(3);
}

TEST_CASE("Module4.4threads" * doctest::timeout(300)) {
  module4(4);
}

TEST_CASE("Module4.5threads" * doctest::timeout(300)) {
  module4(5);
}

TEST_CASE("Module4.6threads" * doctest::timeout(300)) {
  module4(6);
}

TEST_CASE("Module4.7threads" * doctest::timeout(300)) {
  module4(7);
}

TEST_CASE("Module4.8threads" * doctest::timeout(300)) {
  module4(8);
}

// ----------------------------------------------------------------------------
// Parallel Modules
// ----------------------------------------------------------------------------

void parallel_modules(unsigned W) {

  std::vector<tf::Taskflow> taskflows(100);

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::atomic<int> counter{0};

  for(auto& tf : taskflows) {
    for(size_t n=0; n<100; n++) {
      auto [A, B, C, D, E, F, G, H] = tf.emplace(
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); },
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); },
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); },
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); },
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); },
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); },
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); },
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); }
      );
      A.precede(B);
      A.precede(C);
      D.precede(E);
      D.precede(F);
    }
    taskflow.composed_of(tf);
  }

  executor.run(taskflow).wait();

  REQUIRE(counter == 80000);
}

TEST_CASE("ParallelModules.1thread" * doctest::timeout(300)) {
  parallel_modules(1);
}

TEST_CASE("ParallelModules.2threads" * doctest::timeout(300)) {
  parallel_modules(2);
}

TEST_CASE("ParallelModules.3thread" * doctest::timeout(300)) {
  parallel_modules(3);
}

TEST_CASE("ParallelModules.4thread" * doctest::timeout(300)) {
  parallel_modules(4);
}


// ----------------------------------------------------------------------------
// Module with Async Launch
// ----------------------------------------------------------------------------

void module_with_async_launch(unsigned W) {

  tf::Executor executor(W);

  tf::Taskflow f0;

  int cnt {0};

  auto A = f0.emplace([&cnt](){ ++cnt; });
  auto B = f0.emplace([&cnt](){ ++cnt; });
  auto C = f0.emplace([&cnt](){ ++cnt; });
  auto D = f0.emplace([&cnt](){ ++cnt; });
  auto E = f0.emplace([&cnt](){ ++cnt; });

  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);

  tf::Taskflow f1;

  // module 1
  std::tie(A, B, C, D, E) = f1.emplace(
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; }
  );
  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);
  auto m1_1 = f1.composed_of(f0);
  E.precede(m1_1);

  executor.async(tf::make_module_task(f1)).get(); 

  REQUIRE(cnt == 10);
}

TEST_CASE("Module.AsyncLaunch.1thread" * doctest::timeout(300)) {
  module_with_async_launch(1);
}

TEST_CASE("Module.AsyncLaunch.2threads" * doctest::timeout(300)) {
  module_with_async_launch(2);
}

TEST_CASE("Module.AsyncLaunch.3threads" * doctest::timeout(300)) {
  module_with_async_launch(3);
}

TEST_CASE("Module.AsyncLaunch.4threads" * doctest::timeout(300)) {
  module_with_async_launch(4);
}

TEST_CASE("Module.AsyncLaunch.5threads" * doctest::timeout(300)) {
  module_with_async_launch(5);
}

TEST_CASE("Module.AsyncLaunch.6threads" * doctest::timeout(300)) {
  module_with_async_launch(6);
}

TEST_CASE("Module.AsyncLaunch.7threads" * doctest::timeout(300)) {
  module_with_async_launch(7);
}

TEST_CASE("Module.AsyncLaunch.8threads" * doctest::timeout(300)) {
  module_with_async_launch(8);
}

// ----------------------------------------------------------------------------
// Module with Silent Async Launch
// ----------------------------------------------------------------------------

void module_with_silent_async_launch(unsigned W) {

  tf::Executor executor(W);

  tf::Taskflow f0;

  int cnt {0};

  auto A = f0.emplace([&cnt](){ ++cnt; });
  auto B = f0.emplace([&cnt](){ ++cnt; });
  auto C = f0.emplace([&cnt](){ ++cnt; });
  auto D = f0.emplace([&cnt](){ ++cnt; });
  auto E = f0.emplace([&cnt](){ ++cnt; });

  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);

  tf::Taskflow f1;

  // module 1
  std::tie(A, B, C, D, E) = f1.emplace(
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; }
  );
  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);
  auto m1_1 = f1.composed_of(f0);
  E.precede(m1_1);

  executor.silent_async(tf::make_module_task(f1)); 
  executor.wait_for_all();

  REQUIRE(cnt == 10);
}

TEST_CASE("Module.SilentAsyncLaunch.1thread" * doctest::timeout(300)) {
  module_with_silent_async_launch(1);
}

TEST_CASE("Module.SilentAsyncLaunch.2threads" * doctest::timeout(300)) {
  module_with_silent_async_launch(2);
}

TEST_CASE("Module.SilentAsyncLaunch.3threads" * doctest::timeout(300)) {
  module_with_silent_async_launch(3);
}

TEST_CASE("Module.SilentAsyncLaunch.4threads" * doctest::timeout(300)) {
  module_with_silent_async_launch(4);
}

TEST_CASE("Module.SilentAsyncLaunch.5threads" * doctest::timeout(300)) {
  module_with_silent_async_launch(5);
}

TEST_CASE("Module.SilentAsyncLaunch.6threads" * doctest::timeout(300)) {
  module_with_silent_async_launch(6);
}

TEST_CASE("Module.SilentAsyncLaunch.7threads" * doctest::timeout(300)) {
  module_with_silent_async_launch(7);
}

TEST_CASE("Module.SilentAsyncLaunch.8threads" * doctest::timeout(300)) {
  module_with_silent_async_launch(8);
}

// ----------------------------------------------------------------------------
// Module with Dependent Async Launch
// ----------------------------------------------------------------------------

void module_with_dependent_async_launch(unsigned W) {

  tf::Executor executor(W);

  tf::Taskflow f0;

  int cnt {0};

  auto A = f0.emplace([&cnt](){ ++cnt; });
  auto B = f0.emplace([&cnt](){ ++cnt; });
  auto C = f0.emplace([&cnt](){ ++cnt; });
  auto D = f0.emplace([&cnt](){ ++cnt; });
  auto E = f0.emplace([&cnt](){ ++cnt; });

  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);

  tf::Taskflow f1;

  // module 1
  std::tie(A, B, C, D, E) = f1.emplace(
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; }
  );
  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);
  auto m1_1 = f1.composed_of(f0);
  E.precede(m1_1);

  auto [task, future] = executor.dependent_async(tf::make_module_task(f1)); 

  future.get();

  REQUIRE(cnt == 10);
}

TEST_CASE("Module.DependentAsyncLaunch.1thread" * doctest::timeout(300)) {
  module_with_dependent_async_launch(1);
}

TEST_CASE("Module.DependentAsyncLaunch.2threads" * doctest::timeout(300)) {
  module_with_dependent_async_launch(2);
}

TEST_CASE("Module.DependentAsyncLaunch.3threads" * doctest::timeout(300)) {
  module_with_dependent_async_launch(3);
}

TEST_CASE("Module.DependentAsyncLaunch.4threads" * doctest::timeout(300)) {
  module_with_dependent_async_launch(4);
}

TEST_CASE("Module.DependentAsyncLaunch.5threads" * doctest::timeout(300)) {
  module_with_dependent_async_launch(5);
}

TEST_CASE("Module.DependentAsyncLaunch.6threads" * doctest::timeout(300)) {
  module_with_dependent_async_launch(6);
}

TEST_CASE("Module.DependentAsyncLaunch.7threads" * doctest::timeout(300)) {
  module_with_dependent_async_launch(7);
}

TEST_CASE("Module.DependentAsyncLaunch.8threads" * doctest::timeout(300)) {
  module_with_dependent_async_launch(8);
}

// ----------------------------------------------------------------------------
// Module with Silent Dependent Async Launch
// ----------------------------------------------------------------------------

void module_with_silent_dependent_async_launch(unsigned W) {

  tf::Executor executor(W);

  tf::Taskflow f0;

  int cnt {0};

  auto A = f0.emplace([&cnt](){ ++cnt; });
  auto B = f0.emplace([&cnt](){ ++cnt; });
  auto C = f0.emplace([&cnt](){ ++cnt; });
  auto D = f0.emplace([&cnt](){ ++cnt; });
  auto E = f0.emplace([&cnt](){ ++cnt; });

  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);

  tf::Taskflow f1;

  // module 1
  std::tie(A, B, C, D, E) = f1.emplace(
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; },
    [&cnt] () { ++cnt; }
  );
  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);
  auto m1_1 = f1.composed_of(f0);
  E.precede(m1_1);

  auto task = executor.silent_dependent_async(tf::make_module_task(f1)); 

  executor.wait_for_all();

  REQUIRE(task.is_done() == true);
  REQUIRE(cnt == 10);
}

TEST_CASE("Module.DependentAsyncLaunch.1thread" * doctest::timeout(300)) {
  module_with_silent_dependent_async_launch(1);
}

TEST_CASE("Module.DependentAsyncLaunch.2threads" * doctest::timeout(300)) {
  module_with_silent_dependent_async_launch(2);
}

TEST_CASE("Module.DependentAsyncLaunch.3threads" * doctest::timeout(300)) {
  module_with_silent_dependent_async_launch(3);
}

TEST_CASE("Module.DependentAsyncLaunch.4threads" * doctest::timeout(300)) {
  module_with_silent_dependent_async_launch(4);
}

TEST_CASE("Module.DependentAsyncLaunch.5threads" * doctest::timeout(300)) {
  module_with_silent_dependent_async_launch(5);
}

TEST_CASE("Module.DependentAsyncLaunch.6threads" * doctest::timeout(300)) {
  module_with_silent_dependent_async_launch(6);
}

TEST_CASE("Module.DependentAsyncLaunch.7threads" * doctest::timeout(300)) {
  module_with_silent_dependent_async_launch(7);
}

TEST_CASE("Module.DependentAsyncLaunch.8threads" * doctest::timeout(300)) {
  module_with_silent_dependent_async_launch(8);
}


