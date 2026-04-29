#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/merge.hpp>

// ============================================================================
// Helper: core correctness check
//
// Generates two random sorted sequences of sizes n1 and n2, runs the
// parallel merge, and compares against std::merge ground truth.
// Works for any element type T and comparator C.
// ============================================================================

template <typename T, typename C = std::less<T>>
void check_merge(unsigned W, size_t n1, size_t n2, C cmp = C{}) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::vector<T> data1(n1), data2(n2), res(n1 + n2), gt(n1 + n2);

  for(auto& d : data1) { d = static_cast<T>(::rand() % 2000 - 1000); }
  for(auto& d : data2) { d = static_cast<T>(::rand() % 2000 - 1000); }
  std::sort(data1.begin(), data1.end(), cmp);
  std::sort(data2.begin(), data2.end(), cmp);
  std::merge(data1.begin(), data1.end(), data2.begin(), data2.end(), gt.begin(), cmp);

  taskflow.merge(data1.begin(), data1.end(),
                 data2.begin(), data2.end(),
                 res.begin(), cmp);
  executor.run(taskflow).wait();

  REQUIRE(res == gt);
}

// ============================================================================
// Basic correctness: edge cases + size sweep
// ============================================================================

void pm_basic(unsigned W) {

  std::srand(static_cast<unsigned>(time(nullptr)));

  // empty inputs
  check_merge<int>(W, 0, 0);
  check_merge<int>(W, 0, 100);
  check_merge<int>(W, 100, 0);
  check_merge<int>(W, 1, 1);

  // equal-size sweep
  for(size_t n : {1, 2, 3, 7, 15, 99, 1000, 10000}) {
    check_merge<int>(W, n, n);
  }

  // unequal sizes — exercises co_rank boundary conditions
  check_merge<int>(W, 1,     1000);
  check_merge<int>(W, 1000,  1);
  check_merge<int>(W, 3,     10000);
  check_merge<int>(W, 10000, 3);

  // other types
  check_merge<long double>(W, 1000, 1000);
  check_merge<float>(W, 999, 1001);
}

// ============================================================================
// co_rank boundary: j==0 UB regression
//
// When one input is empty or much smaller than the other, co_rank can reach
// j==0, which in the original code caused seq2[-1] dereference (UB).
// With -fsanitize=undefined these cases would crash; even without UBSAN the
// output would be wrong for very unequal sizes under multi-threaded execution.
// ============================================================================

void pm_corank_boundary(unsigned W) {

  std::srand(static_cast<unsigned>(time(nullptr)));

  // n2=0: every co_rank call yields j=0 — was UB before fix
  check_merge<int>(W, 1000, 0);
  check_merge<int>(W, 1,    0);
  check_merge<int>(W, 100,  0);

  // n1=0: every co_rank call yields i=0
  check_merge<int>(W, 0, 1000);
  check_merge<int>(W, 0, 1);

  // single element in one sequence — j reaches 0 for most chunks
  check_merge<int>(W, 1000, 1);
  check_merge<int>(W, 9999, 1);
  check_merge<int>(W, 1,    1000);
  check_merge<int>(W, 1,    9999);
}

// ============================================================================
// Custom comparator: descending order
// ============================================================================

void pm_custom_cmp(unsigned W) {

  std::srand(static_cast<unsigned>(time(nullptr)));

  for(size_t n : {0, 1, 99, 1000, 10000}) {
    check_merge<int>(W, n, n, std::greater<int>{});
    check_merge<int>(W, n, n / 2 + 1, std::greater<int>{});
  }
}

// ============================================================================
// Stateful ranges (std::ref): init task sets bounds before merge runs
// ============================================================================

void pm_stateful(unsigned W) {

  std::srand(static_cast<unsigned>(time(nullptr)));

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::vector<int> data1, data2, res;
  std::vector<int>::iterator beg1, end1, beg2, end2, d_beg;

  for(size_t n : {0, 1, 7, 100, 1000}) {
    data1.resize(n);
    data2.resize(n);
    res.resize(n * 2);

    for(auto& d : data1) { d = ::rand() % 2000 - 1000; }
    for(auto& d : data2) { d = ::rand() % 2000 - 1000; }
    std::sort(data1.begin(), data1.end());
    std::sort(data2.begin(), data2.end());

    taskflow.clear();

    auto init = taskflow.emplace([&]() {
      beg1  = data1.begin();
      end1  = data1.end();
      beg2  = data2.begin();
      end2  = data2.end();
      d_beg = res.begin();
    });

    auto merge_task = taskflow.merge(
      std::ref(beg1), std::ref(end1),
      std::ref(beg2), std::ref(end2),
      std::ref(d_beg)
    );

    init.precede(merge_task);
    executor.run(taskflow).wait();

    std::vector<int> gt(n * 2);
    std::merge(data1.begin(), data1.end(), data2.begin(), data2.end(), gt.begin());
    REQUIRE(res == gt);
  }
}

// ============================================================================
// Async task variants
// ============================================================================

void pm_async(unsigned W) {

  std::srand(static_cast<unsigned>(time(nullptr)));

  tf::Executor executor(W);

  for(size_t n : {0, 1, 7, 100, 1000, 10000}) {
    std::vector<int> data1(n), data2(n), res(n * 2), gt(n * 2);

    for(auto& d : data1) { d = ::rand() % 2000 - 1000; }
    for(auto& d : data2) { d = ::rand() % 2000 - 1000; }
    std::sort(data1.begin(), data1.end());
    std::sort(data2.begin(), data2.end());
    std::merge(data1.begin(), data1.end(), data2.begin(), data2.end(), gt.begin());

    executor.async(tf::make_merge_task(
      data1.begin(), data1.end(),
      data2.begin(), data2.end(),
      std::less<>{}, res.begin()
    ));
    executor.wait_for_all();

    REQUIRE(res == gt);
  }
}

void pm_silent_async(unsigned W) {

  std::srand(static_cast<unsigned>(time(nullptr)));

  tf::Executor executor(W);

  for(size_t n : {0, 1, 7, 100, 1000, 10000}) {
    std::vector<int> data1(n), data2(n), res(n * 2), gt(n * 2);

    for(auto& d : data1) { d = ::rand() % 2000 - 1000; }
    for(auto& d : data2) { d = ::rand() % 2000 - 1000; }
    std::sort(data1.begin(), data1.end());
    std::sort(data2.begin(), data2.end());
    std::merge(data1.begin(), data1.end(), data2.begin(), data2.end(), gt.begin());

    executor.silent_async(tf::make_merge_task(
      data1.begin(), data1.end(),
      data2.begin(), data2.end(),
      std::less<>{}, res.begin()
    ));
    executor.wait_for_all();

    REQUIRE(res == gt);
  }
}

void pm_dependent_async(unsigned W) {

  std::srand(static_cast<unsigned>(time(nullptr)));

  tf::Executor executor(W);

  for(size_t n : {0, 1, 7, 100, 1000, 10000}) {
    std::vector<int> data1(n), data2(n), res(n * 2), gt(n * 2);

    for(auto& d : data1) { d = ::rand() % 2000 - 1000; }
    for(auto& d : data2) { d = ::rand() % 2000 - 1000; }
    std::sort(data1.begin(), data1.end());
    std::sort(data2.begin(), data2.end());
    std::merge(data1.begin(), data1.end(), data2.begin(), data2.end(), gt.begin());

    executor.dependent_async(tf::make_merge_task(
      data1.begin(), data1.end(),
      data2.begin(), data2.end(),
      std::less<>{}, res.begin()
    ));
    executor.wait_for_all();

    REQUIRE(res == gt);
  }
}

// ============================================================================
// TEST CASES: basic correctness
// ============================================================================

TEST_CASE("ParallelMerge.basic.1thread"  * doctest::timeout(300)) { pm_basic(1); }
TEST_CASE("ParallelMerge.basic.2threads" * doctest::timeout(300)) { pm_basic(2); }
TEST_CASE("ParallelMerge.basic.4threads" * doctest::timeout(300)) { pm_basic(4); }
TEST_CASE("ParallelMerge.basic.8threads" * doctest::timeout(300)) { pm_basic(8); }

// ============================================================================
// TEST CASES: co_rank j==0 boundary regression
// ============================================================================

TEST_CASE("ParallelMerge.corank_boundary.1thread"  * doctest::timeout(300)) { pm_corank_boundary(1); }
TEST_CASE("ParallelMerge.corank_boundary.2threads" * doctest::timeout(300)) { pm_corank_boundary(2); }
TEST_CASE("ParallelMerge.corank_boundary.4threads" * doctest::timeout(300)) { pm_corank_boundary(4); }
TEST_CASE("ParallelMerge.corank_boundary.8threads" * doctest::timeout(300)) { pm_corank_boundary(8); }

// ============================================================================
// TEST CASES: custom comparator
// ============================================================================

TEST_CASE("ParallelMerge.custom_cmp.1thread"  * doctest::timeout(300)) { pm_custom_cmp(1); }
TEST_CASE("ParallelMerge.custom_cmp.2threads" * doctest::timeout(300)) { pm_custom_cmp(2); }
TEST_CASE("ParallelMerge.custom_cmp.4threads" * doctest::timeout(300)) { pm_custom_cmp(4); }
TEST_CASE("ParallelMerge.custom_cmp.8threads" * doctest::timeout(300)) { pm_custom_cmp(8); }

// ============================================================================
// TEST CASES: stateful ranges
// ============================================================================

TEST_CASE("ParallelMerge.stateful.1thread"  * doctest::timeout(300)) { pm_stateful(1); }
TEST_CASE("ParallelMerge.stateful.2threads" * doctest::timeout(300)) { pm_stateful(2); }
TEST_CASE("ParallelMerge.stateful.4threads" * doctest::timeout(300)) { pm_stateful(4); }
TEST_CASE("ParallelMerge.stateful.8threads" * doctest::timeout(300)) { pm_stateful(8); }

// ============================================================================
// TEST CASES: async variants
// ============================================================================

TEST_CASE("ParallelMerge.Async.1thread"  * doctest::timeout(300)) { pm_async(1); }
TEST_CASE("ParallelMerge.Async.2threads" * doctest::timeout(300)) { pm_async(2); }
TEST_CASE("ParallelMerge.Async.4threads" * doctest::timeout(300)) { pm_async(4); }
TEST_CASE("ParallelMerge.Async.8threads" * doctest::timeout(300)) { pm_async(8); }

TEST_CASE("ParallelMerge.SilentAsync.1thread"  * doctest::timeout(300)) { pm_silent_async(1); }
TEST_CASE("ParallelMerge.SilentAsync.2threads" * doctest::timeout(300)) { pm_silent_async(2); }
TEST_CASE("ParallelMerge.SilentAsync.4threads" * doctest::timeout(300)) { pm_silent_async(4); }
TEST_CASE("ParallelMerge.SilentAsync.8threads" * doctest::timeout(300)) { pm_silent_async(8); }

TEST_CASE("ParallelMerge.DependentAsync.1thread"  * doctest::timeout(300)) { pm_dependent_async(1); }
TEST_CASE("ParallelMerge.DependentAsync.2threads" * doctest::timeout(300)) { pm_dependent_async(2); }
TEST_CASE("ParallelMerge.DependentAsync.4threads" * doctest::timeout(300)) { pm_dependent_async(4); }
TEST_CASE("ParallelMerge.DependentAsync.8threads" * doctest::timeout(300)) { pm_dependent_async(8); }
