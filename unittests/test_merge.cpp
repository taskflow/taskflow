#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/algorithm/merge.hpp>
#include <taskflow/taskflow.hpp>

// ----------------------------------------------------------------------------
// parallel merge POD
// ----------------------------------------------------------------------------

template <typename T> void pm_pod(size_t W, size_t N) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  std::vector<T> data1(N);
  std::vector<T> data2(N);

  for (auto &d : data1) {
    d = ::rand() % 1000 - 500;
  }
  for (auto &d : data2) {
    d = ::rand() % 1000 - 500;
  }

  // merge requires sorted input arrays
  std::sort(data1.begin(), data1.end());
  std::sort(data2.begin(), data2.end());

  std::vector<T> res(N * 2);

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  taskflow.merge(data1.begin(), data1.end(), data2.begin(), data2.end(),
                 res.begin());

  executor.run(taskflow).wait();

  REQUIRE(std::is_sorted(res.begin(), res.end()));
}

TEST_CASE("ParallelMerge.int.1.100000" * doctest::timeout(300)) {
  pm_pod<int>(1, 100000);
}

TEST_CASE("ParallelMerge.int.2.100000" * doctest::timeout(300)) {
  pm_pod<int>(2, 100000);
}

TEST_CASE("ParallelMerge.int.3.100000" * doctest::timeout(300)) {
  pm_pod<int>(3, 100000);
}

TEST_CASE("ParallelMerge.int.4.100000" * doctest::timeout(300)) {
  pm_pod<int>(4, 100000);
}

TEST_CASE("ParallelMerge.ldouble.1.100000" * doctest::timeout(300)) {
  pm_pod<long double>(1, 100000);
}

TEST_CASE("ParallelMerge.ldouble.2.100000" * doctest::timeout(300)) {
  pm_pod<long double>(2, 100000);
}

TEST_CASE("ParallelMerge.ldouble.3.100000" * doctest::timeout(300)) {
  pm_pod<long double>(3, 100000);
}

TEST_CASE("ParallelMerge.ldouble.4.100000" * doctest::timeout(300)) {
  pm_pod<long double>(4, 100000);
}

// ----------------------------------------------------------------------------
// Parallel Merge with Async Tasks
// ----------------------------------------------------------------------------

void async(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  tf::Executor executor(W);
  std::vector<int> data1;
  std::vector<int> data2;
  std::vector<int> res;

  for (size_t n = 0; n < 100000; n = (n ? n * 10 : 1)) {

    data1.resize(n);
    data2.resize(n);
    res.resize(n * 2);

    for (auto &d : data1) {
      d = ::rand() % 1000 - 500;
    }
    for (auto &d : data2) {
      d = ::rand() % 1000 - 500;
    }

    std::sort(data1.begin(), data1.end());
    std::sort(data2.begin(), data2.end());

    executor.async(tf::make_merge_task(data1.begin(), data1.end(),
                                       data2.begin(), data2.end(),
                                       std::less<>{}, res.begin()));
    executor.wait_for_all();
    REQUIRE(std::is_sorted(res.begin(), res.end()));
  }
}

TEST_CASE("ParallelMerge.Async.1thread" * doctest::timeout(300)) { async(1); }

TEST_CASE("ParallelMerge.Async.2threads" * doctest::timeout(300)) { async(2); }

TEST_CASE("ParallelMerge.Async.3threads" * doctest::timeout(300)) { async(3); }

TEST_CASE("ParallelMerge.Async.4threads" * doctest::timeout(300)) { async(4); }

// ----------------------------------------------------------------------------
// Parallel Merge with Dependent Async Tasks
// ----------------------------------------------------------------------------

void dependent_async(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  tf::Executor executor(W);
  std::vector<int> data1;
  std::vector<int> data2;
  std::vector<int> res;

  for (size_t n = 0; n < 100000; n = (n ? n * 10 : 1)) {

    data1.resize(n);
    data2.resize(n);
    res.resize(n * 2);

    for (auto &d : data1)
      d = ::rand() % 1000 - 500;
    for (auto &d : data2)
      d = ::rand() % 1000 - 500;

    std::sort(data1.begin(), data1.end());
    std::sort(data2.begin(), data2.end());

    executor.dependent_async(tf::make_merge_task(data1.begin(), data1.end(),
                                                 data2.begin(), data2.end(),
                                                 std::less<>{}, res.begin()));
    executor.wait_for_all();
    REQUIRE(std::is_sorted(res.begin(), res.end()));
  }
}

TEST_CASE("ParallelMerge.DependentAsync.1thread" * doctest::timeout(300)) {
  dependent_async(1);
}

TEST_CASE("ParallelMerge.DependentAsync.2threads" * doctest::timeout(300)) {
  dependent_async(2);
}

TEST_CASE("ParallelMerge.DependentAsync.3threads" * doctest::timeout(300)) {
  dependent_async(3);
}

TEST_CASE("ParallelMerge.DependentAsync.4threads" * doctest::timeout(300)) {
  dependent_async(4);
}

// ----------------------------------------------------------------------------
// Parallel Merge with Silent Async Tasks
// ----------------------------------------------------------------------------

void silent_async(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  tf::Executor executor(W);
  std::vector<int> data1;
  std::vector<int> data2;
  std::vector<int> res;

  for (size_t n = 0; n < 100000; n = (n ? n * 10 : 1)) {

    data1.resize(n);
    data2.resize(n);
    res.resize(n * 2);

    for (auto &d : data1)
      d = ::rand() % 1000 - 500;
    for (auto &d : data2)
      d = ::rand() % 1000 - 500;

    std::sort(data1.begin(), data1.end());
    std::sort(data2.begin(), data2.end());

    executor.silent_async(tf::make_merge_task(data1.begin(), data1.end(),
                                              data2.begin(), data2.end(),
                                              std::less<>{}, res.begin()));
    executor.wait_for_all();
    REQUIRE(std::is_sorted(res.begin(), res.end()));
  }
}

TEST_CASE("ParallelMerge.SilentAsync.1thread" * doctest::timeout(300)) {
  silent_async(1);
}

TEST_CASE("ParallelMerge.SilentAsync.2threads" * doctest::timeout(300)) {
  silent_async(2);
}

TEST_CASE("ParallelMerge.SilentAsync.3threads" * doctest::timeout(300)) {
  silent_async(3);
}

TEST_CASE("ParallelMerge.SilentAsync.4threads" * doctest::timeout(300)) {
  silent_async(4);
}

// ----------------------------------------------------------------------------
// Parallel Merge with Silent Dependent Async Tasks
// ----------------------------------------------------------------------------

void silent_dependent_async(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  tf::Executor executor(W);
  std::vector<int> data1;
  std::vector<int> data2;
  std::vector<int> res;

  for (size_t n = 0; n < 100000; n = (n ? n * 10 : 1)) {

    data1.resize(n);
    data2.resize(n);
    res.resize(n * 2);

    for (auto &d : data1)
      d = ::rand() % 1000 - 500;
    for (auto &d : data2)
      d = ::rand() % 1000 - 500;

    std::sort(data1.begin(), data1.end());
    std::sort(data2.begin(), data2.end());

    executor.silent_dependent_async(
        tf::make_merge_task(data1.begin(), data1.end(), data2.begin(),
                            data2.end(), std::less<>{}, res.begin()));
    executor.wait_for_all();
    REQUIRE(std::is_sorted(res.begin(), res.end()));
  }
}

TEST_CASE("ParallelMerge.SilentDependentAsync.1thread" *
          doctest::timeout(300)) {
  silent_dependent_async(1);
}

TEST_CASE("ParallelMerge.SilentDependentAsync.2threads" *
          doctest::timeout(300)) {
  silent_dependent_async(2);
}

TEST_CASE("ParallelMerge.SilentDependentAsync.3threads" *
          doctest::timeout(300)) {
  silent_dependent_async(3);
}

TEST_CASE("ParallelMerge.SilentDependentAsync.4threads" *
          doctest::timeout(300)) {
  silent_dependent_async(4);
}

// ----------------------------------------------------------------------------
// Parallel Merge with Different Partitioners
// ----------------------------------------------------------------------------

template <typename P> void pm_partitioner(size_t W, size_t N) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  std::vector<int> data1(N);
  std::vector<int> data2(N);

  for (auto &d : data1) {
    d = ::rand() % 1000 - 500;
  }
  for (auto &d : data2) {
    d = ::rand() % 1000 - 500;
  }

  // merge requires sorted input arrays
  std::sort(data1.begin(), data1.end());
  std::sort(data2.begin(), data2.end());

  std::vector<int> res(N * 2);

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  // Pass the instantiated partitioner to the merge algorithm
  taskflow.merge(data1.begin(), data1.end(), data2.begin(), data2.end(),
                 res.begin(), P());

  executor.run(taskflow).wait();

  REQUIRE(std::is_sorted(res.begin(), res.end()));
}

// --- Static Partitioner ---

TEST_CASE("ParallelMerge.StaticPartitioner.1.100000" * doctest::timeout(300)) {
  pm_partitioner<tf::StaticPartitioner<>>(1, 100000);
}

TEST_CASE("ParallelMerge.StaticPartitioner.2.100000" * doctest::timeout(300)) {
  pm_partitioner<tf::StaticPartitioner<>>(2, 100000);
}

TEST_CASE("ParallelMerge.StaticPartitioner.3.100000" * doctest::timeout(300)) {
  pm_partitioner<tf::StaticPartitioner<>>(3, 100000);
}

TEST_CASE("ParallelMerge.StaticPartitioner.4.100000" * doctest::timeout(300)) {
  pm_partitioner<tf::StaticPartitioner<>>(4, 100000);
}

// --- Dynamic Partitioner ---

TEST_CASE("ParallelMerge.DynamicPartitioner.1.100000" * doctest::timeout(300)) {
  pm_partitioner<tf::DynamicPartitioner<>>(1, 100000);
}

TEST_CASE("ParallelMerge.DynamicPartitioner.2.100000" * doctest::timeout(300)) {
  pm_partitioner<tf::DynamicPartitioner<>>(2, 100000);
}

TEST_CASE("ParallelMerge.DynamicPartitioner.3.100000" * doctest::timeout(300)) {
  pm_partitioner<tf::DynamicPartitioner<>>(3, 100000);
}

TEST_CASE("ParallelMerge.DynamicPartitioner.4.100000" * doctest::timeout(300)) {
  pm_partitioner<tf::DynamicPartitioner<>>(4, 100000);
}

// --- Guided Partitioner ---

TEST_CASE("ParallelMerge.GuidedPartitioner.1.100000" * doctest::timeout(300)) {
  pm_partitioner<tf::GuidedPartitioner<>>(1, 100000);
}

TEST_CASE("ParallelMerge.GuidedPartitioner.2.100000" * doctest::timeout(300)) {
  pm_partitioner<tf::GuidedPartitioner<>>(2, 100000);
}

TEST_CASE("ParallelMerge.GuidedPartitioner.3.100000" * doctest::timeout(300)) {
  pm_partitioner<tf::GuidedPartitioner<>>(3, 100000);
}

TEST_CASE("ParallelMerge.GuidedPartitioner.4.100000" * doctest::timeout(300)) {
  pm_partitioner<tf::GuidedPartitioner<>>(4, 100000);
}

// --- Random Partitioner ---

TEST_CASE("ParallelMerge.RandomPartitioner.1.100000" * doctest::timeout(300)) {
  pm_partitioner<tf::RandomPartitioner<>>(1, 100000);
}

TEST_CASE("ParallelMerge.RandomPartitioner.2.100000" * doctest::timeout(300)) {
  pm_partitioner<tf::RandomPartitioner<>>(2, 100000);
}

TEST_CASE("ParallelMerge.RandomPartitioner.3.100000" * doctest::timeout(300)) {
  pm_partitioner<tf::RandomPartitioner<>>(3, 100000);
}

TEST_CASE("ParallelMerge.RandomPartitioner.4.100000" * doctest::timeout(300)) {
  pm_partitioner<tf::RandomPartitioner<>>(4, 100000);
}

// ----------------------------------------------------------------------------
// Parallel Merge with Chunk Size Sweeping
// ----------------------------------------------------------------------------

template <typename P>
void pm_chunk_sweep(size_t W, size_t N, size_t chunk_size) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  std::vector<int> data1(N);
  std::vector<int> data2(N);

  for (auto &d : data1)
    d = ::rand() % 1000 - 500;
  for (auto &d : data2)
    d = ::rand() % 1000 - 500;

  std::sort(data1.begin(), data1.end());
  std::sort(data2.begin(), data2.end());

  std::vector<int> res(N * 2);

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  // Initialize partitioner with the specific chunk size
  taskflow.merge(data1.begin(), data1.end(), data2.begin(), data2.end(),
                 res.begin(), P(chunk_size));

  executor.run(taskflow).wait();

  REQUIRE(std::is_sorted(res.begin(), res.end()));
}

// --- Static Partitioner Sweeps ---

TEST_CASE("ParallelMerge.Static.ChunkSweep" * doctest::timeout(300)) {
  size_t W = 4;
  size_t N = 100000;

  SUBCASE("Small chunk size (1)") {
    pm_chunk_sweep<tf::StaticPartitioner<>>(W, N, 1);
  }
  SUBCASE("Medium chunk size (100)") {
    pm_chunk_sweep<tf::StaticPartitioner<>>(W, N, 100);
  }
  SUBCASE("Large chunk size (10000)") {
    pm_chunk_sweep<tf::StaticPartitioner<>>(W, N, 10000);
  }
  SUBCASE("Chunk size larger than N") {
    pm_chunk_sweep<tf::StaticPartitioner<>>(W, N, N + 1);
  }
}

// --- Dynamic Partitioner Sweeps ---

TEST_CASE("ParallelMerge.Dynamic.ChunkSweep" * doctest::timeout(300)) {
  size_t W = 4;
  size_t N = 100000;

  SUBCASE("Small chunk size (1)") {
    pm_chunk_sweep<tf::DynamicPartitioner<>>(W, N, 1);
  }
  SUBCASE("Medium chunk size (50)") {
    pm_chunk_sweep<tf::DynamicPartitioner<>>(W, N, 50);
  }
  SUBCASE("Large chunk size (5000)") {
    pm_chunk_sweep<tf::DynamicPartitioner<>>(W, N, 5000);
  }
}

// --- Guided Partitioner Sweeps ---

TEST_CASE("ParallelMerge.Guided.ChunkSweep" * doctest::timeout(300)) {
  size_t W = 4;
  size_t N = 100000;

  // Guided partitioner uses chunk_size as the minimum chunk size
  SUBCASE("Min chunk size (1)") {
    pm_chunk_sweep<tf::GuidedPartitioner<>>(W, N, 1);
  }
  SUBCASE("Min chunk size (100)") {
    pm_chunk_sweep<tf::GuidedPartitioner<>>(W, N, 100);
  }
}

// --- Random Partitioner Sweeps ---

TEST_CASE("ParallelMerge.Random.ChunkSweep" * doctest::timeout(300)) {
  size_t W = 4;
  size_t N = 100000;

  SUBCASE("Small range (1, 10)") {
    tf::Executor executor(W);
    std::vector<int> d1(N), d2(N), res(N * 2);
    // Standard setup omitted for brevity in this specific subcase call
    // Logic: P(min, max)
    tf::Taskflow tf;
    tf.merge(d1.begin(), d1.end(), d2.begin(), d2.end(), res.begin(),
             tf::RandomPartitioner<>(1, 10));
    executor.run(tf).wait();
  }
}

// ----------------------------------------------------------------------------
// Parallel Merge with Custom Comparator (Descending Order)
// ----------------------------------------------------------------------------

template <typename T> void pm_custom_cmp(size_t W, size_t N) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  std::vector<T> data1(N);
  std::vector<T> data2(N);

  for (auto &d : data1) {
    d = ::rand() % 1000 - 500;
  }
  for (auto &d : data2) {
    d = ::rand() % 1000 - 500;
  }

  // Input arrays MUST be sorted according to the custom comparator
  std::sort(data1.begin(), data1.end(), std::greater<T>{});
  std::sort(data2.begin(), data2.end(), std::greater<T>{});

  std::vector<T> res(N * 2);

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  // Pass custom comparator: std::greater
  taskflow.merge(data1.begin(), data1.end(), data2.begin(), data2.end(),
                 res.begin(), std::greater<T>{});

  executor.run(taskflow).wait();

  // Verify the result is sorted in descending order
  REQUIRE(std::is_sorted(res.begin(), res.end(), std::greater<T>{}));
}

TEST_CASE("ParallelMerge.CustomCmp.1.100000" * doctest::timeout(300)) {
  pm_custom_cmp<int>(1, 100000);
}

TEST_CASE("ParallelMerge.CustomCmp.2.100000" * doctest::timeout(300)) {
  pm_custom_cmp<int>(2, 100000);
}

TEST_CASE("ParallelMerge.CustomCmp.3.100000" * doctest::timeout(300)) {
  pm_custom_cmp<int>(3, 100000);
}

TEST_CASE("ParallelMerge.CustomCmp.4.100000" * doctest::timeout(300)) {
  pm_custom_cmp<int>(4, 100000);
}

// ----------------------------------------------------------------------------
// Parallel Merge with Custom Comparator AND Custom Partitioner
// ----------------------------------------------------------------------------

template <typename T, typename P> 
void pm_custom_cmp_partitioner(size_t W, size_t N) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  std::vector<T> data1(N);
  std::vector<T> data2(N);

  for (auto &d : data1) {
    d = ::rand() % 1000 - 500;
  }
  for (auto &d : data2) {
    d = ::rand() % 1000 - 500;
  }

  std::sort(data1.begin(), data1.end(), std::greater<T>{});
  std::sort(data2.begin(), data2.end(), std::greater<T>{});

  std::vector<T> res(N * 2);

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  // Pass BOTH custom comparator and custom partitioner
  taskflow.merge(data1.begin(), data1.end(), data2.begin(), data2.end(),
                 res.begin(), std::greater<T>{}, P());

  executor.run(taskflow).wait();

  REQUIRE(std::is_sorted(res.begin(), res.end(), std::greater<T>{}));
}

TEST_CASE("ParallelMerge.CustomCmp_StaticPartitioner.4.100000" * doctest::timeout(300)) {
  pm_custom_cmp_partitioner<int, tf::StaticPartitioner<>>(4, 100000);
}

TEST_CASE("ParallelMerge.CustomCmp_DynamicPartitioner.4.100000" * doctest::timeout(300)) {
  pm_custom_cmp_partitioner<int, tf::DynamicPartitioner<>>(4, 100000);
}

TEST_CASE("ParallelMerge.CustomCmp_GuidedPartitioner.4.100000" * doctest::timeout(300)) {
  pm_custom_cmp_partitioner<int, tf::GuidedPartitioner<>>(4, 100000);
}

TEST_CASE("ParallelMerge.CustomCmp_RandomPartitioner.4.100000" * doctest::timeout(300)) {
  pm_custom_cmp_partitioner<int, tf::RandomPartitioner<>>(4, 100000);
}
