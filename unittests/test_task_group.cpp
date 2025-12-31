#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

TEST_CASE("TaskGroup.Basics") {

  tf::Executor executor;
  
  // create a task group from a taskflow task
  tf::Taskflow taskflow;
  taskflow.emplace([&](){
    tf::TaskGroup tg = executor.task_group();
    REQUIRE(tg.size() == 0);
  });
  executor.run(taskflow);

  // create a task group from an async task
  executor.async([&](){
    tf::TaskGroup tg = executor.task_group();
    REQUIRE(tg.size() == 0);
  });
  
  executor.silent_async([&](){
    tf::TaskGroup tg = executor.task_group();
    REQUIRE(tg.size() == 0);
  });

  // create a task group from a dependent-async task
  executor.dependent_async([&](){
    tf::TaskGroup tg = executor.task_group();
    REQUIRE(tg.size() == 0);
  });
  
  executor.silent_dependent_async([&](){
    tf::TaskGroup tg = executor.task_group();
    REQUIRE(tg.size() == 0);
  });
  
  executor.wait_for_all();
}

// ----------------------------------------------------------------------------
// Async
// ----------------------------------------------------------------------------

void async(unsigned W) {
  
  tf::Executor executor(W);

  executor.async([&](){

    std::atomic<size_t> counter(0);

    tf::TaskGroup tg = executor.task_group();
    
    for(size_t i=0; i<1000; ++i) {
      tg.async([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
      tg.silent_async([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
    }
    
    tg.corun();

    REQUIRE(counter.load(std::memory_order_relaxed) == 2000);

  }).wait();

}

TEST_CASE("TaskGroup.Async.1thread") {
  async(1);
}

TEST_CASE("TaskGroup.Async.2threads") {
  async(2);
}

TEST_CASE("TaskGroup.Async.3threads") {
  async(3);
}

TEST_CASE("TaskGroup.Async.4threads") {
  async(4);
}

TEST_CASE("TaskGroup.Async.5threads") {
  async(5);
}

TEST_CASE("TaskGroup.Async.6threads") {
  async(6);
}

TEST_CASE("TaskGroup.Async.7threads") {
  async(7);
}

TEST_CASE("TaskGroup.Async.8threads") {
  async(8);
}

// ----------------------------------------------------------------------------
// Dependent Async
// ----------------------------------------------------------------------------

void dependent_async(unsigned W) {
  
  tf::Executor executor(W);

  executor.async([&](){
    
    size_t counter = 0;

    tf::TaskGroup tg = executor.task_group();
      
    {
      auto prev = tg.silent_dependent_async([&](){ counter++; });
      for(size_t i=1; i<1000; ++i) {
        prev = tg.silent_dependent_async([&](){ counter++; }, prev);
      }
      tg.corun();
      REQUIRE(counter == 1000);
    }
    
    {
      auto [prev, fu] = tg.dependent_async([&](){ counter = 0; });
      for(size_t i=0; i<1000; ++i) {
        std::tie(prev, fu) = tg.dependent_async([&](){ counter++; }, prev);
      }
      tg.corun();
      REQUIRE(counter == 1000);
    }

  }).wait();

}

TEST_CASE("TaskGroup.DependentAsync.1thread") {
  dependent_async(1);
}

TEST_CASE("TaskGroup.DependentAsync.2threads") {
  dependent_async(2);
}

TEST_CASE("TaskGroup.DependentAsync.3threads") {
  dependent_async(3);
}

TEST_CASE("TaskGroup.DependentAsync.4threads") {
  dependent_async(4);
}

TEST_CASE("TaskGroup.DependentAsync.5threads") {
  dependent_async(5);
}

TEST_CASE("TaskGroup.DependentAsync.6threads") {
  dependent_async(6);
}

TEST_CASE("TaskGroup.DependentAsync.7threads") {
  dependent_async(7);
}

TEST_CASE("TaskGroup.DependentAsync.8threads") {
  dependent_async(8);
}

// ----------------------------------------------------------------------------
// Cancellation
// ----------------------------------------------------------------------------

void cancellation(unsigned W) {

  REQUIRE(W>1);

  tf::Executor executor(W);

  executor.async([&executor, W](){

    auto tg = executor.task_group();

    std::atomic<bool> should_never_run = true;
    std::atomic<size_t> arrivals(0);
    
    // block the other W-1 workers block
    for(size_t i=0; i<W-1; ++i) {
      tg.async([&](){
        ++arrivals;
        while(arrivals != 0);
      });
    }
    
    // wait until the other W-1 workers block
    while(arrivals != W-1);

    // now spawn other tasks which should never run after cancellation
    for(size_t i=0; i<100; ++i) {
      tg.async([&](){ should_never_run = false; });
    }
    
    // now cancel the task group and unblock the other workers
    REQUIRE(tg.is_cancelled() == false);
    tg.cancel();
    REQUIRE(tg.is_cancelled() == true);
    arrivals = 0;

    tg.corun();

    REQUIRE(should_never_run == true);
  }).wait();
}

TEST_CASE("TaskGroup.Cancellation.2threads") {
  cancellation(2);
}

TEST_CASE("TaskGroup.Cancellation.3threads") {
  cancellation(3);
}

TEST_CASE("TaskGroup.Cancellation.4threads") {
  cancellation(4);
}

TEST_CASE("TaskGroup.Cancellation.5threads") {
  cancellation(5);
}

TEST_CASE("TaskGroup.Cancellation.6threads") {
  cancellation(6);
}

TEST_CASE("TaskGroup.Cancellation.7threads") {
  cancellation(7);
}

TEST_CASE("TaskGroup.Cancellation.8threads") {
  cancellation(8);
}

// ----------------------------------------------------------------------------
// Fibonacci
// ----------------------------------------------------------------------------

size_t fibonacci(size_t N, tf::Executor& executor) {

  if (N < 2) {
    return N; 
  }
  
  size_t res1, res2;

  auto tg = executor.task_group();

  tg.silent_async([N, &res1, &executor](){ res1 = fibonacci(N-1, executor); });
  
  // tail optimization
  res2 = fibonacci(N-2, executor);

  // use corun to avoid blocking the worker from waiting the two children tasks to finish
  tg.corun();

  return res1 + res2;
}

size_t fibonacci(size_t T, size_t N) {
  tf::Executor executor(T);
  return executor.async([N, &executor](){ return fibonacci(N, executor); }).get();
}

TEST_CASE("Runtime.Fibonacci.1thread" * doctest::timeout(250)) {
  REQUIRE(fibonacci(1, 25) == 75025);
}

TEST_CASE("Runtime.Fibonacci.2threads" * doctest::timeout(250)) {
  REQUIRE(fibonacci(2, 25) == 75025);
}

TEST_CASE("Runtime.Fibonacci.3threads" * doctest::timeout(250)) {
  REQUIRE(fibonacci(3, 25) == 75025);
}

TEST_CASE("Runtime.Fibonacci.4threads" * doctest::timeout(250)) {
  REQUIRE(fibonacci(4, 25) == 75025);
}

size_t fibonacci_swapped(size_t N, tf::Executor& executor) {

  if (N < 2) {
    return N; 
  }
  
  size_t res1, res2;

  auto tg = executor.task_group();

  res2 = fibonacci_swapped(N-2, executor);

  tg.silent_async([N, &res1, &executor](){ res1 = fibonacci_swapped(N-1, executor); });

  // use corun to avoid blocking the worker from waiting the two children tasks to finish
  tg.corun();

  return res1 + res2;
}

size_t fibonacci_swapped(size_t T, size_t N) {
  tf::Executor executor(T);
  return executor.async([N, &executor](){ return fibonacci_swapped(N, executor); }).get();
}

TEST_CASE("Runtime.FibonacciSwapped.1thread" * doctest::timeout(250)) {
  REQUIRE(fibonacci_swapped(1, 25) == 75025);
}

TEST_CASE("Runtime.FibonacciSwapped.2threads" * doctest::timeout(250)) {
  REQUIRE(fibonacci_swapped(2, 25) == 75025);
}

TEST_CASE("Runtime.FibonacciSwapped.3threads" * doctest::timeout(250)) {
  REQUIRE(fibonacci_swapped(3, 25) == 75025);
}

TEST_CASE("Runtime.FibonacciSwapped.4threads" * doctest::timeout(250)) {
  REQUIRE(fibonacci_swapped(4, 25) == 75025);
}

// --------------------------------------------------------
// Testcase: MergeSort
// --------------------------------------------------------

void merge_sort_spawn(tf::Executor& executor, std::vector<int>& data, int beg, int end) {

  if(!(beg < end) || end - beg == 1) {
    return;
  }

  if(end - beg <= 5) {
    std::sort(data.begin() + beg, data.begin() + end);
    return;
  }

  int m = (beg + end + 1) / 2;

  auto tg = executor.task_group();

  tg.silent_async([&data, beg, m, &executor] () {
    merge_sort_spawn(executor, data, beg, m);
  });

  merge_sort_spawn(executor, data, m, end);

  tg.corun();

  std::vector<int> tmpl, tmpr;
  for(int i=beg; i<m; ++i) tmpl.push_back(data[i]);
  for(int i=m; i<end; ++i) tmpr.push_back(data[i]);

  // merge to data
  size_t i=0, j=0, k=beg;
  while(i<tmpl.size() && j<tmpr.size()) {
    data[k++] = (tmpl[i] < tmpr[j] ? tmpl[i++] : tmpr[j++]);
  }

  // remaining SL
  for(; i<tmpl.size(); ++i) data[k++] = tmpl[i];

  // remaining SR
  for(; j<tmpr.size(); ++j) data[k++] = tmpr[j];
  
}

void merge_sort(unsigned W) {

  tf::Executor executor(W);
  std::vector<int> data, gold;

  for(int end=10; end <= 10000; end *= 10) {

    data.resize(end);
    gold.resize(end);

    for(size_t k=0; k<data.size(); ++k) {
      data[k] = ::rand() % 100;
      gold[k] = data[k];
    }
    
    executor.async(
      [&data, end, &executor](){ merge_sort_spawn(executor, data, 0, end); }
    ).wait();

    std::sort(gold.begin(), gold.end());

    REQUIRE(gold == data);
  }
}

TEST_CASE("TaskGroup.MergeSort.1thread" * doctest::timeout(300)) {
  merge_sort(1);
}

TEST_CASE("TaskGroup.MergeSort.2threads" * doctest::timeout(300)) {
  merge_sort(2);
}

TEST_CASE("TaskGroup.MergeSort.3threads" * doctest::timeout(300)) {
  merge_sort(3);
}

TEST_CASE("TaskGroup.MergeSort.4threads" * doctest::timeout(300)) {
  merge_sort(4);
}

TEST_CASE("TaskGroup.MergeSort.5threads" * doctest::timeout(300)) {
  merge_sort(5);
}

TEST_CASE("TaskGroup.MergeSort.6threads" * doctest::timeout(300)) {
  merge_sort(6);
}

TEST_CASE("TaskGroup.MergeSort.7threads" * doctest::timeout(300)) {
  merge_sort(7);
}

TEST_CASE("TaskGroup.MergeSort.8threads" * doctest::timeout(300)) {
  merge_sort(8);
}

// --------------------------------------------------------
// Testcase: QuickSort
// --------------------------------------------------------

void quick_sort_spawn(
  tf::Executor& executor, 
  std::vector<int>& data, 
  std::vector<int>::iterator beg,
  std::vector<int>::iterator end
) {

  if(!(beg < end) || std::distance(beg, end) == 1) {
    return;
  }

  if(std::distance(beg, end) <= 5) {
    std::sort(beg, end);
    return;
  }

  auto pvt = beg + std::distance(beg, end) / 2;

  std::iter_swap(pvt, end-1);

  pvt = std::partition(beg, end-1, [end] (int item) {
    return item < *(end - 1);
  });

  std::iter_swap(pvt, end-1);

  auto tg = executor.task_group();

  tg.silent_async([=, &data, &executor] () {
    quick_sort_spawn(executor, data, beg, pvt);
  });

  quick_sort_spawn(executor, data, pvt+1, end);

  tg.corun();
}

void quick_sort(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;
  std::vector<int> data, gold;

  taskflow.emplace([&data, &executor](){
    quick_sort_spawn(executor, data, data.begin(), data.end());
  });

  for(size_t end=1; end <= 10000; end *= 10) {

    data.resize(end);
    gold.resize(end);

    for(size_t k=0; k<data.size(); ++k) {
      data[k] = ::rand()%100;
      gold[k] = data[k];
    }

    executor.run(taskflow).wait();

    std::sort(gold.begin(), gold.end());

    REQUIRE(gold == data);
  }
  
}

TEST_CASE("TaskGroup.QuickSort.1thread" * doctest::timeout(300)) {
  quick_sort(1);
}

TEST_CASE("TaskGroup.QuickSort.2threads" * doctest::timeout(300)) {
  quick_sort(2);
}

TEST_CASE("TaskGroup.QuickSort.3threads" * doctest::timeout(300)) {
  quick_sort(3);
}

TEST_CASE("TaskGroup.QuickSort.4threads" * doctest::timeout(300)) {
  quick_sort(4);
}

TEST_CASE("TaskGroup.QuickSort.5threads" * doctest::timeout(300)) {
  quick_sort(5);
}

TEST_CASE("TaskGroup.QuickSort.6threads" * doctest::timeout(300)) {
  quick_sort(6);
}

TEST_CASE("TaskGroup.QuickSort.7threads" * doctest::timeout(300)) {
  quick_sort(7);
}

TEST_CASE("TaskGroup.QuickSort.8threads" * doctest::timeout(300)) {
  quick_sort(8);
}







