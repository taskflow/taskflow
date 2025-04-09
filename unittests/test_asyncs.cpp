#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

// --------------------------------------------------------
// Testcase: Async
// --------------------------------------------------------

void async(unsigned W) {

  tf::Executor executor(W);

  std::vector<std::future<int>> fus;

  std::atomic<int> counter(0);

  int N = 100000;

  for(int i=0; i<N; ++i) {
    if(auto r = i%3; r==0) {
      fus.emplace_back(executor.async(std::to_string(i), [&](){
        counter.fetch_add(1, std::memory_order_relaxed);
        return -2;
      }));
    }else if(r == 1) {
      fus.emplace_back(executor.async(tf::DefaultTaskParams{}, [&](){
        counter.fetch_add(1, std::memory_order_relaxed);
        return -2;
      }));
    }
    else {
      fus.emplace_back(executor.async(tf::TaskParams{std::to_string(i)}, [&](){
        counter.fetch_add(1, std::memory_order_relaxed);
        return -2;
      }));
    }
  }

  executor.wait_for_all();

  REQUIRE(counter == N);

  int c = 0;
  for(auto& fu : fus) {
    c += fu.get();
  }

  REQUIRE(-c == 2*N);
}

TEST_CASE("Async.1thread" * doctest::timeout(300)) {
  async(1);
}

TEST_CASE("Async.2threads" * doctest::timeout(300)) {
  async(2);
}

TEST_CASE("Async.4threads" * doctest::timeout(300)) {
  async(4);
}

TEST_CASE("Async.8threads" * doctest::timeout(300)) {
  async(8);
}

TEST_CASE("Async.16threads" * doctest::timeout(300)) {
  async(16);
}

// --------------------------------------------------------
// Testcase: NestedAsync
// --------------------------------------------------------

void nested_async(unsigned W) {

  tf::Executor executor(W);

  std::vector<std::future<int>> fus;

  std::atomic<int> counter(0);

  int N = 100000;

  for(int i=0; i<N; ++i) {
    fus.emplace_back(executor.async([&](){
      counter.fetch_add(1, std::memory_order_relaxed);
      executor.async([&](){
        counter.fetch_add(1, std::memory_order_relaxed);
        executor.async([&](){
          counter.fetch_add(1, std::memory_order_relaxed);
          executor.async([&](){
            counter.fetch_add(1, std::memory_order_relaxed);
          });
        });
      });
      return -2;
    }));
  }

  executor.wait_for_all();

  REQUIRE(counter == 4*N);

  int c = 0;
  for(auto& fu : fus) {
    c += fu.get();
  }

  REQUIRE(-c == 2*N);
}

TEST_CASE("NestedAsync.1thread" * doctest::timeout(300)) {
  nested_async(1);
}

TEST_CASE("NestedAsync.2threads" * doctest::timeout(300)) {
  nested_async(2);
}

TEST_CASE("NestedAsync.4threads" * doctest::timeout(300)) {
  nested_async(4);
}

TEST_CASE("NestedAsync.8threads" * doctest::timeout(300)) {
  nested_async(8);
}

TEST_CASE("NestedAsync.16threads" * doctest::timeout(300)) {
  nested_async(16);
}

// --------------------------------------------------------
// Testcase MixedExecutorAsync
// --------------------------------------------------------

void mixed_executor_async(size_t N) {

  const size_t T = 1000;

  std::vector<tf::Executor> executors(N);
  
  std::atomic<size_t> counter(0);

  auto check_wid = [&](size_t e){
    for(size_t i=0; i<N; i++) {
      if(i == e) {
        REQUIRE(executors[i].this_worker_id() != -1);
      }
      else {
        REQUIRE(executors[i].this_worker_id() == -1);
      }
    }
  };

  for(size_t j=0; j<T; j++) {
    for(size_t i=0; i<N; i++) {
      executors[i].async([&, i, j](){
        check_wid(i);
        counter.fetch_add(1, std::memory_order_relaxed);
        auto n = j % N;
        executors[n].async([&, n](tf::Runtime&){
          check_wid(n);
          counter.fetch_add(1, std::memory_order_relaxed);
        });
      });
      
      executors[i].silent_async([&, i, j](){
        check_wid(i);
        counter.fetch_add(1, std::memory_order_relaxed);
        auto n = (j + 1) % N;
        executors[n].silent_async([&, n](tf::Runtime){
          check_wid(n);
          counter.fetch_add(1, std::memory_order_relaxed);
        });
      });
    }
  }

  while(counter.load() != 4000*N);

  for(auto& executor : executors) {
    executor.wait_for_all();
  }
}

TEST_CASE("MixedAsync.1Executor" * doctest::timeout(300)) {
  mixed_executor_async(1);
}

TEST_CASE("MixedAsync.2Executors" * doctest::timeout(300)) {
  mixed_executor_async(2);
}

TEST_CASE("MixedAsync.4Executors" * doctest::timeout(300)) {
  mixed_executor_async(4);
}

TEST_CASE("MixedAsync.5Executors" * doctest::timeout(300)) {
  mixed_executor_async(5);
}

TEST_CASE("MixedAsync.6Executors" * doctest::timeout(300)) {
  mixed_executor_async(6);
}

TEST_CASE("MixedAsync.7Executors" * doctest::timeout(300)) {
  mixed_executor_async(7);
}

TEST_CASE("MixedAsync.8Executors" * doctest::timeout(300)) {
  mixed_executor_async(8);
}

// --------------------------------------------------------
// Testcase: MixedAsync
// --------------------------------------------------------

void mixed_async(unsigned W) {

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  std::atomic<int> counter(0);

  int N = 10000;

  for(int i=0; i<N; i=i+1) {
    tf::Task A, B, C, D;
    std::tie(A, B, C, D) = taskflow.emplace(
      [&] () {
        executor.async([&](){
          counter.fetch_add(1, std::memory_order_relaxed);
        });
      },
      [&] () {
        executor.async([&](){
          counter.fetch_add(1, std::memory_order_relaxed);
        });
      },
      [&] () {
        executor.silent_async([&](){
          counter.fetch_add(1, std::memory_order_relaxed);
        });
      },
      [&] () {
        executor.silent_async([&](){
          counter.fetch_add(1, std::memory_order_relaxed);
        });
      }
    );

    A.precede(B, C);
    D.succeed(B, C);
  }

  executor.run(taskflow);
  executor.wait_for_all();

  REQUIRE(counter == 4*N);

}

TEST_CASE("MixedAsync.1thread" * doctest::timeout(300)) {
  mixed_async(1);
}

TEST_CASE("MixedAsync.2threads" * doctest::timeout(300)) {
  mixed_async(2);
}

TEST_CASE("MixedAsync.4threads" * doctest::timeout(300)) {
  mixed_async(4);
}

TEST_CASE("MixedAsync.8threads" * doctest::timeout(300)) {
  mixed_async(8);
}

TEST_CASE("MixedAsync.16threads" * doctest::timeout(300)) {
  mixed_async(16);
}

// --------------------------------------------------------
// Testcase: RuntimeAsync
// --------------------------------------------------------

void runtime_async(size_t W) {

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  std::atomic<int> counter{0};

  auto A = taskflow.emplace(
    [&](){ counter.fetch_add(1, std::memory_order_relaxed); }
  );
  auto B = taskflow.emplace(
    [&](){ counter.fetch_add(1, std::memory_order_relaxed); }
  );

  taskflow.emplace(
    [&](){ counter.fetch_add(1, std::memory_order_relaxed); }
  );

  auto S1 = taskflow.emplace([&] (tf::Runtime& sf){
    for(int i=0; i<1000; i++) {
      sf.silent_async(
        [&](){counter.fetch_add(1, std::memory_order_relaxed);}
      );
    }
    sf.corun();
  });

  auto S2 = taskflow.emplace([&] (tf::Runtime& sf){
    sf.silent_async([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
    for(int i=0; i<1000; i++) {
      sf.silent_async([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
    }
    sf.corun();
  });

  taskflow.emplace([&] (tf::Runtime& sf){
    sf.silent_async([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
    for(int i=0; i<1000; i++) {
      sf.async(
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); }
      );
    }
    sf.corun();
  });

  taskflow.emplace([&] (tf::Runtime& sf){
    for(int i=0; i<1000; i++) {
      sf.async([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
    }
    sf.corun();
  });

  A.precede(S1, S2);
  B.succeed(S1, S2);

  executor.run(taskflow).wait();

  REQUIRE(counter == 4005);
}

TEST_CASE("RuntimeAsync.1thread") {
  runtime_async(1);
}

TEST_CASE("RuntimeAsync.2threads") {
  runtime_async(2);
}

TEST_CASE("RuntimeAsync.3threads") {
  runtime_async(3);
}

TEST_CASE("RuntimeAsync.4threads") {
  runtime_async(4);
}

TEST_CASE("RuntimeAsync.5threads") {
  runtime_async(5);
}

TEST_CASE("RuntimeAsync.6threads") {
  runtime_async(6);
}

TEST_CASE("RuntimeAsync.7threads") {
  runtime_async(7);
}

TEST_CASE("RuntimeAsync.8threads") {
  runtime_async(8);
}

TEST_CASE("RuntimeAsync.11threads") {
  runtime_async(11);
}
