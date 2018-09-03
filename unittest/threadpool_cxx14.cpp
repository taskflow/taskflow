// 2018/09/02 - contributed by Glen Fraser
//   - added wait_for_all test
//
// 2018/08/28 - contributed by Glen Fraser
//   - added basic test

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/threadpool/threadpool_cxx14.hpp>
#include <vector>
#include <utility>
#include <chrono>
#include <limits.h>

// --------------------------------------------------------
// Testcase: Threadpool.Basic
// --------------------------------------------------------
TEST_CASE("Threadpool.Basic" * doctest::timeout(5)) {

  using namespace std::chrono_literals;
  
  size_t num_workers = 4;
  size_t num_tasks = 1000;

  tf::Threadpool tp(static_cast<unsigned>(num_workers));
  REQUIRE(tp.num_workers() == num_workers);

  std::atomic<int> counter {0};
  std::vector<std::future<void>> void_tasks;

  for(size_t i=0;i<num_tasks;i++){
    void_tasks.emplace_back(tp.async([&counter, &tp]() {
      REQUIRE(tp.is_worker());
      counter += 1;
    }));
  }

  REQUIRE(!tp.is_worker());

  {
    auto& fu = void_tasks.front();
    REQUIRE(fu.wait_for(1s) == std::future_status::ready);
    REQUIRE(tp.num_tasks() < num_tasks);
  }

  for (auto& fu : void_tasks) {
      fu.get();
  }
  REQUIRE(tp.num_tasks() == 0);
  REQUIRE(counter == num_tasks); 

  tp.spawn(1);
  REQUIRE(tp.num_workers() == num_workers + 1);

  tp.shutdown();
  REQUIRE(tp.num_workers() == 0);

  tp.spawn(num_workers);
  REQUIRE(tp.num_workers() == num_workers);
}

// --------------------------------------------------------
// Testcase: Threadpool.WaitForAll
// --------------------------------------------------------
TEST_CASE("Threadpool.WaitForAll" * doctest::timeout(5)) {

  using namespace std::chrono_literals;

  size_t num_workers = 4;
  size_t num_tasks = 100;

  tf::Threadpool tp(static_cast<unsigned>(num_workers));

  std::atomic<int> counter{ 0 };
  std::vector<std::future<void>> void_tasks;

  for (size_t i = 0; i<num_tasks; i++) {
    void_tasks.emplace_back(tp.async([=, &counter]() {
      std::this_thread::sleep_for((num_tasks - i) * 1ms);
      ++counter;
    }));
  }

  REQUIRE(void_tasks.front().wait_for(1s) == std::future_status::ready);
  REQUIRE(void_tasks.back().wait_for(0s) != std::future_status::ready);
  tp.wait_for_all();

  // Last task should be done by now...
  REQUIRE(void_tasks.back().wait_for(0s) == std::future_status::ready);
  // In fact, all tasks should be done!
  for (auto& fu : void_tasks) {
    REQUIRE(fu.wait_for(0s) == std::future_status::ready);
  }

  REQUIRE(tp.num_tasks() == 0);
  REQUIRE(counter == num_tasks);
}
