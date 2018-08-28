#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"

#include <taskflow/threadpool/threadpool_cxx14.hpp>
#include <vector>
#include <utility>
#include <chrono>
#include <limits.h>

// --------------------------------------------------------
// Testcase: Cxx14.Threadpool
// --------------------------------------------------------
TEST_CASE("Cxx14.Threadpool" * doctest::timeout(5)) {
    
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

