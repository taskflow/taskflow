#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <taskflow.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <utility>
#include <future>
#include <tuple>

TEST_CASE("TaskFlow"){
  tf::Taskflow tf(std::thread::hardware_concurrency());
  REQUIRE(tf.num_workers() == std::thread::hardware_concurrency());
  
  const size_t nof_tasks {100};
  std::atomic<size_t> counter {0};
 
  std::vector<std::tuple<int64_t,std::future<void>>> tasks;
  for(size_t i=0;i<nof_tasks;i++){
    tasks.emplace_back( std::move(tf.emplace([&counter]() {counter += 1;})) );
  }
  REQUIRE(tf.num_tasks() == nof_tasks);

  tf.wait_for_all();
  REQUIRE(counter == nof_tasks);

}


