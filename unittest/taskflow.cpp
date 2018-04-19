#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <taskflow.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <utility>
#include <future>
#include <tuple>
#include <thread>
#include <chrono>

TEST_CASE("TaskFlow"){

  tf::Taskflow tf(std::thread::hardware_concurrency());
  REQUIRE(tf.num_workers() == std::thread::hardware_concurrency());
  
  const size_t nof_tasks {100};
  std::atomic<int> counter {0};
  std::vector<int64_t> keys;
  std::vector<std::tuple<int64_t,std::future<void>>> tasks;

  SUBCASE("Without precedence"){
    for(size_t i=0;i<nof_tasks;i++){
      tasks.emplace_back( std::move(tf.emplace([&counter]() {counter += 1;})) );
    }
    REQUIRE(tf.num_tasks() == nof_tasks);

    tf.wait_for_all();
    REQUIRE(counter == nof_tasks);
    REQUIRE(tf.num_tasks() == 0);
  }

  SUBCASE("With precedence"){
    for(size_t i=0;i<nof_tasks;i++){
      tasks.emplace_back( std::move(tf.emplace([&counter]() {counter += 1;})) );
      if(i>0){
        tf.precede(std::get<0>(tasks[i-1]), std::get<0>(tasks[i]));
      }
    }
    tf.wait_for_all();
    REQUIRE(counter == nof_tasks);
    REQUIRE(tf.num_tasks() == 0);
  }

  SUBCASE("Silent emplace without precedence"){
    for(size_t i=0;i<nof_tasks;i++){
      keys.emplace_back(tf.silent_emplace([&counter]() {counter += 1;}));
    }
    tf.wait_for_all();
    REQUIRE(counter == nof_tasks);
    REQUIRE(tf.num_tasks() == 0);
  }

  SUBCASE("Silent emplace with precedence"){
    for(size_t i=0;i<nof_tasks;i++){
      keys.emplace_back(tf.silent_emplace([&counter]() {counter += 1;}));
      if(i>0){
        tf.precede(keys[i-1], keys[i]);
      }
    }
    tf.wait_for_all();
    REQUIRE(counter == nof_tasks);
    REQUIRE(tf.num_tasks() == 0);
  }

  SUBCASE("Dispatch"){
    using namespace std::chrono_literals;
    for(size_t i=0;i<nof_tasks;i++){
      keys.emplace_back(tf.silent_emplace([&counter]() {counter += 1;}));
    }
    auto fu = tf.dispatch();
    fu.get();
    REQUIRE(counter == nof_tasks);
    REQUIRE(tf.num_tasks() == 0);
  }

  SUBCASE("Silent dispatch"){
    using namespace std::chrono_literals;
    for(size_t i=0;i<nof_tasks;i++){
      keys.emplace_back(tf.silent_emplace([&counter]() {counter += 1;}));
    }
    tf.silent_dispatch();
    std::this_thread::sleep_for(1ms);
    REQUIRE(counter == nof_tasks);
    REQUIRE(tf.num_tasks() == 0);
  }
  

}


