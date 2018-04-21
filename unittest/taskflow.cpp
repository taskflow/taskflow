#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"

#include <taskflow.hpp>
#include <vector>
#include <utility>
#include <chrono>

// --------------------------------------------------------
// Testcase: Taskflow.Builder
// --------------------------------------------------------
TEST_CASE("Taskflow.Builder"){

  constexpr auto num_workers = 4;
  constexpr auto num_tasks = 100;

  tf::Taskflow tf(num_workers);
  REQUIRE(tf.num_workers() == num_workers);

  std::atomic<int> counter {0};
  std::vector<tf::Taskflow::TaskBuilder> keys;
  std::vector<std::pair<tf::Taskflow::TaskBuilder, std::future<void>>> tasks;

  SUBCASE("EmbarrassinglyParallel"){

    for(size_t i=0;i<num_tasks;i++) {
      tasks.emplace_back(tf.emplace([&counter]() {counter += 1;}));
    }
    REQUIRE(tf.num_tasks() == num_tasks);
    tf.wait_for_all();
    REQUIRE(counter == num_tasks);
    REQUIRE(tf.num_tasks() == 0);

    counter = 0;
    
    for(size_t i=0;i<num_tasks;i++){
      keys.emplace_back(tf.silent_emplace([&counter]() {counter += 1;}));
    }
    REQUIRE(tf.num_tasks() == num_tasks);
    tf.wait_for_all();
    REQUIRE(counter == num_tasks);
    REQUIRE(tf.num_tasks() == 0);
  }
  
  SUBCASE("BinarySequence"){
    for(size_t i=0;i<num_tasks;i++){
      if(i%2 == 0){
        tasks.emplace_back(tf.emplace([&counter]() { REQUIRE(counter == 0); counter += 1;}));
      }
      else{
        tasks.emplace_back(tf.emplace([&counter]() { REQUIRE(counter == 1); counter -= 1;}));
      }
      if(i>0){
        tasks[i-1].first.precede(tasks[i].first);
      }
    }
    tf.wait_for_all();
  }

  SUBCASE("LinearCounter"){
    for(size_t i=0;i<num_tasks;i++){
      tasks.emplace_back( std::move(tf.emplace([&counter, i]() { REQUIRE(counter == i); counter += 1;})) );
      if(i>0){
        tf.precede(std::get<0>(tasks[i-1]), std::get<0>(tasks[i]));
      }
    }
    tf.wait_for_all();
    REQUIRE(counter == num_tasks);
    REQUIRE(tf.num_tasks() == 0);
  }
 
  SUBCASE("Broadcast"){
    auto src = tf.silent_emplace([&counter]() {counter -= 1;});
    for(size_t i=1; i<num_tasks; i++){
      keys.emplace_back(tf.silent_emplace([&counter]() {REQUIRE(counter == -1);}));
    }
    tf.broadcast(src, keys);
    tf.wait_for_all();
    REQUIRE(counter == - 1);
    REQUIRE(tf.num_tasks() == 0);
  }

  SUBCASE("Gather"){
    auto dst = tf.silent_emplace([&counter]() { REQUIRE(counter == num_tasks - 1);});
    for(size_t i=1;i<num_tasks;i++){
      keys.emplace_back(tf.silent_emplace([&counter]() {counter += 1;}));
    }
    dst.gather(keys);
    tf.wait_for_all();
    REQUIRE(counter == num_tasks - 1);
    REQUIRE(tf.num_tasks() == 0);
  }

  SUBCASE("MapReduce"){
    auto src = tf.silent_emplace([&counter]() {counter = 0;});
    for(size_t i=0;i<num_tasks;i++){
      keys.emplace_back(tf.silent_emplace([&counter]() {counter += 1;}));
    }
    tf.broadcast(src, keys);
    auto dst = tf.silent_emplace([&counter]() { REQUIRE(counter == num_tasks);});
    tf.gather(keys, dst);
    tf.wait_for_all();
    REQUIRE(tf.num_tasks() == 0);
  }

  SUBCASE("Linearize"){
    for(size_t i=0;i<num_tasks;i++){
      keys.emplace_back(tf.silent_emplace([&counter, i]() { REQUIRE(counter == i); counter += 1;}));
    }
    tf.linearize(keys);
    tf.wait_for_all();
    REQUIRE(counter == num_tasks);
    REQUIRE(tf.num_tasks() == 0);
  }

  SUBCASE("Kite"){
    auto src = tf.silent_emplace([&counter]() {counter = 0;});
    for(size_t i=0;i<num_tasks;i++){
      keys.emplace_back(tf.silent_emplace([&counter, i]() { REQUIRE(counter == i); counter += 1; }));
    }
    tf.broadcast(src, keys);
    tf.linearize(keys);
    auto dst = tf.silent_emplace([&counter]() { REQUIRE(counter == num_tasks);});
    tf.gather(keys, dst);
    tf.wait_for_all();
    REQUIRE(tf.num_tasks() == 0);
  }
}

// --------------------------------------------------------
// Testcase: Taskflow.Dispatch
// --------------------------------------------------------
TEST_CASE("Taskflow.Dispatch") {
    
  using namespace std::chrono_literals;
  
  constexpr auto num_workers = 4;
  constexpr auto num_tasks = 100;
  
  tf::Taskflow tf(num_workers);
  REQUIRE(tf.num_workers() == num_workers);

  std::atomic<int> counter {0};
  std::vector<tf::Taskflow::TaskBuilder> keys;
    
  for(size_t i=0;i<num_tasks;i++){
    keys.emplace_back(tf.silent_emplace([&counter]() {counter += 1;}));
  }

  SUBCASE("Dispatch"){
    auto fu = tf.dispatch();
    REQUIRE(tf.num_tasks() == 0);
    REQUIRE(fu.wait_for(1s) == std::future_status::ready);
    REQUIRE(counter == num_tasks);
  }

  SUBCASE("SilentDispatch"){
    tf.silent_dispatch();
    REQUIRE(tf.num_tasks() == 0);
    std::this_thread::sleep_for(1s);
    REQUIRE(counter == num_tasks);
  }

  SUBCASE("WaitForAll") {
    tf.wait_for_all();
    REQUIRE(tf.num_tasks() == 0);
    REQUIRE(counter == num_tasks); 
  }
}





