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

TEST_CASE("TaskFlow.Basics"){

  const auto num_workers = std::max(4u, std::thread::hardware_concurrency());

  tf::Taskflow<> tf(num_workers);
  REQUIRE(tf.num_workers() == num_workers);
  
  constexpr size_t num_tasks {100};

  std::atomic<int> counter {0};
  std::vector<tf::Taskflow<>::TaskBuilder> keys;
  std::vector<std::pair<tf::Taskflow<>::TaskBuilder, std::future<void>>> tasks;

  SUBCASE("Sequential tasks"){
    for(size_t i=0;i<num_tasks;i++){
      if(i%2 == 0){
        tasks.emplace_back( std::move(tf.emplace([&counter]() { REQUIRE(counter == 0); counter += 1;})) );
      }
      else{
        tasks.emplace_back( std::move(tf.emplace([&counter]() { REQUIRE(counter == 1); counter -= 1;})) );
      }
      if(i>0){
        tf.precede(std::get<0>(tasks[i-1]), std::get<0>(tasks[i]));
      }
    }
    tf.wait_for_all();
  }

  ///*
  SUBCASE("Without precedence"){
    for(size_t i=0;i<num_tasks;i++){
      tasks.emplace_back( std::move(tf.emplace([&counter]() {counter += 1;})) );
    }
    REQUIRE(tf.num_tasks() == num_tasks);

    tf.wait_for_all();
    REQUIRE(counter == num_tasks);
    REQUIRE(tf.num_tasks() == 0);
  }

  SUBCASE("With precedence"){
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

  SUBCASE("Silent emplace without precedence"){
    for(size_t i=0;i<num_tasks;i++){
      keys.emplace_back(tf.silent_emplace([&counter]() {counter += 1;}));
    }
    tf.wait_for_all();
    REQUIRE(counter == num_tasks);
    REQUIRE(tf.num_tasks() == 0);
  }

  SUBCASE("Silent emplace with precedence"){
    for(size_t i=0;i<num_tasks;i++){
      keys.emplace_back(tf.silent_emplace([&counter, i]() { REQUIRE(i == counter); counter += 1;}));
      if(i>0){
        tf.precede(keys[i-1], keys[i]);
      }
    }
    tf.wait_for_all();
    REQUIRE(counter == num_tasks);
    REQUIRE(tf.num_tasks() == 0);
  }

  SUBCASE("Dispatch"){
    for(size_t i=0;i<num_tasks;i++){
      keys.emplace_back(tf.silent_emplace([&counter]() {counter += 1;}));
    }
    auto fu = tf.dispatch();
    fu.get();
    REQUIRE(counter == num_tasks);
    REQUIRE(tf.num_tasks() == 0);
  }

  SUBCASE("Silent dispatch"){
    using namespace std::chrono_literals;
    for(size_t i=0;i<num_tasks;i++){
      keys.emplace_back(tf.silent_emplace([&counter]() {counter += 1;}));
    }
    tf.silent_dispatch();
    std::this_thread::sleep_for(1s);
    REQUIRE(counter == num_tasks);
    REQUIRE(tf.num_tasks() == 0);
  }
 
  SUBCASE("Broadcast"){
    using namespace std::chrono_literals;
    auto src = tf.silent_emplace([&counter]() {counter -= 1;});
    for(size_t i=1;i<num_tasks;i++){
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
    tf.gather(keys,dst);
    tf.wait_for_all();
    REQUIRE(counter == num_tasks - 1);
    REQUIRE(tf.num_tasks() == 0);
  }

  SUBCASE("Broadcast + Gather"){
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
    using namespace std::chrono_literals;
    for(size_t i=0;i<num_tasks;i++){
      keys.emplace_back(tf.silent_emplace([&counter, i]() { REQUIRE(counter == i); counter += 1;}));
    }
    tf.linearize(keys);
    tf.wait_for_all();
    REQUIRE(counter == num_tasks);
    REQUIRE(tf.num_tasks() == 0);
  }

  SUBCASE("Broadcast + Linearize + Gather "){
    using namespace std::chrono_literals;
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


