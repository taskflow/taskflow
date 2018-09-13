// 2018/09/10 - added by Tsung-Wei Huang
//   - added tests for SpeculativeThreadpool
//   - added dynamic tasking tests
//   - added spawn and shutdown tests
//
// 2018/09/02 - created by Guannan
//   - test_silent_async
//   - test_async
//   - test_wait_for_all

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/threadpool/threadpool.hpp>

// Procedure: test_owner
template <typename ThreadpoolType>
void test_owner(ThreadpoolType& tp) {
  
  REQUIRE(tp.is_owner()); 
  
  tp.silent_async([&](){
    if(tp.num_workers() == 0) {
      REQUIRE(tp.is_owner());
    }
    else {
      REQUIRE(!tp.is_owner());
    }
  });

  std::vector<std::thread> threads;
  for(int i=0; i<10; ++i) {
    threads.emplace_back([&] () {
      REQUIRE(!tp.is_owner());
    });
  }
  for(auto& t : threads) {
    t.join();
  }
}

// Procedure: test_silent_async
template <typename ThreadpoolType>
void test_silent_async(ThreadpoolType& tp, const size_t num_tasks) {

  std::atomic<size_t> counter{0};
  
  size_t sum = 0;
  for(size_t i=0; i<num_tasks; i++){
    sum += i;
    tp.silent_async([i=i, &counter](){ counter += i; });
  }
  tp.wait_for_all(); //make sure all silent threads end

  REQUIRE(counter == sum);
}

// Procedure: test_async
template <typename ThreadpoolType>
void test_async(ThreadpoolType& tp, const size_t num_tasks){
 
  std::vector<std::future<int>> int_future;
  std::vector<int> int_result;

  for(size_t i=0; i<num_tasks; i++){
    int_future.emplace_back(tp.async(
      [size = i](){
        int sum = 0;
        for(int i=0; i<=static_cast<int>(size); i++){
          sum += i;
        }       
        return sum;
      }
      )
    );
    
    int sum_result = 0;
    for(int j=0; j<=static_cast<int>(i); j++) sum_result += j;
    int_result.push_back(sum_result);
  }
  
  REQUIRE(int_future.size() == int_result.size());

  for(size_t i=0; i<int_future.size(); i++){
    REQUIRE(int_future[i].get() == int_result[i]);
  } 
}

// Procedure: test_wait_for_all
template <typename ThreadpoolType>
void test_wait_for_all(ThreadpoolType& tp){

  using namespace std::literals::chrono_literals;

  const size_t num_workers = tp.num_workers();
  const size_t num_tasks = 20;
  std::atomic<size_t> counter;

  for(int i=0; i<10; ++i) {

    counter = 0;
    
    for(size_t i=0; i<num_tasks; i++){
      tp.silent_async([&counter](){ 
        std::this_thread::sleep_for(200us);
        counter++; 
      });
    }
    REQUIRE(counter <= num_tasks);  // pay attention to the case of 0 worker

    tp.wait_for_all();

    REQUIRE(counter == num_tasks);
    REQUIRE(tp.num_workers() == num_workers);
  }
}

// Procedure: test_spawn_shutdown
template <typename T>
void test_spawn_shutdown(T& tp) {
  
  using namespace std::literals::chrono_literals;

  const size_t num_workers = tp.num_workers();
  const size_t num_tasks = 1024;

  tp.spawn(num_workers);
  REQUIRE(tp.num_workers() == num_workers * 2);

  tp.shutdown();
  REQUIRE(tp.num_workers() == 0);
  REQUIRE(tp.num_tasks() == 0);

  tp.spawn(num_workers);
  REQUIRE(tp.num_workers() == num_workers);

  for(int k=0; k<5; ++k) {

    REQUIRE(tp.num_workers() == num_workers);
    
    std::atomic<size_t> counter {0};

    for(size_t i=0; i<num_tasks; i++){
      tp.silent_async([&] () { 
        counter++;
        tp.silent_async([&] () {
          counter++;
          tp.silent_async([&] () {
            counter++;
            tp.silent_async([&] () {
              counter++;
            }); 
          });
        });
      });
    }

    tp.spawn(num_workers);

    REQUIRE(counter <= num_tasks * 4);

    tp.shutdown();

    REQUIRE(counter == num_tasks * 4);
    REQUIRE(tp.num_workers() == 0);
    REQUIRE(tp.num_tasks() == 0);

    tp.spawn(num_workers);
    REQUIRE(tp.num_workers() == num_workers);
    REQUIRE(tp.num_tasks() == 0);
  }
}

// Procedure: test_dynamic_tasking
template <typename T>
void test_dynamic_tasking(T& threadpool) {

  std::atomic<size_t> sum {0};
  std::atomic<size_t> cnt {0};

  std::function<void(int)> insert;
  std::promise<int> promise;
  auto future = promise.get_future();
  
  insert = [&threadpool, &insert, &sum, &promise, &cnt] (int i) {
    if(i > 0) {
      ++cnt;
      threadpool.silent_async([i=i-1, &insert] () {
        insert(i);
      });
    }
    else {
      if(auto s = ++sum; s == threadpool.num_workers()) {
        promise.set_value(1);
      }
    }
  };

  if(auto W = threadpool.num_workers(); W > 0) {
    for(size_t i=0; i<threadpool.num_workers(); i++){
      insert(100);
    }
  }
  else {
    promise.set_value(1);
  }

  // synchronize until all tasks finish
  threadpool.wait_for_all();
  
  REQUIRE(cnt == 100 * threadpool.num_workers());
  REQUIRE(future.get() == 1);
  REQUIRE(sum == threadpool.num_workers());
}

// --------------------------------------------------------
// Testcase: SimpleThreadpool
// --------------------------------------------------------
TEST_CASE("SimpleThreadpool" * doctest::timeout(300)) {

  const size_t num_tasks = 100;

  SUBCASE("PlaceTask"){
    for(unsigned i=0; i<=4; ++i) {
      tf::SimpleThreadpool tp(i);
      test_async(tp, num_tasks);
      test_silent_async(tp, num_tasks);
    }
  }
  
  SUBCASE("WaitForAll"){
    for(unsigned i=0; i<=4; ++i) {
      tf::SimpleThreadpool tp(i);
      test_wait_for_all(tp);
    }
  }
}

// --------------------------------------------------------
// Testcase: ProactiveThreadpool
// --------------------------------------------------------
TEST_CASE("ProactiveThreadpool" * doctest::timeout(300)) {

  const size_t num_tasks = 100;

  SUBCASE("Owner") {
    for(unsigned i=0; i<=4; ++i) {
      tf::ProactiveThreadpool tp(i);
      test_owner(tp);
    }
  }

  SUBCASE("PlaceTask"){
    for(unsigned i=0; i<=4; ++i) {
      tf::ProactiveThreadpool tp(i);
      test_async(tp, num_tasks);
      test_silent_async(tp, num_tasks);
    }
  }
  
  SUBCASE("WaitForAll"){
    for(unsigned i=0; i<=4; ++i) {
      tf::ProactiveThreadpool tp(i);
      test_wait_for_all(tp);
    }
  }
  
  SUBCASE("SpawnShutdown") {
    for(unsigned i=0; i<=4; ++i) {
      tf::ProactiveThreadpool tp(i);
      test_spawn_shutdown(tp);
    }
  }

  SUBCASE("DynamicTasking") {
    for(unsigned i=0; i<=4; ++i) {
      tf::ProactiveThreadpool tp(i);
      test_dynamic_tasking(tp);
    }
  }
}

// --------------------------------------------------------
// Testcase: SpeculativeThreadpool
// --------------------------------------------------------
TEST_CASE("SpeculativeThreadpool" * doctest::timeout(300)) {

  const size_t num_tasks = 100;
  
  SUBCASE("Owner") {
    for(unsigned i=0; i<=4; ++i) {
      tf::SpeculativeThreadpool tp(i);
      test_owner(tp);
    }
  }

  SUBCASE("PlaceTask"){
    for(unsigned i=0; i<=4; ++i) {
      tf::SpeculativeThreadpool tp(i);
      test_async(tp, num_tasks);
      test_silent_async(tp, num_tasks);
    }
  }
  
  SUBCASE("WaitForAll"){
    for(unsigned i=0; i<=4; ++i) {
      tf::SpeculativeThreadpool tp(i);
      test_wait_for_all(tp);
    }
  }
  
  SUBCASE("SpawnShutdown") {
    for(unsigned i=0; i<=4; ++i) {
      tf::SpeculativeThreadpool tp(i);
      test_spawn_shutdown(tp);
    }
  }

  SUBCASE("DynamicTasking") {
    for(unsigned i=0; i<=4; ++i) {
      tf::SpeculativeThreadpool tp(i);
      test_dynamic_tasking(tp);
    }
  }
}

