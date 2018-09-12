// 2018/09/03 - added by Tsung-Wei Huang
//   - refactored ProactiveThreadpool unittest
//   - added tests for SimpleThreadpool
//
// 2018/09/02 - created by Guannan
//   - test_silent_async
//   - test_async
//   - test_wait_for_all

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/threadpool/threadpool.hpp>

// Procedure: test_silent_async
template <typename ThreadpoolType>
void test_silent_async(ThreadpoolType& tp, const size_t task_num) {

  std::atomic<size_t> counter{0};
  
  size_t sum = 0;
  for(size_t i=0; i<task_num; i++){
    sum += i;
    tp.silent_async([i=i, &counter](){ counter += i; });
  }
  tp.shutdown(); //make sure all silent threads end

  REQUIRE(counter == sum);

}

// Procedure: test_async
template <typename ThreadpoolType>
void test_async(ThreadpoolType& tp, const size_t task_num){
 
  std::vector<std::future<int>> int_future;
  std::vector<int> int_result;

  for(size_t i=0; i<task_num; i++){
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

  const size_t worker_num = tp.num_workers();
  const size_t task_num = 20;
  std::atomic<size_t> counter{0};
  
  for(size_t i=0; i<task_num; i++){
    tp.silent_async([&counter](){ 
      std::this_thread::sleep_for(200us);
      counter++; 
    });
  }
  REQUIRE(counter <= task_num);  // pay attention to the case of 0 worker

  tp.shutdown();

  REQUIRE(counter == task_num);
  REQUIRE(tp.num_workers() == 0);

  tp.spawn(static_cast<unsigned>(worker_num));
  REQUIRE(tp.num_workers() == worker_num);

  counter = 0;
  for(size_t i=0; i<task_num; i++){
    tp.silent_async([&counter](){ 
      std::this_thread::sleep_for(200us);
      counter++; 
    });
  }
  tp.shutdown();

  REQUIRE(counter == task_num);
}

// --------------------------------------------------------
// Testcase: SimpleThreadpool
// --------------------------------------------------------
TEST_CASE("SimpleThreadpool" * doctest::timeout(300)) {

  const size_t task_num = 100;

  SUBCASE("PlaceTask"){
    for(unsigned i=0; i<=4; ++i) {
      tf::SimpleThreadpool tp(i);
      test_async(tp, task_num);
      test_silent_async(tp, task_num);
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

  const size_t task_num = 100;

  SUBCASE("PlaceTask"){
    for(unsigned i=0; i<=4; ++i) {
      tf::ProactiveThreadpool tp(i);
      test_async(tp, task_num);
      test_silent_async(tp, task_num);
    }
  }
  
  SUBCASE("WaitForAll"){
    for(unsigned i=0; i<=4; ++i) {
      tf::ProactiveThreadpool tp(i);
      test_wait_for_all(tp);
    }
  }
}



