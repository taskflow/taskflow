// conbributed by Guannan

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/threadpool/threadpool.hpp>
#include <taskflow/threadpool/proactive_threadpool.hpp>
#include <atomic>
#include <future>
#include <vector>

template <typename ThreadpoolType>
void test_threadpool_silent_async(ThreadpoolType& tp, const size_t task_num) {

  std::atomic<size_t> counter{0};
  
  size_t sum = 0;
  for(size_t i=0; i<task_num; i++){
    sum += i;
    tp.silent_async([i=i, &counter](){ counter += i; });
  }
  tp.shutdown(); //make sure all silent threads end

  CHECK(counter == sum);

}

template <typename ThreadpoolType>
void test_threadpool_async(ThreadpoolType& tp, const size_t task_num){
 
  std::vector<std::future<int>> int_future;
  std::vector<int> int_result;

  for(size_t i=0; i<task_num; i++){
    int_future.emplace_back(tp.async(
      [size = i](){
        int sum = 0;
        for(size_t i=0; i<=size; i++){
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
  
  CHECK(int_future.size() == int_result.size());

  for(size_t i=0; i<int_future.size(); i++){
    CHECK(int_future[i].get() == int_result[i]);
  } 
 
}

// --------------------------------------------------------
// Testcase: ProactiveThreadpool
// --------------------------------------------------------
TEST_CASE("Threadpool.ProactiveThreadpool" * doctest::timeout(5)) {

  size_t task_num = 100;

  SUBCASE("EmptyWorkers"){
    tf::ProactiveThreadpool tp(0);
    test_threadpool_async(tp, task_num);
    test_threadpool_silent_async(tp, task_num); 
  }

  SUBCASE("NoneEmptyWorkers"){
    tf::ProactiveThreadpool tp(4);
    test_threadpool_async(tp, task_num);
    test_threadpool_silent_async(tp, task_num);
  }

}

