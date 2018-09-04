// 2018/8/31 contributed by Guannan
//
// Examples to test different threadpool implementations:
//   - SimpleThreadpool
//   - ProactiveThreadpool

#include <taskflow/threadpool/threadpool.hpp>
#include <chrono>
#include <random>

// Procedure: benchmark_empty_jobs
void benchmark_empty_jobs() {

  std::cout << "Benchmarking threadpool throughput on empty jobs ...\n";

  unsigned thread_num = 4;
  unsigned int task_num = 10000000;
  
  auto start = std::chrono::high_resolution_clock::now();

  tf::ProactiveThreadpool proactive(thread_num);
  for(size_t i=0; i<task_num; i++){
    proactive.silent_async([](){}); 
  }
  proactive.shutdown();
  
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "ProactiveThreadpool elapsed time: " << elapsed.count() << " ms\n";

  start = std::chrono::high_resolution_clock::now();

  tf::SimpleThreadpool simple(thread_num);
  for(size_t i=0; i<task_num; i++){
    simple.silent_async([](){}); 
  }
  simple.shutdown();
  
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "SimpleThreadpool elapsed time: " << elapsed.count() << " ms\n";
}

// Procedure: benchmark_atomic_add
void benchmark_atomic_add() {

  std::cout << "Benchmarking threadpool throughput on atomic add ...\n";
  
  unsigned thread_num = 4;
  unsigned int task_num = 10000000;
  
  std::atomic<int> counter(0);
  auto start = std::chrono::high_resolution_clock::now();
  
  tf::ProactiveThreadpool proactive(thread_num);
  for(size_t i=0; i<task_num; i++){
    proactive.silent_async([&counter](){ counter++; }); 
  }
  proactive.shutdown();
  
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "ProactiveThreadpool elapsed time: " << elapsed.count() << " ms\n";

  counter = 0;
  start = std::chrono::high_resolution_clock::now();
  tf::SimpleThreadpool simple(thread_num);

  for(size_t i=0; i<task_num; i++){
    simple.silent_async([&counter](){ counter++; }); 
  }
  simple.shutdown();
  
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "SimpleThreadpool elapsed time: " << elapsed.count() << " ms\n";
}

// Function: main
int main(int argc, char* argv[]) {

  benchmark_empty_jobs();
  benchmark_atomic_add();
  
  return 0;
}
