//2018/8/31 contributed by Guannan
// Examples to test throughput of various theadpools

#include <taskflow/threadpool/threadpool.hpp>
#include <taskflow/threadpool/proactive_threadpool.hpp>
#include <chrono>
#include <atomic>
#include <thread>

void benchmark_empty_jobs() {

  std::cout << "Testing threadpool throughput on empty jobs..." << std::endl;

  unsigned thread_num = 4;
  unsigned int task_num = 10000000;
  
  auto start = std::chrono::high_resolution_clock::now();

  tf::ProactiveThreadpool pool(thread_num);
  for(size_t i=0; i<task_num; i++){
    pool.silent_async([](){}); 
  }
  pool.shutdown();
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "ProactiveThreadpool elapsed time: " << elapsed.count() << std::endl;

  start = std::chrono::high_resolution_clock::now();

  tf::Threadpool tf_pool(thread_num);
  for(size_t i=0; i<task_num; i++){
    tf_pool.silent_async([](){}); 
  }
  tf_pool.shutdown();
  
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Basic elapsed time: " << elapsed.count() << std::endl;
  std::cout << std::endl;
}


void benchmark_atomic_add() {

  std::cout << "Testing threadpool throughput on atomic add..." << std::endl;
  
  unsigned thread_num = 4;
  unsigned int task_num = 10000000;
  
  std::atomic<int> counter(0);
  auto start = std::chrono::high_resolution_clock::now();
  
  tf::ProactiveThreadpool pool(thread_num);
  for(size_t i=0; i<task_num; i++){
    pool.silent_async([&counter](){ counter++; }); 
  }
  pool.shutdown();
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "ProactiveThreadpool elapsed time: " << elapsed.count() << std::endl;

  counter = 0;
  start = std::chrono::high_resolution_clock::now();
  tf::Threadpool tf_pool(thread_num);

  for(size_t i=0; i<task_num; i++){
    tf_pool.silent_async([&counter](){ counter++; }); 
  }
  tf_pool.shutdown();
  
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Basic elapsed time: " << elapsed.count() << std::endl;
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {

  benchmark_empty_jobs();
  benchmark_atomic_add();

}
