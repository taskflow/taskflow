// 2018/8/31 contributed by Guannan
//
// Examples to test different threadpool implementations:
//   - SimpleThreadpool
//   - ProactiveThreadpool

#include <taskflow/threadpool/threadpool.hpp>
#include <chrono>
#include <random>

const int thread_num = 4;
const int task_num   = 16;

// ----------------------------------------------------------------------------

// Procedure: linear_insertions
template <typename T>
auto linear_insertions() {
  
  auto beg = std::chrono::high_resolution_clock::now();
  
  T threadpool(thread_num);

  std::atomic<size_t> sum {0};

  std::function<void(int)> insert;
  std::promise<void> promise;
  auto future = promise.get_future();
  
  insert = [&threadpool, &insert, &sum, &promise] (int i) {
    if(i > 0) {
      threadpool.silent_async([i=i-1, &insert] () {
        insert(i);
      });
    }
    else {
      if(++sum; sum == thread_num) {
        promise.set_value();
      }
    }
  };

  for(size_t i=0; i<thread_num; i++){
    insert(task_num / thread_num);
  }
  
  // synchronize until all tasks finish
  threadpool.shutdown();

  //future.get();
  //assert(sum == thread_num);
  
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// Procedure: benchmark_linear_insertions
void benchmark_linear_insertions() {

  std::cout << "==== Linear Insertions ====\n";

  //std::cout << "Proactive threadpool takes: " 
  //          << linear_insertions<tf::ProactiveThreadpool>() << " ms\n";

  std::cout << "Simple threadpool takes: " 
            << linear_insertions<tf::SimpleThreadpool>() << " ms\n";
}

// ----------------------------------------------------------------------------

// Function: empty_jobs
template <typename T>
auto empty_jobs() {
  
  auto beg = std::chrono::high_resolution_clock::now();

  T threadpool(thread_num);

  for(size_t i=0; i<task_num; i++){
    threadpool.silent_async([](){}); 
  }

  threadpool.shutdown();
  
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// Procedure: benchmark_empty_jobs
void benchmark_empty_jobs() {
  std::cout << "==== Empty Jobs ====\n";

  std::cout << "Proactive threadpool takes: " 
            << empty_jobs<tf::ProactiveThreadpool>() << " ms\n";

  std::cout << "Simple threadpool takes: " 
            << empty_jobs<tf::SimpleThreadpool>() << " ms\n";
}

// ----------------------------------------------------------------------------

// Function: atomic_add
template <typename T>
auto atomic_add() {
  
  std::atomic<int> counter(0);
  auto beg = std::chrono::high_resolution_clock::now();
  
  T threadpool(thread_num);
  for(size_t i=0; i<task_num; i++){
    threadpool.silent_async([&counter](){ counter++; }); 
  }
  threadpool.shutdown();

  assert(counter == task_num);
  
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// Procedure: benchmark_atomic_add
void benchmark_atomic_add() {
  std::cout << "==== Atomic Add ====\n";

  std::cout << "Proactive threadpool takes: " 
            << atomic_add<tf::ProactiveThreadpool>() << " ms\n";

  std::cout << "Simple threadpool takes: " 
            << atomic_add<tf::SimpleThreadpool>() << " ms\n";
}

// Function: main
int main(int argc, char* argv[]) {

  benchmark_linear_insertions();
  //benchmark_empty_jobs();
  //benchmark_atomic_add();
  
  return 0;
}
