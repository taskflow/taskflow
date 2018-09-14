// 2018/8/31 contributed by Guannan
//
// Examples to test different threadpool implementations:
//   - SimpleThreadpool
//   - ProactiveThreadpool

#include <taskflow/threadpool/threadpool.hpp>
#include <chrono>
#include <random>

// ----------------------------------------------------------------------------

// Procedure: linear_insertions
template <typename T>
auto linear_insertions() {

  const int num_threads = 4;
  const int num_tasks   = 2000000;
  
  auto beg = std::chrono::high_resolution_clock::now();
  
  T threadpool(num_threads);

  std::atomic<size_t> sum {0};

  std::function<void(int)> insert;
  std::promise<int> promise;
  auto future = promise.get_future();
  
  insert = [&threadpool, &insert, &sum, &promise] (int i) {
    if(i > 0) {
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

  for(size_t i=0; i<num_threads; i++){
    insert(num_tasks / num_threads);
  }

  // synchronize until all tasks finish
  threadpool.wait_for_all();
  
  assert(future.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
  assert(future.get() == 1); 
  assert(sum == num_threads);
  
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// Procedure: benchmark_linear_insertions
void benchmark_linear_insertions() {

  std::cout << "==== Linear Insertions ====\n";

  std::cout << "Speculative threadpool elapsed time: " 
            << linear_insertions<tf::SpeculativeThreadpool>() << " ms\n";

  std::cout << "Proactive threadpool elapsed time: " 
            << linear_insertions<tf::ProactiveThreadpool>() << " ms\n";

  std::cout << "Simple threadpool elapsed time: " 
            << linear_insertions<tf::SimpleThreadpool>() << " ms\n";
}

// ----------------------------------------------------------------------------

// Function: empty_jobs
template <typename T>
auto empty_jobs() {
  
  const int num_threads = 4;
  const int num_tasks   = 1000000;
  
  auto beg = std::chrono::high_resolution_clock::now();

  T threadpool(num_threads);

  for(size_t i=0; i<num_tasks; i++){
    threadpool.silent_async([](){}); 
  }

  threadpool.shutdown();
  
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// Procedure: benchmark_empty_jobs
void benchmark_empty_jobs() {

  std::cout << "==== Empty Jobs ====\n";
  
  std::cout << "Speculative threadpool elapsed time: " 
            << empty_jobs<tf::SpeculativeThreadpool>() << " ms\n";

  std::cout << "Proactive threadpool elapsed time: " 
            << empty_jobs<tf::ProactiveThreadpool>() << " ms\n";

  std::cout << "Simple threadpool elapsed time: " 
            << empty_jobs<tf::SimpleThreadpool>() << " ms\n";
}

// ----------------------------------------------------------------------------

// Function: atomic_add
template <typename T>
auto atomic_add() {
  
  const int num_threads = 4;
  const int num_tasks   = 1000000;
  
  std::atomic<int> counter(0);
  auto beg = std::chrono::high_resolution_clock::now();
  
  T threadpool(num_threads);
  for(size_t i=0; i<num_tasks; i++){
    threadpool.silent_async([&counter](){ counter++; }); 
  }
  threadpool.shutdown();

  assert(counter == num_tasks);
  
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// Procedure: benchmark_atomic_add
void benchmark_atomic_add() {

  std::cout << "==== Atomic Add ====\n";
  
  std::cout << "Speculative threadpool elapsed time: " 
            << atomic_add<tf::SpeculativeThreadpool>() << " ms\n";

  std::cout << "Proactive threadpool elapsed time: " 
            << atomic_add<tf::ProactiveThreadpool>() << " ms\n";

  std::cout << "Simple threadpool elapsed time: " 
            << atomic_add<tf::SimpleThreadpool>() << " ms\n";
}

// ----------------------------------------------------------------------------

// Function: main
int main(int argc, char* argv[]) {

  benchmark_linear_insertions();
  benchmark_empty_jobs();
  benchmark_atomic_add();
  
  return 0;
}
