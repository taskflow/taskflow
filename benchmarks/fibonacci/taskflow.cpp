#include <taskflow/taskflow.hpp>
#include "fibonacci.hpp"

tf::Executor& get_executor() {
  static tf::Executor executor;
  return executor;
}

// fibonacci computation 
size_t spawn_async(size_t num_fibonacci, tf::Runtime& rt) {

  if (num_fibonacci < 2) {
    return num_fibonacci; 
  }
  
  size_t res1, res2;

  rt.silent_async([num_fibonacci, &res1](tf::Runtime& rt1){
    res1 = spawn_async(num_fibonacci-1, rt1);
  });

  res2 = spawn_async(num_fibonacci-2, rt);

  // use corun to avoid blocking the worker from waiting the two children tasks to finish
  rt.corun();

  return res1 + res2;
}


size_t fibonacci_taskflow(size_t num_threads, size_t num_fibonacci) {
  size_t res;
  static tf::Executor executor(num_threads);
  get_executor().async([num_fibonacci, &res](tf::Runtime& rt){
    res = spawn_async(num_fibonacci, rt);
  }).get();
  return res;
}

std::chrono::microseconds measure_time_taskflow(size_t num_threads, size_t num_fibonacci) {
  auto beg = std::chrono::high_resolution_clock::now();
  fibonacci_taskflow(num_threads, num_fibonacci);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


