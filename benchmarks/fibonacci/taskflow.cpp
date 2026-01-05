#include <taskflow/taskflow.hpp>
#include "fibonacci.hpp"

// fibonacci computation 
size_t fibonacci(size_t n, tf::Executor& executor) {

  if (n < 2) {
    return n; 
  }
  
  size_t res1, res2;

  tf::TaskGroup tg = executor.task_group();

  tg.silent_async([n, &res1, &executor](){ res1 = fibonacci(n-1, executor); });

  res2 = fibonacci(n-2, executor);

  tg.corun();

  return res1 + res2;
}


size_t fibonacci_taskflow(tf::Executor& executor, size_t n) {
  return executor.async([n, &executor](){ return fibonacci(n, executor); }).get();
}

std::chrono::microseconds measure_time_taskflow(size_t num_threads, size_t n) {
  static tf::Executor executor(num_threads);
  auto beg = std::chrono::high_resolution_clock::now();
  fibonacci_taskflow(executor, n);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


