#include <taskflow/taskflow.hpp>
#include "primes.hpp"
#include <ranges>
#include <taskflow/algorithm/reduce.hpp>                                                                                                                                 


size_t primes_taskflow(size_t num_threads, size_t value) {

  auto iota = std::ranges::views::iota(size_t(1), value);
  auto rev = std::ranges::views::reverse(iota);

  tf::Executor executor(num_threads);

  tf::Taskflow taskflow;

  int sum = 0;

  taskflow.transform_reduce(
    rev.begin(), rev.end(), sum, std::plus<>{}, is_prime
  );

  executor.run(taskflow).wait();
 
  return sum; 
}


std::chrono::microseconds measure_time_taskflow(size_t num_threads, size_t value) {
  auto beg = std::chrono::high_resolution_clock::now();

  auto result = primes_taskflow(num_threads, value);
  
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


