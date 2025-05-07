#include "fibonacci.hpp"
#include <tbb/task_group.h>
#include <tbb/global_control.h>

// fibonacci computation 
size_t fibonacci_tbb(size_t num_fibonacci) {

  if (num_fibonacci < 2) {
    return num_fibonacci;
  }

  size_t x = 0, y = 0;
  tbb::task_group tg;

  // Spawn the first sub-task
  tg.run([&] { x = fibonacci_tbb(num_fibonacci - 1); });

  // Execute the second sub-task in the current thread
  y = fibonacci_tbb(num_fibonacci - 2);

  // Wait for the first sub-task to finish
  tg.wait();

  return x + y;
}

std::chrono::microseconds measure_time_tbb(size_t num_threads, size_t num_fibonacci) {
  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_threads
  );
  auto beg = std::chrono::high_resolution_clock::now();
  fibonacci_tbb(num_fibonacci);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


