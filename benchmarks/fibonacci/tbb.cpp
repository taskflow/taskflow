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

std::chrono::microseconds measure_time_tbb(unsigned num_threads, size_t num_fibonacci) {
  auto beg = std::chrono::high_resolution_clock::now();
  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_threads
  );

  size_t result = 0;

  result = fibonacci_tbb(num_fibonacci);
  
  auto end = std::chrono::high_resolution_clock::now();
  
  assert(result == fibonacci_sequence[num_fibonacci]);
  
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


