#include "fibonacci.hpp"

// fibonacci computation 
size_t fibonacci_std(size_t num_fibonacci) {

  if (num_fibonacci < 2) {
    return num_fibonacci;
  }

  return fibonacci_std(num_fibonacci-1) + fibonacci_std(num_fibonacci-2);
}

/*
// fibonacci computation in iterative 
size_t fibonacci_std_iterative(size_t num_fibonacci) {
  if (num_fibonacci < 2)
      return num_fibonacci;

  size_t prev = 0, curr = 1;

  for (size_t i = 2; i <= num_fibonacci; ++i) {
    size_t next = prev + curr;
    prev = curr;
    curr = next;
  }

  return curr;
}

*/


std::chrono::microseconds measure_time_std(unsigned num_threads, size_t num_fibonacci) {
  auto beg = std::chrono::high_resolution_clock::now();
  size_t result = fibonacci_std(num_fibonacci);
  auto end = std::chrono::high_resolution_clock::now();
  
  assert(result == fibonacci_sequence[num_fibonacci]);
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

