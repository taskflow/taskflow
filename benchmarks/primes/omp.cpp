#include "primes.hpp"
#include <omp.h>
#include <ranges>

size_t primes_omp(size_t num_threads, size_t value) {
  size_t sum = 0;

  #pragma omp parallel for num_threads(num_threads) reduction(+ : sum) schedule(dynamic, primes_chunk)
  for(size_t i=1; i<value; ++i) {
    sum += is_prime(i);
  }

  return sum;
}


std::chrono::microseconds measure_time_omp(size_t num_threads, size_t value) {
  auto beg = std::chrono::high_resolution_clock::now();
  primes_omp(num_threads, value);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


