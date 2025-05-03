#include "primes.hpp"
#include <omp.h>
#include <ranges>

size_t primes_omp(size_t num_threads, size_t value) {
  auto output = 0;

  auto iota = std::ranges::views::iota(size_t(1), value);

  auto sum = 0;
#pragma omp parallel for num_threads(num_threads) reduction(+ : sum) schedule(dynamic, primes_chunk) firstprivate(iota)
    
  for (auto &&elem : std::ranges::views::reverse(iota)) {
    sum += is_prime(elem);
  }

  output = sum;

  return output;  

}


std::chrono::microseconds measure_time_omp(size_t num_threads, size_t value) {
  auto beg = std::chrono::high_resolution_clock::now();

  size_t result = 0;

  result = primes_omp(num_threads, value);

  auto end = std::chrono::high_resolution_clock::now();
  
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


