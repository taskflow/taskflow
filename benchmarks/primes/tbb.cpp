#include "primes.hpp"
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

size_t primes_tbb(size_t num_threads, size_t value) {

  tbb::task_arena arena(num_threads);

  int output = 0;

  output = arena.execute([&] {
    return tbb::parallel_reduce(
      tbb::blocked_range(size_t(1), value, primes_chunk),
      0,
      [&](auto range, auto sum) {
        for (auto i = range.begin(); i < range.end(); ++i) {
          sum += is_prime(i);
        }
        return sum;
      },
      std::plus<>());
  });

  return output;
}



std::chrono::microseconds measure_time_tbb(size_t num_threads, size_t value) {
  auto beg = std::chrono::high_resolution_clock::now();
  primes_tbb(num_threads, value);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


