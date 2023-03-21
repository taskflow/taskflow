#include "for_each.hpp"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>

// for_each_tbb
void for_each_tbb(size_t num_threads) {

  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_threads
  );

  tbb::parallel_for(0, (int)vec.size(), 1, [&](int i){
    vec[i] = std::tan(vec[i]);
  });
}

std::chrono::microseconds measure_time_tbb(size_t num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  for_each_tbb(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
