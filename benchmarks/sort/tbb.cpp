#include "sort.hpp"
#include <tbb/parallel_sort.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>

// sort_tbb
void sort_tbb(unsigned num_threads) {

  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_threads
  );

  tbb::parallel_sort(vec.begin(), vec.end());
  //std::cout << sort() << std::endl;
}

std::chrono::microseconds measure_time_tbb(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  sort_tbb(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
