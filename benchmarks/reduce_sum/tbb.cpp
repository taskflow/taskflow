#include "reduce_sum.hpp"
#include <tbb/tbb.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

// reduce_sum_tbb
void reduce_sum_tbb(unsigned num_threads) {

  using namespace tbb;
  using namespace tbb::flow;
 
  tbb::task_scheduler_init init(num_threads);

  tbb::parallel_reduce(
    tbb::blocked_range<double*>(vec.data(), vec.data() + vec.size()),
    0.0,
    [](const blocked_range<double*>& r, double value) {
      return std::accumulate(r.begin(), r.end(), value);
    },
    [](double l, double r) -> double {
      return l + r;
    }
  );

  //std::cout << reduce_sum() << std::endl;
}

std::chrono::microseconds measure_time_tbb(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  reduce_sum_tbb(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
