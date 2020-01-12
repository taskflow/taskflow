#include "sparse.hpp"
#include <tbb/tbb.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/flow_graph.h>

void imbalance_tbb(unsigned num_threads) {
  using namespace tbb;
  using namespace tbb::flow;
  
  tbb::task_scheduler_init init(num_threads);

  parallel_for(0, M, 1, [&](unsigned i) {
    compute_one_iteration(i);
  });
}

std::chrono::microseconds measure_time_tbb(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  imbalance_tbb(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


