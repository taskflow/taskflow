#include "embarrassing_parallelism.hpp"

#include <tbb/global_control.h>
#include <tbb/flow_graph.h>

// the embarrassing_parallelism computation
void embarrassing_parallelism_tbb(unsigned num_threads, size_t num_tasks) {

  using namespace tbb;
  using namespace tbb::flow;

  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_threads
  );

  tbb::flow::graph g;

  std::vector<
    tbb::flow::continue_node<tbb::flow::continue_msg>
  > nodes;

  for(size_t i = 0; i < num_tasks; ++i) {
    nodes.emplace_back(g, [i](const tbb::flow::continue_msg&) { dummy(i); });
  }

  // Trigger each node once
  for(auto& n : nodes) {
    n.try_put(tbb::flow::continue_msg{});
  }

  g.wait_for_all();
}

std::chrono::microseconds measure_time_tbb(unsigned num_threads, size_t num_tasks) {
  auto beg = std::chrono::high_resolution_clock::now();
  embarrassing_parallelism_tbb(num_threads, num_tasks);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


