#include "binary_tree.hpp"
#include <tbb/global_control.h>
#include <tbb/flow_graph.h>

// binary_tree_tbb
void binary_tree_tbb(size_t num_layers, unsigned num_threads) {

  using namespace tbb;
  using namespace tbb::flow;
  
  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_threads
  );

  std::atomic<size_t> counter {0};
   
  graph g;
    
  std::vector<continue_node<continue_msg>*> tasks(1 << num_layers);

  for(unsigned i=1; i<tasks.size(); i++) {
    tasks[i] = new continue_node<continue_msg>(g,
      [&]( const continue_msg& ) {
        counter.fetch_add(1, std::memory_order_relaxed);
      }
    );
  }
  
  for(unsigned i=1; i<tasks.size(); i++) {
    unsigned l = i << 1;
    unsigned r = l + 1;
    if(l < tasks.size() && r < tasks.size()) {
      make_edge(*tasks[i], *tasks[l]);
      make_edge(*tasks[i], *tasks[r]);
    }
  }
  
  tasks[1]->try_put(continue_msg());
  g.wait_for_all();

  for(auto& task : tasks) {
    delete task;
  }
  
  assert(counter + 1 == tasks.size());
}

std::chrono::microseconds measure_time_tbb(
  size_t num_layers,
  unsigned num_threads
) {
  auto beg = std::chrono::high_resolution_clock::now();
  binary_tree_tbb(num_layers, num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
