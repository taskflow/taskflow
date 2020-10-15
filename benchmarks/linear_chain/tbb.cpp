#include "linear_chain.hpp"
#include <tbb/global_control.h>
#include <tbb/flow_graph.h>

// linear_chain_tbb
void linear_chain_tbb(size_t length, unsigned num_threads) {

  using namespace tbb;
  using namespace tbb::flow;

  size_t counter = 0;
  
  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_threads
  );

  graph g;
    
  std::vector<continue_node<continue_msg>*> tasks(length);

  for(size_t i=0; i<tasks.size(); i++) {
    tasks[i] = new continue_node<continue_msg>(g,
      [&]( const continue_msg& ) {
        counter++;
      }
    );
    if(i) {
      make_edge(*tasks[i-1], *tasks[i]);
    }
  }
  
  tasks[0]->try_put(continue_msg());
  g.wait_for_all();

  for(auto& task : tasks) {
    delete task;
  }
  
  assert(counter == tasks.size());
}

std::chrono::microseconds measure_time_tbb(
  size_t length,
  unsigned num_threads
) {
  auto beg = std::chrono::high_resolution_clock::now();
  linear_chain_tbb(length, num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
