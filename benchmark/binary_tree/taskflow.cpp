#include "binary_tree.hpp"
#include <taskflow/taskflow.hpp> 

// binary_tree_taskflow
void binary_tree_taskflow(size_t num_layers, unsigned num_threads) {

  std::atomic<size_t> counter {0};

  std::vector<tf::Task> tasks(1 << num_layers);
  
  tf::Executor executor(num_threads);
  tf::Taskflow taskflow;

  for(unsigned i=1; i<tasks.size(); i++) {
    tasks[i] = taskflow.emplace([&](){
      counter.fetch_add(1, std::memory_order_relaxed);
    });
  }

  for(unsigned i=1; i<tasks.size(); i++) {
    unsigned l = i << 1;
    unsigned r = l + 1;
    if(l < tasks.size() && r < tasks.size()) {
      tasks[i].precede(tasks[l], tasks[r]);
    }
  }
  
  executor.run(taskflow).get();
  assert(counter + 1 == tasks.size());
}

std::chrono::microseconds measure_time_taskflow(
  size_t num_layers,
  unsigned num_threads
) {
  auto beg = std::chrono::high_resolution_clock::now();
  binary_tree_taskflow(num_layers, num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


