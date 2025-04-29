#include "linear_chain.hpp"
#include <taskflow/taskflow.hpp>

// binary_tree_taskflow
void linear_chain_taskflow(size_t length, unsigned num_threads) {

  size_t counter {0};

  std::vector<tf::Task> tasks(length);

  static tf::Executor executor(num_threads);
  tf::Taskflow taskflow;

  for(size_t i=0; i<length; ++i) {
    tasks[i] = taskflow.emplace([&] () { counter++; });
  }

  taskflow.linearize(tasks);

  executor.run(taskflow).get();
  assert(counter == tasks.size());
}

std::chrono::microseconds measure_time_taskflow(
  size_t length,
  unsigned num_threads
) {
  auto beg = std::chrono::high_resolution_clock::now();
  linear_chain_taskflow(length, num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


