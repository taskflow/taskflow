#include "linear_chain.hpp"
#include <taskflow/taskflow.hpp>

// binary_tree_taskflow
void linear_chain_taskflow(size_t length, tf::Executor& executor) {

  size_t counter {0};

  std::vector<tf::Task> tasks(length);
  tf::Taskflow taskflow;

  for(size_t i=0; i<length; ++i) {
    tasks[i] = taskflow.emplace([&] () { counter++; });
  }

  taskflow.linearize(tasks);

  executor.run(taskflow).get();
  assert(counter == tasks.size());
}

std::chrono::microseconds measure_time_taskflow(size_t length, unsigned num_threads) {
  static tf::Executor executor(num_threads);
  auto beg = std::chrono::high_resolution_clock::now();
  linear_chain_taskflow(length, executor);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


