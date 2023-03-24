#include "for_each.hpp"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

void for_each_taskflow(size_t num_threads) {

  tf::Executor executor(num_threads);
  tf::Taskflow taskflow;

  taskflow.for_each(tf::ExecutionPolicy<tf::StaticPartitioner>(),
    vec.begin(), vec.end(), [](double& v){
      v = std::tan(v);
    }
  );

  executor.run(taskflow).get();
}

std::chrono::microseconds measure_time_taskflow(size_t num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  for_each_taskflow(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


