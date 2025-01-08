#include "reduce_sum.hpp"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/reduce.hpp>

void reduce_sum_taskflow(unsigned num_threads) {

  tf::Executor executor(num_threads);
  tf::Taskflow taskflow;

  double result = 0.0;

  taskflow.reduce_by_index(
    tf::IndexRange<size_t>(0, vec.size(), 1),
    result,
    [&](tf::IndexRange<size_t> range, std::optional<double> running_total) {
      double partial_sum = running_total ? *running_total : 0.0;
      for(size_t i=range.begin(); i<range.end(); i+=range.step_size()) {
        partial_sum += vec[i];
      }
      return partial_sum;
    },
    std::plus<double>()
  );

  executor.run(taskflow).get();

}

std::chrono::microseconds measure_time_taskflow(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  reduce_sum_taskflow(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


