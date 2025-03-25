#include "for_each.hpp"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

void for_each_taskflow(size_t num_threads) {

  static tf::Executor executor(num_threads);
  tf::Taskflow taskflow;

  tf::IndexRange<size_t> range(0, vec.size(), 1);

  taskflow.for_each_by_index(range, [&](tf::IndexRange<size_t> sr){ 
    for(size_t i=sr.begin(); i<sr.end(); i+=sr.step_size()) {
      vec[i] = std::tan(vec[i]);
    }
  });

  executor.run(taskflow).get();


  //executor.async(tf::make_for_each_by_index_task(range, [&](tf::IndexRange<size_t> sr) {
  //  for(size_t i=sr.begin(); i<sr.end(); i+=sr.step_size()) {
  //    vec[i] = std::tan(vec[i]);
  //  }
  //})).wait();
}

std::chrono::microseconds measure_time_taskflow(size_t num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  for_each_taskflow(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


