#include "reduce_sum.hpp"
#include <taskflow/taskflow.hpp> 

void reduce_sum_taskflow(unsigned num_threads) {

  tf::Executor executor(num_threads); 
  tf::Taskflow taskflow;

  double result;

  taskflow.reduce(vec.begin(), vec.end(), result, [](double l, double r){
    return l + r;
  });

  executor.run(taskflow).get(); 
}

std::chrono::microseconds measure_time_taskflow(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  reduce_sum_taskflow(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


