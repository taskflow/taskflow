#include "async_task.hpp"
#include <future>

// async_task cstdutation
void async_task_std(unsigned, size_t num_tasks) {
  std::vector<std::future<void>> futures;
  for(size_t i=0; i<num_tasks; i++){
    futures.emplace_back(std::async(func));
  }
  for(auto& fu : futures) {
    fu.get();
  }
}

std::chrono::microseconds measure_time_std(unsigned num_threads, size_t num_tasks) {
  auto beg = std::chrono::high_resolution_clock::now();
  async_task_std(num_threads, num_tasks);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
