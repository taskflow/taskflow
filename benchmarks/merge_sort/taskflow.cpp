#include "merge_sort.hpp"
#include <taskflow/taskflow.hpp>

template <typename T>
void msort(tf::Executor& executor, std::vector<T>& data, std::vector<T>& buffer, size_t begin, size_t end) {
  size_t n = end - begin;
  if (n <= 1024) { std::sort(data.begin() + begin, data.begin() + end); return; }
  size_t mid = begin + n / 2;
  tf::TaskGroup tg = executor.task_group();
  tg.silent_async([=, &data, &buffer, &executor](){ msort(executor, data, buffer, mid, end); });
  msort(executor, data, buffer, begin, mid);
  tg.corun();
  std::merge(data.begin()+begin, data.begin()+mid, data.begin()+mid, data.begin()+end,
             buffer.begin() + begin);
  std::copy(buffer.begin() + begin, buffer.begin() + end, data.begin() + begin);
}

template <typename T>
void msort(tf::Executor& executor, std::vector<T>& data) {
  std::vector<T> buffer(data.size());
  executor.async([&]() { msort(executor, data, buffer, 0, data.size()); }).wait();
}

void merge_sort_taskflow(tf::Executor& executor) {
  executor.async([&](){ msort(executor, vec); }).get();
}

std::chrono::microseconds measure_time_taskflow(size_t num_threads) {
  static tf::Executor executor(num_threads);
  auto beg = std::chrono::high_resolution_clock::now();
  merge_sort_taskflow(executor);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


