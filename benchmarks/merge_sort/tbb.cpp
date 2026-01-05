#include "merge_sort.hpp"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_group.h>
#include <tbb/global_control.h>

template <typename T>
void merge_sort_tbb(std::vector<T>& data, std::vector<T>& buffer, size_t begin, size_t end) {
  size_t n = end - begin;
  if (n <= 1024) { std::sort(data.begin() + begin, data.begin() + end); return; }
  size_t mid = begin + n / 2;
  tbb::task_group tg;
  tg.run([=, &data, &buffer] (){ merge_sort_tbb(data, buffer, mid, end); });
  merge_sort_tbb(data, buffer, begin, mid);
  tg.wait();
  std::merge(data.begin()+begin, data.begin()+mid, data.begin()+mid, data.begin()+end,
             buffer.begin() + begin);
  std::copy(buffer.begin() + begin, buffer.begin() + end, data.begin() + begin);
}

template <typename T>
void merge_sort_tbb(std::vector<T>& data) {
  std::vector<T> buffer(data.size());
  merge_sort_tbb(data, buffer, 0, data.size());
}

// merge_sort_tbb
void merge_sort_tbb(size_t num_threads) {
  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_threads
  );
  merge_sort_tbb(vec);
}

std::chrono::microseconds measure_time_tbb(size_t num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  merge_sort_tbb(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
