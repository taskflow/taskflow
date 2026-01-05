#include "merge_sort.hpp"
#include <omp.h>

template <typename T>
void merge_sort_omp(std::vector<T>& data, std::vector<T>& buffer, size_t begin, size_t end) {

  size_t n = end - begin;

  // Sequential cutoff
  if (n <= 1024) {
    std::sort(data.begin() + begin, data.begin() + end);
    return;
  }

  size_t mid = begin + n / 2;

  // Spawn task for right half
  #pragma omp task shared(data, buffer)
  {
    merge_sort_omp(data, buffer, mid, end);
  }

  // Process left half in current thread
  merge_sort_omp(data, buffer, begin, mid);

  // Wait for spawned task
  #pragma omp taskwait

  // Merge
  std::merge(data.begin() + begin, data.begin() + mid,
             data.begin() + mid,   data.begin() + end,
             buffer.begin() + begin);

  std::copy(buffer.begin() + begin,
            buffer.begin() + end,
            data.begin() + begin);
}

template <typename T>
void merge_sort_omp(size_t num_threads, std::vector<T>& data) {
  std::vector<T> buffer(data.size());
  #pragma omp parallel num_threads(num_threads)
  {
    #pragma omp single
    {
      merge_sort_omp(data, buffer, 0, data.size());
    }
  }
}

// merge_sort_omp
void merge_sort_omp(size_t nthreads) {
  merge_sort_omp(nthreads, vec);
}

std::chrono::microseconds measure_time_omp(size_t num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  merge_sort_omp(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

