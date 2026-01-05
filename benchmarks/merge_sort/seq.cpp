#include "merge_sort.hpp"

template <typename T>
void msort(std::vector<T>& data, std::vector<T>& buffer, size_t begin, size_t end) {

  size_t n = end - begin;

  // Sequential cutoff
  if (n <= 1024) {
    std::sort(data.begin() + begin, data.begin() + end);
    return;
  }

  size_t mid = begin + n / 2;

  // Spawn task for right half
  msort(data, buffer, mid, end);

  // Process left half in current thread
  msort(data, buffer, begin, mid);

  // Merge
  std::merge(data.begin() + begin, data.begin() + mid,
             data.begin() + mid,   data.begin() + end,
             buffer.begin() + begin);

  std::copy(buffer.begin() + begin,
            buffer.begin() + end,
            data.begin() + begin);
}

template <typename T>
void msort(std::vector<T>& data) {
  std::vector<T> buffer(data.size());
  msort(data, buffer, 0, data.size());
}

std::chrono::microseconds measure_time_seq() {
  auto beg = std::chrono::high_resolution_clock::now();
  msort(vec);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

