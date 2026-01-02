#include <algorithm>
#include <vector>
#include <exception>
#include <iostream>
#include <numeric>
#include <taskflow/taskflow.hpp>
#include "nqueens.hpp"

int nqueens(int j, std::vector<char>&a, tf::Executor& executor) {

  int N = a.size();

  if (N == j) {
    return 1;
  }

  auto tg = executor.task_group();

  std::vector<std::vector<char>> buf;
  buf.resize(N, std::vector<char>(N));
  
  std::vector<int> parts(N);

  for (int i = 0; i < N; i++) {

    for (int k = 0; k < j; k++) {
      buf[i][k] = a[k];
    }

    buf[i][j] = i;

    if (queens_ok(j + 1, buf[i].data())) {
      tg.silent_async([&parts, &buf, i, j, &executor]() {
        parts[i] = nqueens(j + 1, buf[i], executor);
      });
    } else {
      parts[i] = 0;
    }
  }

  tg.corun();

  return std::accumulate(parts.begin(), parts.end(), 0L);
}

int nqueens(int i, size_t num_threads, std::vector<char>& buf) {
  static tf::Executor executor(num_threads);
  return executor.async([i, &buf](){ return nqueens(i, buf, executor); }).get();
}

std::chrono::microseconds measure_time_taskflow(size_t num_threads, size_t num_nqueens) {
  std::vector<char> buf(num_nqueens);

  auto beg = std::chrono::high_resolution_clock::now();
  auto result = nqueens(0, num_threads, buf);
  auto end = std::chrono::high_resolution_clock::now();

  if(result != answers[num_nqueens]) {
    throw std::runtime_error("incorrect result");
  }

  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
