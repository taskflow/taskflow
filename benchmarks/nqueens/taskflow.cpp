#include <algorithm>
#include <vector>
#include <exception>
#include <iostream>
#include <numeric>
#include <taskflow/taskflow.hpp>
#include "nqueens.hpp"

auto spawn_async(int j, std::vector<char>&a, tf::Runtime& rt) -> int {

  int N = a.size();

  if (N == j) {
    return 1;
  }

  std::vector<std::vector<char>> buf;
  buf.resize(N, std::vector<char>(N));
  
  std::vector<int> parts(N);

  for (int i = 0; i < N; i++) {

    for (int k = 0; k < j; k++) {
      buf[i][k] = a[k];
    }

    buf[i][j] = i;

    if (queens_ok(j + 1, buf[i].data())) {
      rt.silent_async([&parts, &buf, i, j](tf::Runtime& rt1) {
        parts[i] = spawn_async(j + 1, buf[i], rt1);
      });
    } else {
      parts[i] = 0;
    }
  }

  rt.corun();

  return std::accumulate(parts.begin(), parts.end(), 0L);
}



int nqueens_taskflow(int i, size_t num_threads, std::vector<char>& buf) {

  int output;
  static tf::Executor executor(num_threads);

  executor.async([i, &buf, &output](tf::Runtime& rt){
    output = spawn_async(i, buf, rt);
  }).get();

  return output;
}


std::chrono::microseconds measure_time_taskflow(size_t num_threads, size_t num_nqueens) {
  std::vector<char> buf(num_nqueens);

  auto beg = std::chrono::high_resolution_clock::now();
  auto result = nqueens_taskflow(0, num_threads, buf);
  auto end = std::chrono::high_resolution_clock::now();

  assert(result == answers[num_queens]);

  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
