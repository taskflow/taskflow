#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <numeric>
#include "nqueens.hpp"
#include <tbb/global_control.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <vector>


auto nqueens_tbb(int j, std::vector<char>& a) -> int {

  int N = a.size();

  if (N == j) {
    return 1;
  }

  std::vector<std::vector<char>> buf;
  buf.resize(N, std::vector<char>(N));
  
  std::vector<int> parts(N);

  tbb::task_group g;

  for (int i = 0; i < N; i++) {

    for (int k = 0; k < j; k++) {
      buf[i][k] = a[k];
    }

    buf[i][j] = i;

    if (queens_ok(j + 1, buf[i].data())) {
      g.run([&parts, &buf, i, j] {
        parts[i] = nqueens_tbb(j + 1, buf[i]);
      });
    } else {
      parts[i] = 0;
    }
  }

  g.wait();

  return std::accumulate(parts.begin(), parts.end(), 0L);
}



std::chrono::microseconds measure_time_tbb(size_t num_threads, size_t num_nqueens) {

  std::vector<char> buf(num_nqueens);

  auto beg = std::chrono::high_resolution_clock::now();
  
  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_threads
  );

  auto result = nqueens_tbb(0, buf);
  
  auto end = std::chrono::high_resolution_clock::now();

  assert(result == answers[num_queens]);

  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
