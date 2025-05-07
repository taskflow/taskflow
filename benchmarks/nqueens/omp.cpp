#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>
#include "nqueens.hpp"


auto omp_nqueens(int j, std::vector<char>& a) -> int {

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
#pragma omp task untied shared(parts, buf) firstprivate(i, j) default(none)
      parts[i] = omp_nqueens(j + 1, buf[i]);
    } else {
      parts[i] = 0;
    }
  }

#pragma omp taskwait

  return std::accumulate(parts.begin(), parts.end(), 0L);
}


std::chrono::microseconds measure_time_omp(size_t num_threads, size_t num_nqueens) {

  std::vector<char> buf(num_nqueens);

  auto beg = std::chrono::high_resolution_clock::now();
 
  int result;

  #pragma omp parallel num_threads(num_threads)
  {
    #pragma omp single
    {
      result = omp_nqueens(0, buf);
    }
  }
  
  auto end = std::chrono::high_resolution_clock::now();

  assert(result == answers[num_queens]);

  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
