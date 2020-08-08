#include <algorithm> 
#include <cassert>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>
#include <random>
#include <cmath>
#include <atomic>

extern int N;
extern double **a, **b, **c;

std::chrono::microseconds measure_time_taskflow(unsigned);
std::chrono::microseconds measure_time_tbb(unsigned);
std::chrono::microseconds measure_time_omp(unsigned);

inline void allocate_matrix() {
  a = static_cast<double**>(std::malloc(N * sizeof(double*)));
  b = static_cast<double**>(std::malloc(N * sizeof(double*)));
  c = static_cast<double**>(std::malloc(N * sizeof(double*)));
  for(int i=0; i<N; ++i) {
    a[i] = static_cast<double*>(std::malloc(N * sizeof(double)));
    b[i] = static_cast<double*>(std::malloc(N * sizeof(double)));
    c[i] = static_cast<double*>(std::malloc(N * sizeof(double)));
  }
}

inline void deallocate_matrix() {
  for(int i=0; i<N; ++i) {
    std::free(a[i]);
    std::free(b[i]);
    std::free(c[i]);
  }
  std::free(a);
  std::free(b);
  std::free(c);
}

inline int64_t reduce_sum() {
  int64_t sum {0};
  for(int i=0; i<N; i++) {
    for(int j=0; j<N; ++j) {
      sum += c[i][j];
    }
  }
  return sum;
}

