#include "matrix_multiplication.hpp"
#include <tbb/tbb.h>
#include <tbb/task_scheduler_init.h>

// matrix_multiplication_tbb
void matrix_multiplication_tbb(unsigned num_threads) {

  using namespace tbb;
  using namespace tbb::flow;
  
  tbb::task_scheduler_init init(num_threads);

  parallel_for(0, N, 1, [=](int i) {
    for(int j=0; j<N; ++j) {
      a[i][j] = i + j;
    }
  });
  
  parallel_for(0, N, 1, [=](int i) {
    for(int j=0; j<N; ++j) {
      b[i][j] = i * j;
    }
  });
  
  parallel_for(0, N, 1, [=](int i) {
    for(int j=0; j<N; ++j) {
      c[i][j] = 0;
    }
  });
  
  parallel_for(0, N, 1, [=](int i) {
    for(int j=0; j<N; ++j) {
      for(int k=0; k<N; ++k) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  });

  //std::cout << reduce_sum() << std::endl;
}

std::chrono::microseconds measure_time_tbb(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  matrix_multiplication_tbb(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
