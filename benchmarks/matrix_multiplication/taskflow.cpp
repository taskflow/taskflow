#include "matrix_multiplication.hpp"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

// matrix_multiplication_taskflow
void matrix_multiplication_taskflow(unsigned num_threads) {

  static tf::Executor executor(num_threads);

  tf::Taskflow taskflow;

  auto init_a = taskflow.for_each_index(0, N, 1, [&] (int i) {
    for(int j=0; j<N; ++j) {
      a[i][j] = i + j;
    }
  });

  auto init_b = taskflow.for_each_index(0, N, 1, [&] (int i) {
    for(int j=0; j<N; ++j) {
      b[i][j] = i * j;
    }
  });

  auto init_c = taskflow.for_each_index(0, N, 1, [&] (int i) {
    for(int j=0; j<N; ++j) {
      c[i][j] = 0;
    }
  });

  auto comp_c = taskflow.for_each_index(0, N, 1, [&] (int i) {
    for(int j=0; j<N; ++j) {
      for(int k=0; k<N; k++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  });


  comp_c.succeed(init_a, init_b, init_c);

  executor.run(taskflow).get();

  //std::cout << reduce_sum() << std::endl;
  //taskflow.dump(std::cout);
}

std::chrono::microseconds measure_time_taskflow(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  matrix_multiplication_taskflow(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


