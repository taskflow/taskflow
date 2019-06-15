#include "matrix_multiplication.hpp"
#include <taskflow/taskflow.hpp> 

// matrix_multiplication_taskflow
void matrix_multiplication_taskflow(unsigned num_threads) {

  tf::Executor executor(num_threads); 
  tf::Taskflow taskflow;

  auto pa = taskflow.parallel_for(0, N, 1, [&] (int i) { 
    for(int j=0; j<N; ++j) {
      a[i][j] = i + j;
    }
  });
  
  auto pb = taskflow.parallel_for(0, N, 1, [&] (int i) { 
    for(int j=0; j<N; ++j) {
      b[i][j] = i * j;
    }
  });
  
  auto pc = taskflow.parallel_for(0, N, 1, [&] (int i) { 
    for(int j=0; j<N; ++j) {
      c[i][j] = 0;;
    }
  });

  auto pr = taskflow.parallel_for(0, N, 1, [&] (int i) {
    for(int j=0; j<N; ++j) {
      for(int k=0; k<N; k++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  });

  pa.second.precede(pr.first);
  pb.second.precede(pr.first);
  pc.second.precede(pr.first);

  executor.run(taskflow).get(); 
  
  //std::cout << reduce_sum() << std::endl;
}

std::chrono::microseconds measure_time_taskflow(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  matrix_multiplication_taskflow(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


