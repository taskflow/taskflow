#include "matrix_multiplication.hpp"
#include <taskflow/taskflow.hpp> 

// matrix_multiplication_taskflow
void matrix_multiplication_taskflow(unsigned num_threads) {

  tf::Executor executor(num_threads); 
  tf::Taskflow taskflow;

  auto sync = taskflow.emplace([](){});

  for(int i=0; i<N; ++i) {
    taskflow.emplace([&, i=i](){
      for(int j=0; j<N; ++j) {
        a[i][j] = i + j;
      }
    }).precede(sync);
  }

  for(int i=0; i<N; ++i) {
    taskflow.emplace([&, i=i](){
      for(int j=0; j<N; ++j) {
        b[i][j] = i * j;
      }
    }).precede(sync);
  }

  for(int i=0; i<N; ++i) {
    taskflow.emplace([&, i=i](){
      for(int j=0; j<N; ++j) {
        c[i][j] = 0;;
      }
    }).precede(sync);
  }

  for(int i=0; i<N; ++i) {
    auto t = taskflow.emplace([&, i=i](){
      for(int j=0; j<N; ++j) {
        for(int k=0; k<N; k++) {
          c[i][j] += a[i][k] * b[k][j];
        }
      }
    });
    sync.precede(t);
  }

  executor.run(taskflow).get(); 
  
  //std::cout << reduce_sum() << std::endl;
}

std::chrono::microseconds measure_time_taskflow(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  matrix_multiplication_taskflow(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


