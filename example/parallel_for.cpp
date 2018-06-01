#include "taskflow.hpp"
#include <cassert>

// Function: fib
int fib(int n) {
  if(n <= 2) return n;
  return (fib(n-1) + fib(n-2))%1024;
}

// ------------------------------------------------------------------------------------------------

// Procedure: sequential
void sequential(int N) {
  auto tbeg = std::chrono::steady_clock::now();
  for(int i=0; i<N; ++i) {
    printf("fib[%d]=%d\n", i, fib(i));
  }
  auto tend = std::chrono::steady_clock::now();
  std::cout << "sequential version takes " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg).count() 
            << " ms\n";
}

// Procedure: taskflow
void taskflow(int N) {
  auto tbeg = std::chrono::steady_clock::now();
  tf::Taskflow tf;
  std::vector<int> range(N);
  for(int n=0; n<N; ++n) {
    range[n] = n;
  }
  tf.parallel_for(range.begin(), range.end(), [&range] (int& i) { 
    printf("fib[%d]=%d\n", i, fib(i));
  });
  tf.wait_for_all();
  auto tend = std::chrono::steady_clock::now();
  std::cout << "taskflow version takes " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg).count() 
            << " ms\n";
}

// Procedure: openmp
void openmp(int N) {
  auto tbeg = std::chrono::steady_clock::now();
  #pragma omp parallel for
  for(int i=0; i<N; ++i) {
    printf("fib[%d]=%d\n", i, fib(i));
  }
  auto tend = std::chrono::steady_clock::now();
  std::cout << "openmp version takes " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg).count() 
            << " ms\n";
}

// ------------------------------------------------------------------------------------------------

// Function: main
int main(int argc, char* argv[]) {

  if(argc != 3) {
    std::cerr << "usage: ./parallel_for [baseline|openmp|taskflow] N\n";
    std::exit(EXIT_FAILURE);
  }
  
  // Run methods
  if(std::string_view method(argv[1]); method == "baseline") {
    sequential(std::atoi(argv[2]));
  }
  else if(method == "openmp") {
    openmp(std::atoi(argv[2]));
  }
  else if(method == "taskflow") {
    taskflow(std::atoi(argv[2]));
  }
  else {
    std::cerr << "wrong method, shoud be [baseline|openmp|taskflow]\n";
  }

  return 0;
}
