// This program demonstrates how to use cpp-taskflow to
// write loop-based parallelism.

#include <taskflow/taskflow.hpp>
#include <cassert>
#include <numeric>

// Procedure: parallel_for_on_range
void parallel_for_on_range(int N) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  std::vector<int> range(N);
  std::iota(range.begin(), range.end(), 0);

  taskflow.parallel_for(range.begin(), range.end(), [&] (const int i) { 
    printf("parallel_for on container item: %d\n", i);
  });

  executor.run(taskflow).get();
}

// Procedure: parallel_for_on_index
void parallel_for_on_index(int N) {
  
  tf::Executor executor;
  tf::Taskflow taskflow;

  // [0, N) with step size 1
  taskflow.parallel_for(0, N, 1, [] (int i) {
    printf("parallel_for on index: %d\n", i);
  });

  executor.run(taskflow).get();
}

// ----------------------------------------------------------------------------

// Function: main
int main() {

  parallel_for_on_range(10);
  parallel_for_on_index(10);

  return 0;
}
