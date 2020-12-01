// This program demonstrates loop-based parallelism using:
//   + STL-styled iterators
//   + plain integral indices

#include <taskflow/taskflow.hpp>
#include <cassert>
#include <numeric>

// Procedure: for_each
void for_each(int N) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  std::vector<int> range(N);
  std::iota(range.begin(), range.end(), 0);

  taskflow.for_each(range.begin(), range.end(), [&] (int i) { 
    printf("for_each on container item: %d\n", i);
  });

  executor.run(taskflow).get();
}

// Procedure: for_each_index
void for_each_index(int N) {
  
  tf::Executor executor;
  tf::Taskflow taskflow;

  // [0, N) with step size 2
  taskflow.for_each_index(0, N, 2, [] (int i) {
    printf("for_each_index on index: %d\n", i);
  });

  executor.run(taskflow).get();
}

// ----------------------------------------------------------------------------

// Function: main
int main() {
  
  for_each(100);
  for_each_index(100);

  return 0;
}






