#include <taskflow/taskflow.hpp>
#include <cassert>
#include <numeric>

// Procedure: parallel_for_on_range
void parallel_for_on_range(int N) {

  std::vector<int> range(N);
  std::iota(range.begin(), range.end(), 0);

  tf::Taskflow tf;
  tf.parallel_for(range.begin(), range.end(), [&] (const int i) { 
    printf("parallel_for on container item: %d\n", i);
  });
  tf.wait_for_all();
}

// Procedure: parallel_for_on_index
void parallel_for_on_index(int N) {
  tf::Taskflow tf;

  // [0, N) with step size 1
  tf.parallel_for(0, N, 1, [] (int i) {
    printf("parallel_for on index: %d\n", i);
  });
  tf.wait_for_all();
}

// ----------------------------------------------------------------------------

// Function: main
int main(int argc, char* argv[]) {

  parallel_for_on_range(10);
  parallel_for_on_index(10);

  return 0;
}
