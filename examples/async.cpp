// The program demonstrates how to create asynchronous task
// from an executor.
#include <taskflow/taskflow.hpp>

int main() {
  
  tf::Executor executor;

  std::future<int> future1 = executor.async([](){
    std::cout << "async task 1 returns 1\n";
    return 1;
  });

  executor.silent_async([](){
    std::cout << "async task 2 does not return (silent)\n";
  });

  executor.wait_for_all();  // wait for the two async tasks to finish

  return 0;
}

