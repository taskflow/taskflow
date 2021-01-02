// The program demonstrates how to cancel submitted asynchronous tasks.

#include <taskflow/taskflow.hpp>

int main() {

  tf::Executor executor;

  std::vector<tf::Future<void>> futures;
  
  // submit 10000 asynchronous tasks
  for(int i=0; i<10000; i++) {
    futures.push_back(executor.async([i](){
      printf("task %d\n", i);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }));
  }
  
  // cancel all asynchronous tasks
  for(auto& fu : futures) {
    fu.cancel();
  }

  executor.wait_for_all();

  return 0;
}

