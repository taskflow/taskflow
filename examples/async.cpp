// The program demonstrates how to create asynchronous task
// from an executor and a subflow.
#include <taskflow/taskflow.hpp>

int main() {

  tf::Executor executor;

  // create asynchronous tasks from the executor
  // (using executor as a thread pool)
  std::future<int> fu = executor.async([](){
    std::cout << "async task 1 returns 1\n";
    return 1;
  });

  executor.silent_async([](){  // silent async task doesn't return any future object
    std::cout << "async task 2 does not return (silent)\n";
  });

  executor.wait_for_all();  // wait for the two async tasks to finish

  // create asynchronous tasks from a subflow
  // all asynchronous tasks are guaranteed to finish when the subflow joins
  tf::Taskflow taskflow;

  std::atomic<int> counter {0};

  taskflow.emplace([&](tf::Subflow& sf){
    for(int i=0; i<100; i++) {
      sf.silent_async([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
    }
    sf.join();

    // when subflow joins, all spawned tasks from the subflow will finish
    if(counter == 100) {
      std::cout << "async tasks spawned from the subflow all finish\n";
    }
    else {
      throw std::runtime_error("this should not happen");
    }
  });

  executor.run(taskflow).wait();

  return 0;
}








