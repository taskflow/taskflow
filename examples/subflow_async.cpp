// The program demonstrates how to create asynchronous task
// from a running subflow.
#include <taskflow/taskflow.hpp>

int main() {

  tf::Taskflow taskflow("Subflow Async");
  tf::Executor executor;

  std::atomic<int> counter{0};

  taskflow.emplace([&](tf::Subflow& sf){
    for(int i=0; i<10; i++) {
      // Here, we use "silent_async" instead of "async" because we do
      // not care the return value. The method "silent_async" gives us
      // less overhead compared to "async".
      // The 10 asynchronous tasks run concurrently.
      sf.silent_async([&](){
        std::cout << "async task from the subflow\n";
        counter.fetch_add(1, std::memory_order_relaxed);
      });
    }
    sf.join();
    std::cout << counter << " = 10\n";
  });

  executor.run(taskflow).wait();

  return 0;
}

