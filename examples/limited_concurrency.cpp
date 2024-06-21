// A simple example with a semaphore constraint that only one task can
// execute at a time.

#include <taskflow/taskflow.hpp>

void sl() {
  std::this_thread::sleep_for(std::chrono::seconds(1));
}

int main() {

  // define a critical region of 1 worker
  tf::Semaphore semaphore(1);

  tf::Taskflow taskflow;
  tf::Executor executor;

  executor.async([&](tf::Runtime& rt){
    rt.acquire(semaphore);
  });
  
  for(size_t i=0; i<100; i++) {
    taskflow.emplace([&, i](tf::Runtime& rt){
      rt.acquire(semaphore);
      std::cout << i << "-th " << "message " << "never " << "interleaves with others\n";
      rt.release(semaphore);
    });
  }

  executor.run(taskflow).wait();


  return 0;
}

