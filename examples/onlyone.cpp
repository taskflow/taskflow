// A simple example with a semaphore constraint that only one task can
// execute at a time.

#include <taskflow/taskflow.hpp>

#include <iostream>
#include <chrono>
#include <thread>

void sl() {
  std::this_thread::sleep_for(std::chrono::seconds(1));
}

int main() {

  tf::Executor executor(4);
  tf::Taskflow taskflow;
  
  // define a critical region of 1 worker
  tf::Semaphore semaphore(1); 

  // create give tasks in taskflow1
  std::vector<tf::Task> tasks {
    taskflow.emplace([](){ sl(); std::cout << "A1" << std::endl; }),
    taskflow.emplace([](){ sl(); std::cout << "B1" << std::endl; }),
    taskflow.emplace([](){ sl(); std::cout << "C1" << std::endl; }),
    taskflow.emplace([](){ sl(); std::cout << "D1" << std::endl; }),
    taskflow.emplace([](){ sl(); std::cout << "E1" << std::endl; })
  };

  for(auto & task : tasks) {
    task.acquire(semaphore);
    task.release(semaphore);
  }
  
  executor.run(taskflow);
  executor.wait_for_all();

  return 0;
}

