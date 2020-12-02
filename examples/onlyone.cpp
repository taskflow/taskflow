// A simple example with a semaphore constraint: only one task can
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
  tf::Taskflow taskflow1;
  tf::Taskflow taskflow2;
  
  // define a critical region of 2 workers
  tf::Semaphore semaphore(2); 

  // create give tasks in taskflow1
  std::vector<tf::Task> tasks1 {
    taskflow1.emplace([](){ sl(); std::cout << "A1" << std::endl; }),
    taskflow1.emplace([](){ sl(); std::cout << "B1" << std::endl; }),
    taskflow1.emplace([](){ sl(); std::cout << "C1" << std::endl; }),
    taskflow1.emplace([](){ sl(); std::cout << "D1" << std::endl; }),
    taskflow1.emplace([](){ sl(); std::cout << "E1" << std::endl; })
  };

  for(auto & task : tasks1) {
    task.acquire(semaphore);
    task.release(semaphore);
  }
  
  // create five tasks in taskflow2
  std::vector<tf::Task> tasks2 {
    taskflow2.emplace([](){ sl(); std::cout << "A2" << std::endl; }),
    taskflow2.emplace([](){ sl(); std::cout << "B2" << std::endl; }),
    taskflow2.emplace([](){ sl(); std::cout << "C2" << std::endl; }),
    taskflow2.emplace([](){ sl(); std::cout << "D2" << std::endl; }),
    taskflow2.emplace([](){ sl(); std::cout << "E2" << std::endl; })
  };
  
  for(auto & task : tasks2) {
    task.acquire(semaphore);
    task.release(semaphore);
  }
  
  executor.run(taskflow2);
  executor.run(taskflow1);
  executor.wait_for_all();

  std::cout << semaphore.count() << '\n';
}

