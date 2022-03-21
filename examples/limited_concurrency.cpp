// A simple example with a semaphore constraint that only one task can
// execute at a time.

#include <taskflow/taskflow.hpp>

void sl() {
  std::this_thread::sleep_for(std::chrono::seconds(1));
}

int main() {

  tf::Executor executor(4);
  tf::Taskflow taskflow;

  // define a critical region of 1 worker
  tf::Semaphore semaphore(1);

  // create give tasks in taskflow
  std::vector<tf::Task> tasks {
    taskflow.emplace([](){ sl(); std::cout << "A" << std::endl; }),
    taskflow.emplace([](){ sl(); std::cout << "B" << std::endl; }),
    taskflow.emplace([](){ sl(); std::cout << "C" << std::endl; }),
    taskflow.emplace([](){ sl(); std::cout << "D" << std::endl; }),
    taskflow.emplace([](){ sl(); std::cout << "E" << std::endl; })
  };

  for(auto & task : tasks) {
    task.acquire(semaphore);
    task.release(semaphore);
  }

  executor.run(taskflow);
  executor.wait_for_all();

  return 0;
}

