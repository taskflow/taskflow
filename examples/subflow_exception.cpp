// This program demonstrates the exception in subflow.

#include <taskflow/taskflow.hpp>    

int main() {

  tf::Executor executor;
  tf::Taskflow taskflow;

  taskflow.emplace([](tf::Subflow& sf) {
    tf::Task A = sf.emplace([]() { 
      std::cout << "Task A\n";
      throw std::runtime_error("exception on A"); 
    });
    tf::Task B = sf.emplace([]() { 
      std::cout << "Task B\n"; 
    });
    A.precede(B);
    sf.join();
  });

  try
  {
    executor.run(taskflow).get();
  }
  catch (const std::runtime_error& re)
  {
    std::cout << "exception thrown from running the taskflow: " << re.what() << '\n';
  }

  return 0;
}
