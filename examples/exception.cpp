// The program demonstrate how to capture an exception thrown
// from a running taskflow
#include <taskflow/taskflow.hpp>  

int main(){

  tf::Executor executor;
  tf::Taskflow taskflow("exception");

  auto [A, B, C, D] = taskflow.emplace(
    []() { std::cout << "TaskA\n"; },
    []() { 
      std::cout << "TaskB\n";
      throw std::runtime_error("Exception on Task B");
    },
    []() { 
      std::cout << "TaskC\n"; 
      throw std::runtime_error("Exception on Task C");
    },
    []() { std::cout << "TaskD will not be printed due to exception\n"; }
  );

  A.precede(B, C);  // A runs before B and C
  D.succeed(B, C);  // D runs after  B and C

  try {
    executor.run(taskflow).get();
  }
  catch(const std::runtime_error& e) {
    // catched either TaskB's or TaskC's exception
    std::cout << e.what() << std::endl;
  }

  return 0;
}


