// This example demonstrates how to use the run_and_wait
// method in the executor.
#include <taskflow/taskflow.hpp>

int main(){

  // create an executor and a taskflow
  tf::Executor executor(2);
  tf::Taskflow taskflow("Demo");

  int counter{0};
  
  // taskflow to run by the main taskflow
  tf::Taskflow others;
  tf::Task A = others.emplace([&](){ counter++; });
  tf::Task B = others.emplace([&](){ counter++; });
  A.precede(B);

  // main taskflow
  tf::Task C = taskflow.emplace([&](){
    executor.run_and_wait(others);
  });
  tf::Task D = taskflow.emplace([&](){
    executor.run_and_wait(others);
  });
  C.precede(D);

  executor.run(taskflow).wait();

  // run others again
  executor.run(others).wait();

  std::cout << "counter is: " << counter << std::endl;

  return 0;
}
