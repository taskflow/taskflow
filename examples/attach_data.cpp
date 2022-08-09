// This example demonstrates how to attach data to a task and run
// the task iteratively with changing data.

#include <taskflow/taskflow.hpp>

int main(){

  tf::Executor executor;
  tf::Taskflow taskflow("attach data to a task");

  int data;

  // create a task and attach it the data
  auto A = taskflow.placeholder();
  A.data(&data).work([A](){
    auto d = *static_cast<int*>(A.data());
    std::cout << "data is " << d << std::endl;
  });

  // run the taskflow iteratively with changing data
  for(data = 0; data<10; data++){
    executor.run(taskflow).wait();
  }

  return 0;
}
