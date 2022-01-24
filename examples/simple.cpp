// A simple example to capture the following task dependencies.
//
//           +---+                 
//     +---->| B |-----+           
//     |     +---+     |           
//   +---+           +-v-+         
//   | A |           | D |         
//   +---+           +-^-+         
//     |     +---+     |           
//     +---->| C |-----+           
//           +---+
//
#include <taskflow/taskflow.hpp>  // the only include you need

int main(){

  tf::Executor executor;
  tf::Taskflow taskflow("simple");

  auto [A, B, C, D] = taskflow.emplace(
    [](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf) { std::cout << "TaskA\n"; },
    [](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf) { std::cout << "TaskB\n"; },
    [](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf) { std::cout << "TaskC\n"; },
    [](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf) { std::cout << "TaskD\n"; }
  );

  A.precede(B, C);  // A runs before B and C
  D.succeed(B, C);  // D runs after  B and C

  executor.run(taskflow).wait();

  return 0;
}
