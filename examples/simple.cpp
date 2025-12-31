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

  auto A = taskflow.emplace([](){ std::cout << "TaskA\n"; }).name("A");
  auto B = taskflow.emplace([](){ std::cout << "TaskB\n"; }).name("B");
  auto C = taskflow.emplace([](){ std::cout << "TaskC\n"; }).name("C");
  auto D = taskflow.emplace([](){ std::cout << "TaskD\n"; }).name("D");

  A.precede(B, C);  // A runs before B and C
  D.succeed(B, C);  // D runs after  B and C

  executor.run(taskflow).wait();
  
  taskflow.dump(std::cout);  // dump the graph to a DOT format via standard output

  return 0;
}

