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
    []() { std::cout << "TaskA\n"; },
    []() { std::cout << "TaskB\n"; },
    []() { std::cout << "TaskC\n"; },
    []() { std::cout << "TaskD\n"; }
  );

  A.name("A");
  B.name("B");
  C.name("C");
  D.name("D");

  A.precede(B, C);  // A runs before B and C
  D.succeed(B, C);  // D runs after  B and C

  executor.run(taskflow).wait();
  
  // dump the taskflow graph into a .dot format
  taskflow.dump(std::cout);

  return 0;
}

