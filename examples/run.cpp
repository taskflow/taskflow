// This example demonstrates how to use different methods to
// run a taskflow.
#include <taskflow/taskflow.hpp>

int main(){

  // create an executor and a taskflow
  tf::Executor executor(1);
  tf::Taskflow taskflow("Demo");

  auto A = taskflow.emplace([&](){ std::cout << "TaskA\n"; }).name("A");
  auto B = taskflow.emplace([&](tf::Subflow& subflow){
    std::cout << "TaskB\n";
    auto B1 = subflow.emplace([&](){ std::cout << "TaskB1\n"; }).name("B1");
    auto B2 = subflow.emplace([&](){ std::cout << "TaskB2\n"; }).name("B2");
    auto B3 = subflow.emplace([&](){ std::cout << "TaskB3\n"; }).name("B3");
    B1.precede(B3);
    B2.precede(B3);
  }).name("B");

  auto C = taskflow.emplace([&](){ std::cout << "TaskC\n"; }).name("C");
  auto D = taskflow.emplace([&](){ std::cout << "TaskD\n"; }).name("D");

  A.precede(B, C);
  B.precede(D);
  C.precede(D);

  // dumpping a taskflow before execution won't visualize subflow tasks
  std::cout << "Dump the taskflow before execution:\n";
  taskflow.dump(std::cout);

  std::cout << "Run the taskflow once without callback\n" << std::endl;
  executor.run(taskflow).get();
  std::cout << std::endl;

  // after execution, we can visualize subflow tasks
  std::cout << "Dump the taskflow after execution:\n";
  taskflow.dump(std::cout);
  std::cout << std::endl;

  std::cout << "Use wait_for_all to wait for the execution to finish\n";
  executor.run(taskflow).get();
  executor.wait_for_all();
  std::cout << std::endl;

  std::cout << "Execute the taskflow two times without a callback\n";
  executor.run(taskflow).get();
  std::cout << "Dump after two executions:\n";
  taskflow.dump(std::cout);
  std::cout << std::endl;

  std::cout << "Execute the taskflow four times with a callback\n";
  executor.run_n(taskflow, 4, [] () { std::cout << "finishes 4 runs\n"; })
          .get();
  std::cout << std::endl;

  std::cout << "Run the taskflow until the predicate returns true\n";
  executor.run_until(taskflow, [counter=3]() mutable {
    std::cout << "Counter = " << counter << std::endl;
    return counter -- == 0;
  }).get();

  taskflow.dump(std::cout);

  return 0;
}
