// This example demonstrates how to use 'dump' method to visualize
// a taskflow graph in DOT format.

#include <taskflow/taskflow.hpp>

int main(){

  tf::Taskflow taskflow("Visualization Demo");

  // ------------------------------------------------------
  // Static Tasking
  // ------------------------------------------------------
  auto A = taskflow.emplace([] () { std::cout << "Task A\n"; });
  auto B = taskflow.emplace([] () { std::cout << "Task B\n"; });
  auto C = taskflow.emplace([] () { std::cout << "Task C\n"; });
  auto D = taskflow.emplace([] () { std::cout << "Task D\n"; });
  auto E = taskflow.emplace([] () { std::cout << "Task E\n"; });

  A.precede(B, C, E);
  C.precede(D);
  B.precede(D, E);

  std::cout << "[dump without name assignment]\n";
  taskflow.dump(std::cout);

  std::cout << "[dump with name assignment]\n";
  A.name("A");
  B.name("B");
  C.name("C");
  D.name("D");
  E.name("E");

  // if the graph contains solely static tasks, you can simpley dump them
  // without running the graph
  taskflow.dump(std::cout);

  // ------------------------------------------------------
  // Dynamic Tasking
  // ------------------------------------------------------
  taskflow.emplace([](tf::Subflow& sf){
    sf.emplace([](){ std::cout << "subflow task1"; }).name("s1");
    sf.emplace([](){ std::cout << "subflow task2"; }).name("s2");
    sf.emplace([](){ std::cout << "subflow task3"; }).name("s3");
  });

  // in order to visualize subflow tasks, you need to run the taskflow
  // to spawn the dynamic tasks first
  tf::Executor executor;
  executor.run(taskflow).wait();

  taskflow.dump(std::cout);


  return 0;
}


