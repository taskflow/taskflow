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
  
  std::cout << "[GraphViz dump without name assignment]\n";
  taskflow.dump(std::cout, tf::DumpFormat::Graphviz);
  std::cout << "[Mermaid dump without name assignment]\n";
  taskflow.dump(std::cout, tf::DumpFormat::Mermaid);
  
  A.name("A");
  B.name("B");
  C.name("C");
  D.name("D");
  E.name("E");

  std::cout << "[GraphViz dump with name assignment]\n";
  // if the graph contains solely static tasks, you can simply dump them
  // without running the graph
  taskflow.dump(std::cout, tf::DumpFormat::Graphviz);
  std::cout << "[Mermaid dump with name assignment]\n";
  taskflow.dump(std::cout, tf::DumpFormat::Mermaid);

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
  std::cout << "[Run flow]\n";
  executor.run(taskflow).wait();

  std::cout << "[GraphViz dump with name assignment and subflow]\n";
  taskflow.dump(std::cout, tf::DumpFormat::Graphviz);
  std::cout << "[Mermaid dump with name assignment and subflow]\n";
  taskflow.dump(std::cout, tf::DumpFormat::Mermaid);

  return 0;
}


