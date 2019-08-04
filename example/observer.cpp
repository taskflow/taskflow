// Demonstrates the use of observer to monitor worker activities

#include <taskflow/taskflow.hpp>

int main(){
  
  tf::Executor executor;
  
  // Create a taskflow of eight tasks
  tf::Taskflow taskflow;

  auto [A, B, C, D, E, F, G, H] = taskflow.emplace(
    [] () { std::cout << "1\n"; },
    [] () { std::cout << "2\n"; },
    [] () { std::cout << "3\n"; },
    [] () { std::cout << "4\n"; },
    [] () { std::cout << "5\n"; },
    [] () { std::cout << "6\n"; },
    [] () { std::cout << "7\n"; },
    [] () { std::cout << "8\n"; }
  );

  A.name("A");
  B.name("B");
  C.name("C");
  D.name("D");
  E.name("E");
  F.name("F");
  G.name("G");
  H.name("H");
  
  // create a default observer
  auto observer = executor.make_observer<tf::ExecutorObserver>();

  // run the taskflow
  executor.run(taskflow).get();
  
  // dump the execution timeline to json (view at chrome://tracing)
  observer->dump(std::cout);

  return 0;
}

