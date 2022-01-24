// This program demonstrates how to use a runtime task to forcefully
// schedule an active task that would never be scheduled.

#include <taskflow/taskflow.hpp>
  
int main() {  
  
  tf::Taskflow taskflow("Runtime Tasking");
  tf::Executor executor;

  tf::Task A, B, C, D;

  std::tie(A, B, C, D) = taskflow.emplace(
    [] (tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf) { return 0; },
    [&C] (tf::Runtime& rt, tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf) {  // C must be captured by reference
      std::cout << "B\n"; 
      rt.schedule(C);
    },
    [] (tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf) { std::cout << "C\n"; },
    [] (tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf) { std::cout << "D\n"; }
  );

  // name tasks
  A.name("A");
  B.name("B");
  C.name("C");
  D.name("D");

  // create conditional dependencies
  A.precede(B, C, D);

  // dump the graph structure
  taskflow.dump(std::cout);
  
  // we will see both B and C in the output
  executor.run(taskflow).wait();

  return 0;
}
