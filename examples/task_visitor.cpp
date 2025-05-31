/**
 This program demonstrates how to traverse a graph using task visitor.

 We first create four tasks: A, B, C, and D, where task A runs before B and C,
 and task D runs after B and C. During the execution of B, it spawns another subflow
 graph of three tasks: B1, B2, and B3, where B3 runs after B1 and B2.
 Upon completion of the subflow, it joins its parent task B.
 
 By default, subflows are automatically cleaned up when they finish to avoid memory explosion. 
 In this example, since we would like to inspect the spawned subflow,
 we disable this behavior by calling `tf::Subflow::retain(true)`.
 
 Note that we must run the subflow once for it to be created.
*/

#include <taskflow/taskflow.hpp>

int main() {

  // Create a taskflow graph with three static tasks and one subflow task.
  tf::Taskflow taskflow("visitor");
  tf::Executor executor;

  auto A = taskflow.emplace([]() { std::cout << "TaskA\n"; });
  auto B = taskflow.emplace([](tf::Subflow& subflow) {
    std::cout << "TaskB is spawning B1, B2, and B3 ...\n";
    auto B1 = subflow.emplace([&](){ printf("  Subtask B1\n"); }).name("B1");
    auto B2 = subflow.emplace([&](){ printf("  Subtask B2\n"); }).name("B2");
    auto B3 = subflow.emplace([&](){ printf("  Subtask B3\n"); }).name("B3");
    B1.precede(B3);
    B2.precede(B3);
    subflow.retain(true);  // retains the subflow
  });

  auto C = taskflow.emplace([] () { std::cout << "TaskC\n"; });
  auto D = taskflow.emplace([] () { std::cout << "TaskD\n"; });
  A.name("A");
  B.name("B");
  C.name("C");
  D.name("D");

  A.precede(B);  // B runs after A
  A.precede(C);  // C runs after A
  B.precede(D);  // D runs after B
  C.precede(D);  // D runs after C

  executor.run(taskflow).wait();

  // examine the graph
  taskflow.dump(std::cout);

  // traverse all tasks in the taskflow
  taskflow.for_each_task([](tf::Task task){
    std::cout << "task " << task.name() << " [type=" << tf::to_string(task.type()) << "]\n";
    // traverse it's successor
    task.for_each_successor([](tf::Task successor) {
      std::cout << "  -> successor   task " << successor.name() << '\n'; 
    });
    // traverse it's predecessor
    task.for_each_predecessor([](tf::Task predecessor) {
      std::cout << "  <- predecessor task " << predecessor.name() << '\n'; 
    });

    // traverse the subflow (in our example, task B)
    task.for_each_subflow_task([](tf::Task stask){
      std::cout << "  subflow task " << stask.name() << '\n';
      // traverse it's successor
      stask.for_each_successor([](tf::Task successor) {
        std::cout << "    -> successor   task " << successor.name() << '\n'; 
      });
      // traverse it's predecessor
      stask.for_each_predecessor([](tf::Task predecessor) {
        std::cout << "    <- predecessor task " << predecessor.name() << '\n'; 
      });
    });
  });

  return 0;
}



