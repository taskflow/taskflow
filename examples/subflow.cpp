/**
 This example demonstrates how to use Taskflow to create a subflow during the
 execution of a task.
 
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
  tf::Executor executor(4);
  tf::Taskflow taskflow("Subflow Demo");

  auto A = taskflow.emplace([] () { std::cout << "TaskA\n"; });
  auto B = taskflow.emplace(
    [cap=std::vector<int>{1,2,3,4,5,6,7,8}] (tf::Subflow& subflow) {
      std::cout << "TaskB is spawning B1, B2, and B3 ...\n";

      auto B1 = subflow.emplace([&]() {
        printf("  Subtask B1: reduce sum = %d\n",
                std::accumulate(cap.begin(), cap.end(), 0, std::plus<int>()));
      }).name("B1");

      auto B2 = subflow.emplace([&]() {
        printf("  Subtask B2: reduce multiply = %d\n",
                std::accumulate(cap.begin(), cap.end(), 1, std::multiplies<int>()));
      }).name("B2");

      auto B3 = subflow.emplace([&]() {
        printf("  Subtask B3: reduce minus = %d\n",
                std::accumulate(cap.begin(), cap.end(), 0, std::minus<int>()));
      }).name("B3");

      B1.precede(B3);
      B2.precede(B3);
      
      // retain the subflow for visualization purpose
      subflow.retain(true);
    }
  );

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

  executor.run_n(taskflow, 3).get();  // block until finished

  // examine the graph
  taskflow.dump(std::cout);

  return 0;
}



