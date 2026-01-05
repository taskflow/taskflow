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

#include <iostream>
#include <print>

import tf;

using tf::Executor;
using tf::Subflow;
using tf::Task;
using tf::Taskflow;

int main() {
    // Create a taskflow graph with three static tasks and one subflow task.
    Taskflow taskflow("visitor");
    Executor executor;

    Task A = taskflow.emplace([]() -> void { std::println("TaskA"); });
    Task B = taskflow.emplace([](Subflow& subflow) -> void {
        std::println("TaskB is spawning B1, B2, and B3 ...");
        Task B1 = subflow.emplace([&]() -> void { std::println("  Subtask B1"); }).name("B1");
        Task B2 = subflow.emplace([&]() -> void { std::println("  Subtask B2"); }).name("B2");
        Task B3 = subflow.emplace([&]() -> void { std::println("  Subtask B3"); }).name("B3");
        B1.precede(B3);
        B2.precede(B3);
        subflow.retain(true); // retains the subflow
    });

    Task C = taskflow.emplace([] () -> void { std::println("TaskC"); });
    Task D = taskflow.emplace([] () -> void { std::println("TaskD"); });
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
    taskflow.for_each_task([](Task task) -> void {
        std::println("task {} [type={}]", task.name(), tf::to_string(task.type()));
        // traverse its successor
        task.for_each_successor([](Task successor) -> void {
            std::println("  -> successor   task {}", successor.name());
        });
        // traverse its predecessor
        task.for_each_predecessor([](Task predecessor) -> void {
            std::println("  <- predecessor task {}", predecessor.name());
        });

        // traverse the subflow (in our example, task B)
        task.for_each_subflow_task([](Task stask) -> void {
            std::println("  subflow task {}", stask.name());
            // traverse its successor
            stask.for_each_successor([](Task successor) -> void {
                std::println("    -> successor   task {}", successor.name());
            });
            // traverse its predecessor
            stask.for_each_predecessor([](Task predecessor) -> void {
                std::println("    <- predecessor task {}", predecessor.name());
            });
        });
    });

    return 0;
}
