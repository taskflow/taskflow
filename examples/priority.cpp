// This program demonstrates how to set priority to a task.
//
// Currently, Taskflow supports only three priority levels:
//   + tf::TaskPriority::HIGH   (numerical value = 0)
//   + tf::TaskPriority::NORMAL (numerical value = 1)
//   + tf::TaskPriority::LOW    (numerical value = 2)
// 
// Priority-based execution is non-preemptive. Once a task 
// has started to execute, it will execute to completion,
// even if a higher priority task has been spawned or enqueued. 

#include <taskflow/taskflow.hpp>

int main() {
  
  // create an executor of only one worker to enable 
  // deterministic behavior
  tf::Executor executor(1);

  tf::Taskflow taskflow;

  int counter {0};
  
  // Here we create five tasks and print thier execution
  // orders which should align with assigned priorities
  auto [A, B, C, D, E] = taskflow.emplace(
    [] () { },
    [&] () { 
      std::cout << "Task B: " << counter++ << '\n';  // 0
    },
    [&] () { 
      std::cout << "Task C: " << counter++ << '\n';  // 2
    },
    [&] () { 
      std::cout << "Task D: " << counter++ << '\n';  // 1
    },
    [] () { }
  );

  A.precede(B, C, D); 
  E.succeed(B, C, D);
  
  // By default, all tasks are of tf::TaskPriority::HIGH
  B.priority(tf::TaskPriority::HIGH);
  C.priority(tf::TaskPriority::LOW);
  D.priority(tf::TaskPriority::NORMAL);

  assert(B.priority() == tf::TaskPriority::HIGH);
  assert(C.priority() == tf::TaskPriority::LOW);
  assert(D.priority() == tf::TaskPriority::NORMAL);
  
  // we should see B, D, and C in their priority order
  executor.run(taskflow).wait();
}

