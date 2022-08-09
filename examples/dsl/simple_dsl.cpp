// 2020/08/28 - Created by netcan: https://github.com/netcan
// A simple example to capture the following task dependencies.
// using Task DSL to describe
// TaskA -> fork(TaskB, TaskC) -> TaskD
#include <taskflow/taskflow.hpp>     // the only include you need
#include <taskflow/dsl.hpp> // for support dsl

int main() {
  tf::Executor executor;
  tf::Taskflow taskflow("simple");
  make_task((A), { std::cout << "TaskA\n"; });
  make_task((B), { std::cout << "TaskB\n"; });
  make_task((C), { std::cout << "TaskC\n"; });
  make_task((D), { std::cout << "TaskD\n"; });

  build_taskflow(           //          +---+
    task(A)                 //    +---->| B |-----+
      ->fork_tasks(B, C)    //    |     +---+     |
      ->task(D)             //  +---+           +-v-+
  )(taskflow);              //  | A |           | D |
                            //  +---+           +-^-+
                            //    |     +---+     |
                            //    +---->| C |-----+
                            //          +---+

  executor.run(taskflow).wait();
  return 0;
}
