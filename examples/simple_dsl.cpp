// 2020/08/28 - Created by netcan: https://github.com/netcan
// A simple example to capture the following task dependencies.
// using Task DSL to describe
// TaskA -> fork(TaskB, TaskC) -> TaskD

#include <taskflow/taskflow.hpp>  // the only include you need
#include <taskflow/dsl/task_dsl.hpp> // for support dsl

int main(){
  tf::Executor executor;
  tf::Taskflow taskflow("simple");
  def_task(A, { return []() { std::cout << "TaskA\n"; }; });
  def_task(B, { return []() { std::cout << "TaskB\n"; }; });
  def_task(C, { return []() { std::cout << "TaskC\n"; }; });
  def_task(D, { return []() { std::cout << "TaskD\n"; }; });

  taskbuild(            //          +---+
    chain(task(A)       //    +---->| B |-----+
        -> fork(B, C)   //    |     +---+     |
        -> task(D))     //  +---+           +-v-+
  ) {taskflow};         //  | A |           | D |
                        //  +---+           +-^-+
                        //    |     +---+     |
                        //    +---->| C |-----+
                        //          +---+

  executor.run(taskflow).wait();
  return 0;
}

