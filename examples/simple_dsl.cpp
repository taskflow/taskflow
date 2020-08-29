// 2020/08/28 - Created by netcan: https://github.com/netcan
// A simple example to capture the following task dependencies.
// using to describe
// TaskA---->TaskB---->TaskD
// TaskA---->TaskC---->TaskD

#include <taskflow/taskflow.hpp>  // the only include you need
#include <taskflow/dsl/task_dsl.hpp> // for support dsl

int main(){
  tf::Executor executor;
  tf::Taskflow taskflow("simple");
  __def_task(A, { return []() { std::cout << "TaskA\n"; }; });
  __def_task(B, { return []() { std::cout << "TaskB\n"; }; });
  __def_task(C, { return []() { std::cout << "TaskC\n"; }; });
  __def_task(D, { return []() { std::cout << "TaskD\n"; }; });

  __taskbuild(          //          +---+
    __chain(__tsk(A)    //    +---->| B |-----+
        -> __fork(B, C) //    |     +---+     |
        -> __tsk(D))    //  +---+           +-v-+
  ) {taskflow};         //  | A |           | D |
                        //  +---+           +-^-+
                        //    |     +---+     |
                        //    +---->| C |-----+
                        //          +---+

  executor.run(taskflow).wait();
  return 0;
}

