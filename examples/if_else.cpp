// This program demonstrates how to create if-else control flow
// using condition tasks.
#include <taskflow/taskflow.hpp>

int main() {
  
  tf::Executor executor;
  tf::Taskflow taskflow;
  
  // create three static tasks and one condition task
  auto [init, cond, yes, no] = taskflow.emplace(
    [] (tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow* pf) { },
    [] (tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow* pf) { return 0; },
    [] (tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow* pf) { std::cout << "yes\n"; },
    [] (tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow* pf) { std::cout << "no\n"; }
  );

  init.name("init");
  cond.name("cond");
  yes.name("yes");
  no.name("no");
  
  cond.succeed(init);

  // With this order, when cond returns 0, execution
  // moves on to yes. When cond returns 1, execution
  // moves on to no.
  cond.precede(yes, no);
  
  // dump the conditioned flow
  taskflow.dump(std::cout);

  executor.run(taskflow).wait();

  return 0;
}

