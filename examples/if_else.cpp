// This program demonstrates how to create if-else control flow
// using condition tasks.
#include <taskflow/taskflow.hpp>

int main() {

  tf::Executor executor;
  tf::Taskflow taskflow;

  // create three static tasks and one condition task
  auto [init, cond, yes, no] = taskflow.emplace(
    [] () { },
    [] () { return 0; },
    [] () { std::cout << "yes\n"; },
    [] () { std::cout << "no\n"; }
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

