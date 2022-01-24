// This program demonstrates how to implement switch-case control flow
// using condition tasks.
#include <taskflow/taskflow.hpp>

int main() {
  
  tf::Executor executor;
  tf::Taskflow taskflow;
  
  auto [source, swcond, case1, case2, case3, target] = taskflow.emplace(
    [](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf){ std::cout << "source\n"; },
    [](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf){ std::cout << "switch\n"; return rand()%3; },
    [](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf){ std::cout << "case 1\n"; return 0; },
    [](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf){ std::cout << "case 2\n"; return 0; },
    [](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf){ std::cout << "case 3\n"; return 0; },
    [](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf){ std::cout << "target\n"; }
  );
  
  source.precede(swcond);
  swcond.precede(case1, case2, case3);
  target.succeed(case1, case2, case3);

  executor.run(taskflow).wait();

  return 0;
}
