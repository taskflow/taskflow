// This program demonstrates how to implement while-loop control flow
// using condition tasks.
#include <taskflow/taskflow.hpp>

int main() {
  
  tf::Executor executor;
  tf::Taskflow taskflow;

  int i;
  
  auto [init, cond, body, back, done] = taskflow.emplace(
    [&](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf){ std::cout << "i=0\n"; i=0; },
    [&](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf){ std::cout << "while i<5\n"; return i < 5 ? 0 : 1; },
    [&](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf){ std::cout << "i++=" << i++ << '\n'; },
    [&](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf){ std::cout << "back\n"; return 0; },
    [&](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf){ std::cout << "done\n"; }
  );

  init.name("init");
  cond.name("while i<5");
  body.name("i++");
  back.name("back");
  done.name("done");

  init.precede(cond);
  cond.precede(body, done);
  body.precede(back);
  back.precede(cond);
  
  taskflow.dump(std::cout);

  executor.run(taskflow).wait();

  return 0;
}

