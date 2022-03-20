// This program demonstrates how to implement do-while control flow
// using condition tasks.
#include <taskflow/taskflow.hpp>

int main() {

  tf::Executor executor;
  tf::Taskflow taskflow;

  int i;

  auto [init, body, cond, done] = taskflow.emplace(
    [&](){ std::cout << "i=0\n"; i=0; },
    [&](){ std::cout << "i++ => i="; i++; },
    [&](){ std::cout << i << '\n'; return i<5 ? 0 : 1; },
    [&](){ std::cout << "done\n"; }
  );

  init.name("init");
  body.name("do i++");
  cond.name("while i<5");
  done.name("done");

  init.precede(body);
  body.precede(cond);
  cond.precede(body, done);

  //taskflow.dump(std::cout);

  executor.run(taskflow).wait();

  return 0;
}

