// This program demonstrates how to launch taskflows using asynchronous tasking.

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/module.hpp>

int main() {

  tf::Executor executor;

  tf::Taskflow A;
  tf::Taskflow B;
  tf::Taskflow C;
  tf::Taskflow D;

  A.emplace([](){ printf("Taskflow A\n"); });
  B.emplace([](){ printf("Taskflow B\n"); });
  C.emplace([](){ printf("Taskflow C\n"); });
  D.emplace([](){ printf("Taskflow D\n"); });

  // launch the four taskflows using async
  printf("launching four taskflows using async ...\n");
  executor.async(tf::make_module_task(A));
  executor.async(tf::make_module_task(B));
  executor.async(tf::make_module_task(C));
  executor.async(tf::make_module_task(D));
  executor.wait_for_all();

  // launch four taskflows with dependencies
  printf("launching four taskflows using dependent async ...\n");
  auto TA = executor.silent_dependent_async(tf::make_module_task(A));
  auto TB = executor.silent_dependent_async(tf::make_module_task(B), TA);
  auto TC = executor.silent_dependent_async(tf::make_module_task(C), TB);
  auto [TD, FD] = executor.dependent_async(tf::make_module_task(D), TC);
  FD.get();

  return 0;
}
