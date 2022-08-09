// 2020/08/28 - Created by netcan: https://github.com/netcan
// This example demonstrates how to use 'dump' method to inspect
// a taskflow graph.
// use task dsl
#include <taskflow/taskflow.hpp>
#include <taskflow/dsl.hpp> // for support dsl

int main() {
  tf::Taskflow tf("Visualization Demo");

  // ------------------------------------------------------
  // Static Tasking
  // ------------------------------------------------------
  make_task((A), { std::cout << "TaskA\n"; };);
  make_task((B), { std::cout << "TaskB\n"; };);
  make_task((C), { std::cout << "TaskC\n"; };);
  make_task((D), { std::cout << "TaskD\n"; };);
  make_task((E), { std::cout << "TaskE\n"; };);

  auto tasks = build_taskflow(
    task(A)
      -> fork_tasks(B, C)
      -> task(D),
    merge_tasks(A, B)
      -> task(E)
  )(tf);

  std::cout << "[dump without name assignment]\n";
  tf.dump(std::cout);

  std::cout << "[dump with name assignment]\n";
  tasks.get_task<A>().name("A");
  tasks.get_task<B>().name("B");
  tasks.get_task<C>().name("C");
  tasks.get_task<D>().name("D");
  tasks.get_task<E>().name("E");
  tf.dump(std::cout);

  // ------------------------------------------------------
  // Dynamic Tasking
  // ------------------------------------------------------

  return 0;
}
