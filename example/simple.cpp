// A simple example to capture the following task dependencies.
//
// TaskA---->TaskB---->TaskD
// TaskA---->TaskC---->TaskD

#include "taskflow.hpp"

int main(){

  tf::Taskflow tf(std::thread::hardware_concurrency());

  auto [A, B, C, D] = tf.silent_emplace(
    [] () { std::cout << "TaskA\n"; },
    [] () { std::cout << "TaskB\n"; },
    [] () { std::cout << "TaskC\n"; },
    [] () { std::cout << "TaskD\n"; }
  );

  A.name("A").precede(B);  // B runs after A
  A.precede(C);  // C runs after A
  B.precede(D);  // D runs after B
  C.precede(D);  // C runs after D

  std::cout << tf.dump_graphviz();

  tf.wait_for_all();  // block until all task finish

  return 0;
}

