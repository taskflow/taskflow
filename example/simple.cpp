// A simple example to capture the following task dependencies.
//
// TaskA---->TaskB---->TaskD
// TaskA---->TaskC---->TaskD

#include "taskflow.hpp"

int main(){

tf::Taskflow tf(0);  // force the master thread to execute all tasks
auto A = tf.silent_emplace([] () { }).name("A");
auto B = tf.silent_emplace([] () { }).name("B");
auto C = tf.silent_emplace([] () { }).name("C");
auto D = tf.silent_emplace([] () { }).name("D");
auto E = tf.silent_emplace([] () { }).name("E");

A.broadcast(B, C, E);
C.precede(D);
B.broadcast(D, E);

std::cout << tf.dump_graphviz();

 return 0;
/*
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

  tf.wait_for_all();  // block until all task finish
*/
  return 0;
}

