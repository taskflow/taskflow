// This example demonstrates how to use 'dump' method to inspect 
// a taskflow graph.

#include "taskflow.hpp"

int main(){

  tf::Taskflow tf;

  auto [A, B, C, D, E] = tf.silent_emplace(
    [] () { std::cout << "Task A" << std::endl; },
    [] () { std::cout << "Task B" << std::endl; },
    [] () { std::cout << "Task C" << std::endl; },
    [] () { std::cout << "Task D" << std::endl; },
    [] () { std::cout << "Task E" << std::endl; }
  );

  A.broadcast(B, C, E); 
  C.precede(D);
  B.broadcast(D, E); 
  
  std::cout << "[dump without name assignment]\n";
  std::cout << tf.dump() << std::endl;
  
  std::cout << "[dump with name assignment]\n";
  A.name("A");
  B.name("B");
  C.name("C");
  D.name("D");
  E.name("E");
  std::cout << tf.dump() << std::endl;

  return 0;
}
