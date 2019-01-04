// This example demonstrates how to use 'dump' method to inspect 
// a taskflow graph.

#include <taskflow/taskflow.hpp>

int main(){

  tf::Taskflow tf;

  auto [A, B, C, D, E] = tf.silent_emplace(
    [] () { std::cout << "Task A" << std::endl; },
    [] () { std::cout << "Task B" << std::endl; },
    [] () { std::cout << "Task C" << std::endl; },
    [] () { std::cout << "Task D" << std::endl; },
    [] () { std::cout << "Task E" << std::endl; }
  );

  A.precede(B, C, E); 
  C.precede(D);
  B.precede(D, E); 
  
  std::cout << "[dump without name assignment]\n";
  tf.dump(std::cout);
  
  std::cout << "[dump with name assignment]\n";
  A.name("A");
  B.name("B");
  C.name("C");
  D.name("D");
  E.name("E");
  tf.dump(std::cout);

  return 0;
}
