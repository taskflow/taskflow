#include <taskflow.hpp>

// A simple example to capture the following task dependencies.
//
// TaskA---->TaskB---->TaskD
// TaskA---->TaskC---->TaskD
//
int main(){
  
  tf::Taskflow<> tf(std::thread::hardware_concurrency());

  auto [A, B, C, D] = tf.silent_emplace(
    [] () { std::cout << "TaskA\n"; },
    [] () { std::cout << "TaskB\n"; },
    [] () { std::cout << "TaskC\n"; },
    [] () { std::cout << "TaskD\n"; }
  );

  A.precede(B);
  A.precede(C);
  B.precede(D);
  C.precede(D);

  tf.wait_for_all(); 

  return 0;
}

