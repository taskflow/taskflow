// A simple example to capture the following task dependencies.
//
// TaskA---->TaskB---->TaskD
// TaskA---->TaskC---->TaskD

#include <taskflow/taskflow.hpp>  // the only include you need
#include <new>

struct keep_apart {
  alignas(64) std::atomic<int> cat;
  alignas(64) std::atomic<int> dog;
};

int main(){

  std::cout << sizeof(keep_apart);

  tf::Taskflow tf;
  
  auto [A, B, C, D] = tf.silent_emplace(   //  the taskflow graph
    [] () { std::cout << "TaskA\n"; },     //                                 
    [] () { std::cout << "TaskB\n"; },     //          +---+                  
    [] () { std::cout << "TaskC\n"; },     //    +---->| B |-----+            
    [] () { std::cout << "TaskD\n"; }      //    |     +---+     |            
  );                                       //  +---+           +-v-+          
                                           //  | A |           | D |          
  A.precede(B);  // B runs after A         //  +---+           +-^-+          
  A.precede(C);  // C runs after A         //    |     +---+     |            
  B.precede(D);  // D runs after B         //    +---->| C |-----+            
  C.precede(D);  // D runs after C         //          +---+

  tf.wait_for_all();  // block until finished

  return 0;
}
