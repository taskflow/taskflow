// A simple example to capture the following task dependencies.
//
// TaskA---->TaskB---->TaskD
// TaskA---->TaskC---->TaskD

#include <taskflow/taskflow.hpp>  // the only include you need

int main(){

  tf::Executor executor;
  tf::Taskflow taskflow;

  auto A = taskflow.emplace([]() { std::cout << "TaskA\n"; });
  auto B = taskflow.emplace([]() { std::cout << "TaskB\n"; });
  auto C = taskflow.emplace([]() { std::cout << "TaskC\n"; });
  auto D = taskflow.emplace([]() { std::cout << "TaskD\n"; });

                                    //                                 
  A.precede(B);  // B runs after A  //          +---+                  
  A.precede(C);  // C runs after A  //    +---->| B |-----+            
  B.precede(D);  // D runs after B  //    |     +---+     |            
  C.precede(D);  // D runs after C  //  +---+           +-v-+          
                                    //  | A |           | D |          
                                    //  +---+           +-^-+          
  executor.run(taskflow).wait();    //    |     +---+     |            
                                    //    +---->| C |-----+            
  return 0;                         //          +---+
}



