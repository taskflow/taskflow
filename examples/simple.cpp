// A simple example to capture the following task dependencies.
//
// TaskA---->TaskB---->TaskD
// TaskA---->TaskC---->TaskD

#include <taskflow/taskflow.hpp>  // the only include you need

int main(){

  tf::Taskflow taskflow ("Error 1: no source");
  
  auto E = taskflow.emplace([](){}).name("E");
  auto C = taskflow.emplace([](){return ::rand()%2; }).name("C");
  auto D = taskflow.emplace([](){}).name("D");
  auto D_aux = taskflow.emplace([](){}).name("D_aux");
  auto F = taskflow.emplace([](){}).name("F");
  E.precede(D);
  C.precede(D_aux, F);
  D_aux.precede(D);

  taskflow.dump(std::cout);  

  //tf::Taskflow taskflow1("taskflow1");
  //tf::Taskflow taskflow2("taskflow2");

  //auto [A, B] = taskflow1.emplace(
  //  [] () { std::cout << "TaskA"; },
  //  [] () { std::cout << "TaskB"; }
  //);
  //A.precede(B);
  //
  //auto [C, D] = taskflow2.emplace(
  //  [] () { std::cout << "TaskC"; },
  //  [] (tf::Subflow& sf) { 
  //    std::cout << "TaskD"; 
  //    auto [D1, D2] = sf.emplace(
  //      [] () { std::cout << "D1"; },
  //      [] () { std::cout << "D2"; }
  //    );
  //    D1.precede(D2);
  //  }
  //);
  //C.precede(D);

  //auto E = taskflow2.composed_of(taskflow1);
  //D.precede(E);

  //executor.run(taskflow2).wait();

  //taskflow2.dump(std::cout);

  //auto A = taskflow.emplace([]() { std::cout << "TaskA\n"; });
  //auto B = taskflow.emplace([]() { std::cout << "TaskB\n"; });
  //auto C = taskflow.emplace([]() { std::cout << "TaskC\n"; });
  //auto D = taskflow.emplace([]() { std::cout << "TaskD\n"; });

  //A.precede(B);  // B runs after A  //          +---+                  
  //A.precede(C);  // C runs after A  //    +---->| B |-----+            
  //B.precede(D);  // D runs after B  //    |     +---+     |            
  //C.precede(D);  // D runs after C  //  +---+           +-v-+          
  //                                  //  | A |           | D |          
  //                                  //  +---+           +-^-+          
  //executor.run(taskflow).wait();    //    |     +---+     |            
  //                                  //    +---->| C |-----+            
  //return 0;                         //          +---+
}

