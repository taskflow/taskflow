// 2019/05/14 - modified by Tsung-Wei Huang
//  - isolated executor from taskflow
//
// 2019/02/15 - modified by Tsung-Wei Huang
//  - refactored the code  
//
// 2019/01/17 - created by Chun-Xun
//  - added example of using the Framework

#include <taskflow/taskflow.hpp>  

int main(){
  
  // create an executor and a taskflow
  tf::Executor executor;
  tf::Taskflow taskflow;

  taskflow.name("Demo");

  auto A = taskflow.emplace([&](){ std::cout << "TaskA\n"; }).name("A");
  auto B = taskflow.emplace([&](auto& subflow){ 
    std::cout << "TaskB\n";
    auto B1 = subflow.emplace([&](){ std::cout << "TaskB1\n"; }).name("B1");
    auto B2 = subflow.emplace([&](){ std::cout << "TaskB2\n"; }).name("B2");
    auto B3 = subflow.emplace([&](){ std::cout << "TaskB3\n"; }).name("B3");
    B1.precede(B3); 
    B2.precede(B3);
  }).name("B");

  auto C = taskflow.emplace([&](){ std::cout << "TaskC\n"; }).name("C");
  auto D = taskflow.emplace([&](){ std::cout << "TaskD\n"; }).name("D");

  A.precede(B, C);
  B.precede(D); 
  C.precede(D);

  std::cout << "Run the framework once without callback\n";

  std::cout << "Dump the framework before execution:\n";
  taskflow.dump(std::cout);
  std::cout << std::endl;

  executor.run(taskflow).get();
  std::cout << std::endl;

  std::cout << "Dump after execution:\n";
  taskflow.dump(std::cout);
  std::cout << std::endl;

  std::cout << "Use wait_for_all to wait for the run to finish\n";
  executor.run(taskflow);
  executor.wait_for_all();
  std::cout << std::endl;

  std::cout << "Execute the framework two times without a callback\n";
  executor.run_n(taskflow, 2).get();
  std::cout << "Dump after two executions:\n";
  taskflow.dump(std::cout);
  std::cout << std::endl;

  std::cout << "Execute the framework four times with a callback\n";
  executor.run_n(taskflow, 4, [] () { std::cout << "The framework finishes\n"; }).get();
  std::cout << std::endl;

  std::cout << "Run the framework until a counter (init value=3) becomes zero\n"; 
  executor.run_until(taskflow, [counter=3]() mutable { 
    std::cout << "Counter = " << counter << std::endl; 
    return counter -- == 0; 
  }).get();

  taskflow.dump(std::cout);

  return 0;
}
