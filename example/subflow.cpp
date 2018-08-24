// This example demonstrates how to use cpp-taskflow to create
// dynamic workload during execution.
//
// We first create four tasks A, B, C, and D. During the execution
// of B, it uses flow builder to creates another three tasks
// B1, B2, and B3, and adds dependencies from B1 and B2 to B3.

#include <taskflow.hpp>  

int main() {

  tf::Taskflow tf(std::thread::hardware_concurrency());

  auto [A, B, C, D] = tf.silent_emplace(
    // Task A
    [] () { std::cout << "TaskA\n"; },              
    // Task B
    [cap=std::vector<int>{1,2,3,4,5,6,7,8}] (auto& subflow) {                             

      std::cout << "TaskB\n";                                  

      auto B1 = subflow.silent_emplace([&]() { 
        printf("  Subtask B1: reduce sum = %d\n", 
                std::accumulate(cap.begin(), cap.end(), 0, std::plus<int>()));
      }).name("B1");        
      
      auto B2 = subflow.silent_emplace([&]() { 
        printf("  Subtask B2: reduce multiply = %d\n", 
                std::accumulate(cap.begin(), cap.end(), 1, std::multiplies<int>()));
      }).name("B2");        
                                                              
      auto B3 = subflow.silent_emplace([&]() { 
        printf("  Subtask B3: reduce minus = %d\n", 
                std::accumulate(cap.begin(), cap.end(), 0, std::minus<int>()));
      }).name("B3");        
                                                              
      B1.precede(B3);
      B2.precede(B3);
    },
    // Task C
    [] () { std::cout << "TaskC\n"; },               
    // Task D
    [] () { std::cout << "TaskD\n"; }                
  );                                                 
                                         
  A.name("A");
  B.name("B");
  C.name("C");
  D.name("D");
              
  A.precede(B);  // B runs after A 
  A.precede(C);  // C runs after A 
  B.precede(D);  // D runs after B 
  C.precede(D);  // D runs after C 
                                   
  tf.wait_for_all();  // block until finished

  return 0;
}


