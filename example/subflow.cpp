#include "taskflow.hpp"  // the only include you need

int main(){
  tf::Taskflow tf(std::thread::hardware_concurrency());

  auto [A, B, C, D] = tf.silent_emplace(
    [] () { std::cout << "TaskA\n"; },              
    [cap=std::vector<int>{1,2,3,4,5,6,7,8}] (tf::FlowBuilder& fb) {                             
      std::cout << "TaskB\n";                                  

      auto B1 = fb.silent_emplace([&]() { 
        printf("  Subtask B1: reduce sum = %d\n", 
                std::accumulate(cap.begin(), cap.end(), 0, std::plus<int>()));
      }).name("B1");        
      
      auto B2 = fb.silent_emplace([&]() { 
        printf("  Subtask B2: reduce multiply = %d\n", 
                std::accumulate(cap.begin(), cap.end(), 1, std::multiplies<int>()));
      }).name("B2");        
                                                              
      auto B3 = fb.silent_emplace([&]() { 
        printf("  Subtask B3: reduce minus = %d\n", 
                std::accumulate(cap.begin(), cap.end(), 0, std::minus<int>()));
      }).name("B3");        
                                                              
      B1.precede(B3);
      B2.precede(B3);
    },
    [] () { std::cout << "TaskC\n"; },               
    [] () { std::cout << "TaskD\n"; }                
  );                                                 
                                                        //  the taskflow graph
  A.name("A");                                          // 
  B.name("B");                                          //          +---+         
  C.name("C");                                          //    +---->| B |-----+   
  D.name("D");                                          //    |     +---+     |
                                                        //  +---+           +-v-+ 
  A.precede(B);  // B runs after A                      //  | A |           | D | 
  A.precede(C);  // C runs after A                      //  +---+           +-^-+ 
  B.precede(D);  // D runs after B                      //    |     +---+     |   
  C.precede(D);  // D runs after C                      //    +---->| C |-----+   
                                                        //          +---+         
  tf.wait_for_all();  // block until finished

  return 0;
}
