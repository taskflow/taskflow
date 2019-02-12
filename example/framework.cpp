// 2019/01/17 - created by Chun-Xun
//   - added example of using the Framework

#include <taskflow/taskflow.hpp>  

int main(){

  tf::Taskflow tf;

  // Create a framework
  tf::Framework f;
  auto A = f.silent_emplace([&](){ std::cout << "TaskA\n"; }).name("A");
  auto B = f.silent_emplace([&](auto& subflow){ 
    std::cout << "TaskB\n";
    auto B1 = subflow.silent_emplace([&](){ std::cout << "TaskB1\n"; }).name("B1");
    auto B2 = subflow.silent_emplace([&](){ std::cout << "TaskB2\n"; }).name("B2");
    auto B3 = subflow.silent_emplace([&](){ std::cout << "TaskB3\n"; }).name("B3");
    B1.precede(B3); 
    B2.precede(B3);
  }).name("B");
  auto C = f.silent_emplace([&](){ std::cout << "TaskC\n"; }).name("C");
  auto D = f.silent_emplace([&](){ std::cout << "TaskD\n"; }).name("D");

  A.precede(B, C);
  B.precede(D); 
  C.precede(D);

  std::cout << "Run the framework once without callback\n";

  std::cout << "Dump before execution:\n";
  f.dump(std::cout);
  std::cout << std::endl;

  auto future = tf.run(f); 
  future.get();
  std::cout << std::endl;

  std::cout << "Dump after execution:\n";
  tf.dump_topologies(std::cout);
  std::cout << std::endl;

  std::cout << "Use wait_for_all to wait for the run to finish\n";
  tf.run(f);
  tf.wait_for_all();
  std::cout << std::endl;

  std::cout << "Execute the framework 2 times without a callback\n";
  tf.run_n(f, 2).get();
  std::cout << "Dump after 2 executions:\n";
  tf.dump_topologies(std::cout);
  std::cout << std::endl;

  std::cout << "Execute the framework 4 times with a callback\n";
  tf.run_n(f, 4, [] () mutable { std::cout << "The framework finishes\n"; }).get();
  std::cout << std::endl;

  std::cout << "Silently run the framework\n";
  tf.silent_run(f); 
  tf.wait_for_all();
  std::cout << std::endl;

  std::cout << "Silently run the framework 2 times \n";
  tf.silent_run_n(f, 2); 
  tf.wait_for_all();
  std::cout << std::endl;

  std::cout << "Silently run the framework with callback\n";
  tf.silent_run_n(f, 1, []() { std::cout << "The framework finishes\n"; }); 
  tf.wait_for_all();
  std::cout << std::endl;

  std::cout << "Run the framework until a counter (init value=3) becomes zero\n"; 
  tf.run_until(f, [counter=3]() mutable { 
    std::cout << "Counter = " << counter << std::endl; 
    return counter -- == 0; 
  }).get();

  return 0;
}
