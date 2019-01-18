// 2019/01/17 - created by Chun-Xun
//   - added example of using the Framework

#include <taskflow/taskflow.hpp>  

int main(){

  tf::Taskflow tf;

  // Create a framework
  tf::Framework f;
  auto A = f.silent_emplace([&](){ std::cout << "TaskA\n"; });
  auto B = f.silent_emplace([&](auto& subflow){ 
    std::cout << "TaskB\n";
    auto B1 = subflow.silent_emplace([&](){ std::cout << "TaskB1\n"; });
    auto B2 = subflow.silent_emplace([&](){ std::cout << "TaskB2\n"; });
    auto B3 = subflow.silent_emplace([&](){ std::cout << "TaskB3\n"; });
    B1.precede(B3); 
    B2.precede(B3);
  });
  auto C = f.silent_emplace([&](){ std::cout << "TaskC\n"; });
  auto D = f.silent_emplace([&](){ std::cout << "TaskD\n"; });

  A.precede(B, C);
  B.precede(D); 
  C.precede(D);

  std::cout << "Run the framework once without callback\n";
  auto future = tf.run(f); 
  future.get();
  std::cout << std::endl;

  // TODO:
  // tf.dump_topologies(std::cout);

  std::cout << "Use wait_for_all to wait for the run to finish\n";
  tf.run(f);
  tf.wait_for_all();
  std::cout << std::endl;

  std::cout << "Execute the framework 2 times without a callback\n";
  tf.run_n(f, 2).get();
  std::cout << std::endl;

  std::cout << "Execute the framework 4 times with a callback\n";
  tf.run_n(f, 4, [i=0] () mutable { std::cout << "-> run #" << ++i << " finished\n"; }).get();
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

  return 0;
}
