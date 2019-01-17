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
    B1.precede(B3); B2.precede(B3);
  });
  auto C = f.silent_emplace([&](){ std::cout << "TaskC\n"; });
  auto D = f.silent_emplace([&](){ std::cout << "TaskD\n"; });

  A.precede(B, C);
  B.precede(D); C.precede(D);


  std::puts("Run the framework without callback");
  auto future = tf.run(f); 
  future.get();
  std::puts("");

  std::puts("Use wait_for_all to wait for the run finished");
  tf.run(f);
  tf.wait_for_all();
  std::puts("");

  std::puts("Execute the framework 2 times without callback");
  tf.run_n(f, 2).get();
  std::puts("");

  std::puts("Execute the framework 4 times with a callback");
  tf.run_n(f, 4, [i=0]() mutable { std::cout << "Repeat " << ++i << "\n"; }).get();
  std::puts("");


  std::puts("Silently run the framework");
  tf.silent_run(f); 
  tf.wait_for_all();
  std::puts("");

  std::puts("Silently run the framework 2 times ");
  tf.silent_run_n(f, 2); 
  tf.wait_for_all();
  std::puts("");

  std::puts("Silently run the framework with callback");
  tf.silent_run_n(f, 1, []() { std::cout << "The framework finishes\n"; }); 
  tf.wait_for_all();
  std::puts("");

  return 0;
}
