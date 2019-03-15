// 2019/03/12 - created by Chun-Xun Lin
//  - added an example showing how to use framework decomposition

#include <taskflow/taskflow.hpp>  // the only include you need

void composition_example_1(tf::Taskflow& tf) {
  std::cout << '\n';
  std::cout << "Composition example 1\n"; 

  // f1 has two independent tasks
  tf::Framework f1;
  auto [f1A, f1B] = f1.name("F1").emplace(
    [&](){ std::cout << "F1 TaskA\n"; },
    [&](){ std::cout << "F1 TaskB\n"; }
  );
  f1A.name("f1A");
  f1B.name("f1B");

  // f2A ---
  //        |----> f2C ----> f1_module_task
  // f2B --- 
  tf::Framework f2;
  auto [f2A, f2B, f2C] = f2.name("F2").emplace(
    [&](){ std::cout << "  F2 TaskA\n"; },
    [&](){ std::cout << "  F2 TaskB\n"; },
    [&](){ std::cout << "  F2 TaskC\n"; }
  );
  f2A.name("f2A");
  f2B.name("f2B");
  f2C.name("f2C");

  f2A.precede(f2C);
  f2B.precede(f2C);
  
  auto f1_module_task = f2.composed_of(f1).name("module");
  f2C.precede(f1_module_task);

  f2.dump(std::cout);

  tf.run_n(f2, 3).get();
}

void composition_example_2(tf::Taskflow& tf) {
  std::cout << '\n';
  std::cout << "Composition example 2\n"; 

  // f1 has two independent tasks
  tf::Framework f1;
  auto [f1A, f1B] = f1.name("F1").emplace(
    [&](){ std::cout << "F1 TaskA\n"; },
    [&](){ std::cout << "F1 TaskB\n"; }
  );
  f1A.name("f1A");
  f1B.name("f1B");

  //  f2A ---
  //         |----> f2C
  //  f2B --- 
  //
  //  f1_module_task
  tf::Framework f2;
  auto [f2A, f2B, f2C] = f2.name("F2").emplace(
    [&](){ std::cout << "  F2 TaskA\n"; },
    [&](){ std::cout << "  F2 TaskB\n"; },
    [&](){ std::cout << "  F2 TaskC\n"; }
  );
  f2A.name("f2A");
  f2B.name("f2B");
  f2C.name("f2C");

  f2A.precede(f2C);
  f2B.precede(f2C);
  f2.composed_of(f1);

  // f3 has a module task (f2) and a regular task
  tf::Framework f3;
  f3.name("F3");
  f3.composed_of(f2);
  f3.emplace([](){ std::cout << "      F3 TaskA\n"; }).name("f3A");

  // f4: f3_module_task -> f2_module_task
  tf::Framework f4; 
  f4.name("F4");
  auto f3_module_task = f4.composed_of(f3);
  auto f2_module_task = f4.composed_of(f2);
  f3_module_task.precede(f2_module_task);

  f4.dump(std::cout);

  tf.run_until(f4, [iter = 1] () mutable { std::cout << '\n'; return iter-- == 0; }, [](){ 
    std::cout << "First run_until finished\n"; 
  }).get();
  tf.run_until(f4, [iter = 2] () mutable { std::cout << '\n'; return iter-- == 0; }, [](){
    std::cout << "Second run_until finished\n"; 
  });
  tf.run_until(f4, [iter = 3] () mutable { std::cout << '\n'; return iter-- == 0; }, [](){
    std::cout << "Third run_until finished\n"; 
  }).get();
}

int main(){
  tf::Taskflow taskflow;
  composition_example_1(taskflow);
  composition_example_2(taskflow);
  return 0;
}



