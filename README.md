# Cpp-Taskflow
A fast header-only library for task-based parallel programming.

# Get Started with Cpp-Taskflow

The following example (`simple.cpp`) contains the basic syntax you need to use Cpp-Taskflow.

```cpp
// A simple example to capture the following task dependencies.
//
// TaskA---->TaskB---->TaskD
// TaskA---->TaskC---->TaskD

#include "taskflow.hpp"

int main(){
  
  tf::Taskflow tf(std::thread::hardware_concurrency());

  auto [A, B, C, D] = tf.silent_emplace(
    [] () { std::cout << "TaskA\n"; },
    [] () { std::cout << "TaskB\n"; },
    [] () { std::cout << "TaskC\n"; },
    [] () { std::cout << "TaskD\n"; }
  );  

  A.precede(B);  // B runs after A
  A.precede(C);  // C runs after A
  B.precede(D);  // D runs after B
  C.precede(D);  // C runs after D

  tf.wait_for_all();  // block until all tasks finish

  return 0;
}

```
Compile and run the code with the following commands:
```bash
~$ g++ simple.cpp -std=c++1z -O2 -lpthread -o simple
~$ ./simple
TaskA
TaskC  <-- concurrent with TaskB
TaskB  <-- concurrent with TaskC
TaskD
```

# System Requirements
To use Cpp-Taskflow, you only need a C++17 compiler:
- GNU C++ Compiler G++ v7.2 with -std=c++1z

# Get Involved
+ Report bugs/issues by submitting a <a href="https://github.com/twhuang-uiuc/cpp-taskflow/issues">GitHub issue</a>.
+ Submit contributions using <a href="https://github.com/twhuang-uiuc/cpp-taskflow/pulls">pull requests<a>.

# Contributors
+ Tsung-Wei Huang
+ Chun-Xun Lin
+ You!

# License

Copyright © 2018, [Tsung-Wei Huang](http://web.engr.illinois.edu/~thuang19/) and [Chun-Xun Lin](https://github.com/clin99).
Released under the [MIT license](https://github.com/twhuang-uiuc/cpp-taskflow/blob/master/LICENSE).

