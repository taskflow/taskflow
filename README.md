# Cpp-Taskflow
An easy-to-use and fast header-only library written in C++17 for task-based parallel programming.

# Scale up Your Program with Cpp-Taskflow

The following example contains most of the syntax you need to use Cpp-Taskflow.

```cpp
#include <taskflow.hpp>

// A simple example to capture the following task dependencies.
//
// TaskA---->TaskB---->TaskD
// TaskA---->TaskC---->TaskD
//
int main(){
  
  tf::Taskflow<> tf(std::thread::hardware_concurrency());

  auto [A, B, C, D] = tf.silent_emplace(
    [] () { std::cout << "TaskA\n"; },
    [] () { std::cout << "TaskB\n"; },
    [] () { std::cout << "TaskC\n"; },
    [] () { std::cout << "TaskD\n"; }
  );  

  A.precede(B);
  A.precede(C);
  B.precede(D);
  C.precede(D);

  tf.wait_for_all(); 

  return 0;
}

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

Copyright © 2018, [Tsung-Wei Huang](http://web.engr.illinois.edu/~thuang19/) and Chun-Xun Lin.
Released under the [MIT license](https://github.com/twhuang-uiuc/cpp-taskflow/blob/master/LICENSE).
