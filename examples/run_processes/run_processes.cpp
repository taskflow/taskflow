// A simple example to run executables in subprocesses and maintain
// the following task dependencies.
//
//           +--------------------+
//     +---->| B run fibonacci 15 |-----+
//     |     +--------------------+     |
//   +--------------+           +-------v------+
//   | A run cancel |           | D run cancel |
//   +--------------+           +-------^------+
//     |     +-------------------+      |
//     +---->| C run fibonacci 21|------+
//           +-------------------+
//
#include <taskflow/taskflow.hpp>
#include "subprocess.hpp" // for blocking subprocess invocation
#include "path_to_executables.hpp"  // path to compiled examples fibonacci and simple



std::string fibonacciCall(int num)
{
    return std::string{pathToFibonacciExec()} + " " + std::to_string(num);
}

int main(){

    tf::Executor executor;
    tf::Taskflow taskflow("simple");

    auto [A, B, C, D] = taskflow.emplace(
        []() { std::cout << "TaskA\n"; runBlocking(pathToCancelExec()); },
        []() { std::cout << "TaskB\n"; runBlocking(fibonacciCall(15)); },
        []() { std::cout << "TaskC\n"; runBlocking(fibonacciCall(21)); },
        []() { std::cout << "TaskD\n"; runBlocking(pathToCancelExec());}
        );

    A.name("A");
    B.name("B");
    C.name("C");
    D.name("D");

    A.precede(B, C);  // A runs before B and C
    D.succeed(B, C);  // D runs after  B and C

    executor.run(taskflow).wait();

    return 0;
}

