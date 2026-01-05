#include <print>

import tf;

using tf::Executor;
using tf::Taskflow;

int main() {
    Executor executor;
    Taskflow taskflow;

    auto [A, B, C, D] = taskflow.emplace(
        []() -> void { std::println("TaskA"); },
        []() -> void { std::println("TaskB"); },
        []() -> void { std::println("TaskC"); },
        []() -> void { std::println("TaskD"); } 
    );                                  

    A.precede(B, C);  // A runs before B and C
    D.succeed(B, C);  // D runs after B and C

    executor.run(taskflow).wait(); 

    return 0;
}
