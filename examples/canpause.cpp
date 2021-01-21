// A simple example to capture the following task dependencies. B can pause
//
//           +---+                 
//     +---->| B |-----+           
//     |     +---+     |           
//   +---+           +-v-+         
//   | A |           | D |         
//   +---+           +-^-+         
//     |     +---+     |           
//     +---->| C |-----+           
//           +---+
//
#include <taskflow/taskflow.hpp>  // the only include you need

#include <cassert>
#include <iostream>
#include <vector>



int main()
{

    tf::Executor executor(1);
    tf::Taskflow taskflow("canpause");

    int g_begin = 0;
    int g_end = 4;

    std::this_thread::sleep_for(std::chrono::seconds(3));

    auto [A, B, C, D] = taskflow.emplace(
        []() { std::cout << "TaskA\n"; },
        [&]() {
            std::cout << "TaskB\n";
            ++g_begin;
            if(g_begin==g_end)
            {
                return std::optional<bool>{false};
            }
            else if (g_begin < g_end)
            {
                return std::optional<bool>{std::nullopt};
            }
            return std::optional<bool>{true};
        },
        []() { std::cout << "TaskC\n"; },
            []() { std::cout << "TaskD\n"; }
        );
    A.name("A");
    B.name("B");
    C.name("C");
    D.name("D");
    A.precede(B,C);  // A runs before B 
    B.precede(D);
    C.precede(D);
    D.succeed(C);

    std::thread th([&]() {
        using namespace std;

        while (g_begin <= g_end)
        {
            this_thread::sleep_for(chrono::seconds(2));
            executor.resumeTask(&taskflow);
            if (g_begin >= g_end)
            {
                break;
            }
            
        }
        });

    executor.run_n(taskflow, 1).wait();

    th.join();

    return 0;
}
