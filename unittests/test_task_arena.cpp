#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/algorithm/lazy.hpp>
#include <chrono>
#include <thread>

TEST_CASE("ArenaCorrectness" * doctest::timeout(300)) {
    tf::Executor e;

    tf::Taskflow taskflow;
    std::shared_ptr<tf::TaskArena> arena = std::make_shared<tf::TaskArena>(e);

    thread_local bool is_outer_task_running = false;
    try
    {
        taskflow.for_each_index(0, 32, 1, [&](int i)
        {
            CHECK(is_outer_task_running == false);

            is_outer_task_running = true;
            e.isolate(arena, [&] {
                // Inside of this functor our worker cannot take tasks that are older than this task (e.g. from the outer for_each_index)
                // We can achieve this by switching current thread to another task queue
                tf::Taskflow child_taskflow;
                child_taskflow.for_each_index(0, 2, 1, [&](int j) {}); // These tasks will be spawned inside of our new TaskQueue of TaskArena "arena"
                e.corun(child_taskflow); // Guaranteed to never run tasks from the outer loop
            });
            is_outer_task_running = false;
        });
        CHECK(e.run(taskflow).wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    }
    catch (const std::exception&)
    {
        std::cout << "Exception occurred" << '\n';
    }
}


TEST_CASE("NestedArenas" * doctest::timeout(300)) {
    tf::Executor e;

    tf::Taskflow taskflow;
    std::shared_ptr<tf::TaskArena> arena = std::make_shared<tf::TaskArena>(e);
    std::shared_ptr<tf::TaskArena> second_arena = std::make_shared<tf::TaskArena>(e);

    thread_local bool is_outer_task_running = false;
    try
    {
        taskflow.for_each_index(0, 32, 1, [&](int i)
        {
            CHECK(is_outer_task_running == false);

            is_outer_task_running = true;
            e.isolate(arena, [&] {
                // Inside of this functor our worker cannot take tasks that are older than this task (e.g. from the outer for_each_index)
                // We can achieve this by switching current thread to another task queue
                tf::Taskflow child_taskflow;
                child_taskflow.for_each_index(0, 2, 1, [&](int j) { // These tasks will be spawned inside of our new TaskQueue of TaskArena "arena"
                    e.isolate(second_arena, [&] {
                        tf::Taskflow third_taskflow;
                        third_taskflow.for_each_index(0, 4, 1, [&](int k) {});
                        e.corun(third_taskflow);
                        });
                    });
                e.corun(child_taskflow); // Guaranteed to never run tasks from the outer loop
                });
            is_outer_task_running = false;
        });
        CHECK(e.run(taskflow).wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    }
    catch (const std::exception&)
    {
        std::cout << "Exception occurred" << '\n';
    }
}

TEST_CASE("LazyDeadlock" * doctest::timeout(300)) {
    
    using namespace std::chrono_literals;
    tf::Executor ex;
    for (size_t i = 0; i < 500; ++i)
    {
        // std::cout << "Iteration: " << i << std::endl;
        tf::Lazy<int> data(
            [&]()
            {
                tf::Taskflow taskflow2;
                for (size_t j = 0; j < 1; ++j)
                {
                    taskflow2.emplace([&] { std::this_thread::sleep_for(10ms); });
                }
                ex.this_worker_id() >= 0 ? ex.corun(taskflow2) : ex.run(taskflow2).get();
                return 99;
            },
            ex);


        tf::Taskflow taskflow1;
        for (size_t k = 0; k < 16; ++k)
        {
            taskflow1.emplace(
                [&]
                {
                    if (*data == 100)
                    {
                        std::cerr << "This can never happen" << std::endl;
                    }
                });
        }

        auto future = ex.run(taskflow1);
        CHECK(future.wait_for(5s) != std::future_status::timeout);
        future.get();
    }
}