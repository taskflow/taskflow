#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

// --------------------------------------------------------
// Testcase: ParallelSubflow
// --------------------------------------------------------

void parallel_subflow(size_t w) {
  
  size_t runtime_tasks_per_line = 20;
  size_t lines = 4;
  size_t subtasks = 1000;
  size_t subtask = 0;

  for (subtask = 0; subtask < subtasks; ++subtask) {
    tf::Executor executor(w);
    tf::Taskflow taskflow;

    auto& init = taskflow.emplace([](){}).name("init");

    auto& end = taskflow.emplace([](){}).name("end");
      
    std::vector<tf::Task> rts;
    std::atomic<size_t> sums = 0;
    
    for (size_t i = 0; i < runtime_tasks_per_line * lines; ++i) {
      std::string rt_name = "rt-" + std::to_string(i);
      
      rts.emplace_back(taskflow.emplace([&sums, &subtask](tf::Runtime& rt) mutable {
        rt.run([&sums, &subtask](tf::Subflow& sf) mutable {
          for (size_t j = 0; j < subtask; ++j) {
            sf.emplace([&sums]() mutable {
              ++sums;
              //std::this_thread::sleep_for(std::chrono::nanoseconds(1));
            });
          }  
        });
      }).name(rt_name));
    }

    for (size_t l = 0; l < lines; ++l) {
      init.precede(rts[l*runtime_tasks_per_line]);
    }

    for (size_t l = 0; l < lines; ++l) {
      for (size_t i = 0; i < runtime_tasks_per_line-1; ++i) {
        rts[i+l*runtime_tasks_per_line].precede(rts[i+l*runtime_tasks_per_line+1]);
      }
    }

    for (size_t l = 1; l < lines+1; ++l) {
      end.succeed(rts[runtime_tasks_per_line*l-1]);
    }

    executor.run(taskflow).wait();
    //taskflow.dump(std::cout);
    REQUIRE(sums == runtime_tasks_per_line*lines*subtask);
  }
}

TEST_CASE("ParallelSubflow.1thread" * doctest::timeout(300)){
  parallel_subflow(1);
}

TEST_CASE("ParallelSubflow.2threads" * doctest::timeout(300)){
  parallel_subflow(2);
}

TEST_CASE("ParallelSubflow.3threads" * doctest::timeout(300)){
  parallel_subflow(3);
}

TEST_CASE("ParallelSubflow.4threads" * doctest::timeout(300)){
  parallel_subflow(4);
}

TEST_CASE("ParallelSubflow.5threads" * doctest::timeout(300)){
  parallel_subflow(5);
}

TEST_CASE("ParallelSubflow.6threads" * doctest::timeout(300)){
  parallel_subflow(6);
}

TEST_CASE("ParallelSubflow.7threads" * doctest::timeout(300)){
  parallel_subflow(7);
}

TEST_CASE("ParallelSubflow.8threads" * doctest::timeout(300)){
  parallel_subflow(8);
}

