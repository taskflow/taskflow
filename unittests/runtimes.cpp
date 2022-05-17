#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>

// --------------------------------------------------------
// Testcase 1: Runtime.Subflow
// --------------------------------------------------------

void runtime_subflow(size_t w) {
  
  size_t runtime_tasks_per_line = 200;
  size_t lines = 400;
  size_t subtasks = 4096;
  size_t subtask = 0;

  tf::Executor executor(w);
  tf::Taskflow parent;
  tf::Taskflow taskflow;

  for (subtask = 0; subtask <= subtasks; subtask = subtask == 0 ? subtask + 1 : subtask*2) {
    
    parent.clear();
    taskflow.clear();

    auto init = taskflow.emplace([](){}).name("init");
    auto end  = taskflow.emplace([](){}).name("end");

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

    parent.composed_of(taskflow);

    executor.run(parent).wait();
    //taskflow.dump(std::cout);
    REQUIRE(sums == runtime_tasks_per_line*lines*subtask);
  }
}

TEST_CASE("Runtime.Subflow.1thread" * doctest::timeout(300)){
  runtime_subflow(1);
}

TEST_CASE("Runtime.Subflow.2threads" * doctest::timeout(300)){
  runtime_subflow(2);
}

TEST_CASE("Runtime.Subflow.3threads" * doctest::timeout(300)){
  runtime_subflow(3);
}

TEST_CASE("Runtime.Subflow.4threads" * doctest::timeout(300)){
  runtime_subflow(4);
}

TEST_CASE("Runtime.Subflow.5threads" * doctest::timeout(300)){
  runtime_subflow(5);
}

TEST_CASE("Runtime.Subflow.6threads" * doctest::timeout(300)){
  runtime_subflow(6);
}

TEST_CASE("Runtime.Subflow.7threads" * doctest::timeout(300)){
  runtime_subflow(7);
}

TEST_CASE("Runtime.Subflow.8threads" * doctest::timeout(300)){
  runtime_subflow(8);
}


// --------------------------------------------------------
// Testcase 2: Pipeline.Runtime.Subflow
// --------------------------------------------------------

void pipeline_runtime_subflow(size_t w) {
  
  size_t num_lines = 10;
  size_t subtasks = 4096;
  size_t subtask = 0;
  size_t max_tokens = 10000000;

  tf::Executor executor(w);
  tf::Taskflow taskflow;
 
  for (subtask = 0; subtask <= subtasks; subtask = subtask == 0 ? subtask + 1 : subtask*2) {
   
    std::atomic<size_t> sums = 0;
    tf::Pipeline pl(
      num_lines, 
      tf::Pipe{
        tf::PipeType::SERIAL, [max_tokens](tf::Pipeflow& pf){
          if (pf.token() == max_tokens) {
            pf.stop();
          }
        }
      },

      tf::Pipe{
        tf::PipeType::PARALLEL, [subtask, &sums](tf::Pipeflow& pf, tf::Runtime& rt) mutable {
          rt.run([subtask, &sums](tf::Subflow& sf) mutable {
            for (size_t i = 0; i < subtask; ++i) {
              sf.emplace([&sums](){
                ++sums;  
              });
            }
          });
        }
      }
    );

    taskflow.composed_of(pl).name("pipeline");
    executor.run(taskflow).wait();
    REQUIRE(sums == subtask*max_tokens);
  }
}


TEST_CASE("Pipeline.Runtime.Subflow.1thread" * doctest::timeout(300)){
  pipeline_runtime_subflow(1);
}

TEST_CASE("Pipeline.Runtime.Subflow.2threads" * doctest::timeout(300)){
  pipeline_runtime_subflow(2);
}

TEST_CASE("Pipeline.Runtime.Subflow.3threads" * doctest::timeout(300)){
  pipeline_runtime_subflow(3);
}

TEST_CASE("Pipeline.Runtime.Subflow.4threads" * doctest::timeout(300)){
  pipeline_runtime_subflow(4);
}

TEST_CASE("Pipeline.Runtime.Subflow.5threads" * doctest::timeout(300)){
  pipeline_runtime_subflow(5);
}

TEST_CASE("Pipeline.Runtime.Subflow.6threads" * doctest::timeout(300)){
  pipeline_runtime_subflow(6);
}

TEST_CASE("Pipeline.Runtime.Subflow.7threads" * doctest::timeout(300)){
  pipeline_runtime_subflow(7);
}

TEST_CASE("Pipeline.Runtime.Subflow.8threads" * doctest::timeout(300)){
  pipeline_runtime_subflow(8);
}


// --------------------------------------------------------
// Testcase 3: Pipeline.Runtime.Subflow
// --------------------------------------------------------
/*
void pipeline_runtime_subflow(size_t w) {
 
  tf::Executor executor(w);
  
  size_t num_taskflows = 4; 
  std::vector<tf::Taskflow> tfs;

  for (size_t i = 0; i < num_taskflows; ++i) {
    tfs.emplace_back(tf::Taskflow);
  }


  tfs[0].emplace([](tf::Runtime& rt){ rt.run() });


  tf::Taskflow taskflow;
 
  for (subtask = 0; subtask <= subtasks; subtask = subtask == 0 ? subtask + 1 : subtask*2) {
   
    std::atomic<size_t> sums = 0;
    tf::Pipeline pl(
      num_lines, 
      tf::Pipe{
        tf::PipeType::SERIAL, [max_tokens](tf::Pipeflow& pf){
          if (pf.token() == max_tokens) {
            pf.stop();
          }
        }
      },

      tf::Pipe{
        tf::PipeType::PARALLEL, [subtask, &sums](tf::Pipeflow& pf, tf::Runtime& rt) mutable {
          rt.run([subtask, &sums](tf::Subflow& sf) mutable {
            for (size_t i = 0; i < subtask; ++i) {
              sf.emplace([&sums](){
                ++sums;  
              });
            }
          });
        }
      }
    );

    taskflow.composed_of(pl).name("pipeline");
    executor.run(taskflow).wait();
    REQUIRE(sums == subtask*max_tokens);
  }
}


TEST_CASE("Pipeline.Runtime.Subflow.1thread" * doctest::timeout(300)){
  pipeline_runtime_subflow(1);
}

TEST_CASE("Pipeline.Runtime.Subflow.2threads" * doctest::timeout(300)){
  pipeline_runtime_subflow(2);
}

TEST_CASE("Pipeline.Runtime.Subflow.3threads" * doctest::timeout(300)){
  pipeline_runtime_subflow(3);
}

TEST_CASE("Pipeline.Runtime.Subflow.4threads" * doctest::timeout(300)){
  pipeline_runtime_subflow(4);
}

TEST_CASE("Pipeline.Runtime.Subflow.5threads" * doctest::timeout(300)){
  pipeline_runtime_subflow(5);
}

TEST_CASE("Pipeline.Runtime.Subflow.6threads" * doctest::timeout(300)){
  pipeline_runtime_subflow(6);
}

TEST_CASE("Pipeline.Runtime.Subflow.7threads" * doctest::timeout(300)){
  pipeline_runtime_subflow(7);
}

TEST_CASE("Pipeline.Runtime.Subflow.8threads" * doctest::timeout(300)){
  pipeline_runtime_subflow(8);
}
*/

