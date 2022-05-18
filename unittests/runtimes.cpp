#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>

// --------------------------------------------------------
// Testcase 1: Runtime.Subflow
// --------------------------------------------------------

void runtime_subflow(size_t w) {
  
  size_t runtime_tasks_per_line = 20;
  size_t lines = 4;
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
// Testcase 2: Pipeline(SP).Runtime.Subflow
// --------------------------------------------------------

void pipeline_sp_runtime_subflow(size_t w) {
  
  size_t num_lines = 2;
  size_t subtasks = 2;
  size_t subtask = 2;
  size_t max_tokens = 10000000;

  tf::Executor executor(w);
  tf::Taskflow taskflow;
 
  //for (subtask = 0; subtask <= subtasks; subtask = subtask == 0 ? subtask + 1 : subtask*2) {
   
    std::atomic<size_t> sums = 0;
    tf::Pipeline pl(
      num_lines, 
      tf::Pipe{
        tf::PipeType::SERIAL, [max_tokens](tf::Pipeflow& pf){
          //std::cout << tf::stringify(pf.token(), '\n');
          if (pf.token() == max_tokens) {
            pf.stop();
          }
        }
      },

      tf::Pipe{
        tf::PipeType::PARALLEL, [subtask, &sums](tf::Pipeflow&, tf::Runtime& rt) mutable {
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
  //}
}


TEST_CASE("Pipeline(SP).Runtime.Subflow.1thread" * doctest::timeout(300)){
  pipeline_sp_runtime_subflow(1);
}

TEST_CASE("Pipeline(SP).Runtime.Subflow.2threads" * doctest::timeout(300)){
  pipeline_sp_runtime_subflow(2);
}

TEST_CASE("Pipeline(SP).Runtime.Subflow.3threads" * doctest::timeout(300)){
  pipeline_sp_runtime_subflow(3);
}

TEST_CASE("Pipeline(SP).Runtime.Subflow.4threads" * doctest::timeout(300)){
  pipeline_sp_runtime_subflow(4);
}

TEST_CASE("Pipeline(SP).Runtime.Subflow.5threads" * doctest::timeout(300)){
  pipeline_sp_runtime_subflow(5);
}

TEST_CASE("Pipeline(SP).Runtime.Subflow.6threads" * doctest::timeout(300)){
  pipeline_sp_runtime_subflow(6);
}

TEST_CASE("Pipeline(SP).Runtime.Subflow.7threads" * doctest::timeout(300)){
  pipeline_sp_runtime_subflow(7);
}

TEST_CASE("Pipeline(SP).Runtime.Subflow.8threads" * doctest::timeout(300)){
  pipeline_sp_runtime_subflow(8);
}


// --------------------------------------------------------
// Testcase 3: Pipeline(SPSPSPSP).Runtime.Subflow
// --------------------------------------------------------

void pipeline_spspspsp_runtime_subflow(size_t w) {
  
  size_t num_lines = 4;
  size_t subtasks = 128;
  size_t subtask = 2;
  size_t max_tokens = 100000;

  tf::Executor executor(w);
  tf::Taskflow taskflow;
 
  for (subtask = 0; subtask <= subtasks; subtask = subtask == 0 ? subtask + 1 : subtask*2) {
   
    taskflow.clear();
    
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
        tf::PipeType::PARALLEL, [subtask, &sums](tf::Pipeflow&, tf::Runtime& rt) mutable {
          rt.run([subtask, &sums](tf::Subflow& sf) mutable {
            for (size_t i = 0; i < subtask; ++i) {
              sf.emplace([&sums](){
                ++sums;  
              });
            }
          });
        }
      },

      tf::Pipe{
        tf::PipeType::SERIAL, [subtask, &sums](tf::Pipeflow&, tf::Runtime& rt) mutable {
          rt.run([subtask, &sums](tf::Subflow& sf) mutable {
            for (size_t i = 0; i < subtask; ++i) {
              sf.emplace([&sums](){
                ++sums;  
              });
            }
          });
        }
      },

      tf::Pipe{
        tf::PipeType::PARALLEL, [subtask, &sums](tf::Pipeflow&, tf::Runtime& rt) mutable {
          rt.run([subtask, &sums](tf::Subflow& sf) mutable {
            for (size_t i = 0; i < subtask; ++i) {
              sf.emplace([&sums](){
                ++sums;  
              });
            }
          });
        }
      },

      tf::Pipe{
        tf::PipeType::SERIAL, [subtask, &sums](tf::Pipeflow&, tf::Runtime& rt) mutable {
          rt.run([subtask, &sums](tf::Subflow& sf) mutable {
            for (size_t i = 0; i < subtask; ++i) {
              sf.emplace([&sums](){
                ++sums;  
              });
            }
          });
        }
      },

      tf::Pipe{
        tf::PipeType::PARALLEL, [subtask, &sums](tf::Pipeflow&, tf::Runtime& rt) mutable {
          rt.run([subtask, &sums](tf::Subflow& sf) mutable {
            for (size_t i = 0; i < subtask; ++i) {
              sf.emplace([&sums](){
                ++sums;  
              });
            }
          });
        }
      },

      tf::Pipe{
        tf::PipeType::SERIAL, [subtask, &sums](tf::Pipeflow&, tf::Runtime& rt) mutable {
          rt.run([subtask, &sums](tf::Subflow& sf) mutable {
            for (size_t i = 0; i < subtask; ++i) {
              sf.emplace([&sums](){
                ++sums;  
              });
            }
          });
        }
      },

      tf::Pipe{
        tf::PipeType::PARALLEL, [subtask, &sums](tf::Pipeflow&, tf::Runtime& rt) mutable {
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
    REQUIRE(sums == subtask*max_tokens*7);
  }
}

/*
TEST_CASE("Pipeline(SPSPSPSP).Runtime.Subflow.1thread" * doctest::timeout(300)){
  pipeline_spspspsp_runtime_subflow(1);
}

TEST_CASE("Pipeline(SPSPSPSP).Runtime.Subflow.2threads" * doctest::timeout(300)){
  pipeline_spspspsp_runtime_subflow(2);
}

TEST_CASE("Pipeline(SPSPSPSP).Runtime.Subflow.3threads" * doctest::timeout(300)){
  pipeline_spspspsp_runtime_subflow(3);
}

TEST_CASE("Pipeline(SPSPSPSP).Runtime.Subflow.4threads" * doctest::timeout(300)){
  pipeline_spspspsp_runtime_subflow(4);
}

TEST_CASE("Pipeline(SPSPSPSP).Runtime.Subflow.5threads" * doctest::timeout(300)){
  pipeline_spspspsp_runtime_subflow(5);
}

TEST_CASE("Pipeline(SPSPSPSP).Runtime.Subflow.6threads" * doctest::timeout(300)){
  pipeline_spspspsp_runtime_subflow(6);
}

TEST_CASE("Pipeline(SPSPSPSP).Runtime.Subflow.7threads" * doctest::timeout(300)){
  pipeline_spspspsp_runtime_subflow(7);
}

TEST_CASE("Pipeline(SPSPSPSP).Runtime.Subflow.8threads" * doctest::timeout(300)){
  pipeline_spspspsp_runtime_subflow(8);
}
*/

// --------------------------------------------------------
// Testcase 4: Pipeline(SPSPSPSP).Runtime.IrregularSubflow
// --------------------------------------------------------

void pipeline_spspspsp_runtime_irregular_subflow(size_t w) {
  
  size_t num_lines = 4;
  size_t max_tokens = 10000000;

  tf::Executor executor(w);
  tf::Taskflow taskflow;
 
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

    /* subflow has the following dependency
     *    
     *     |--> B
     *  A--|
     *     |--> C
     */
    tf::Pipe{
      tf::PipeType::PARALLEL, [&sums](tf::Pipeflow&, tf::Runtime& rt) mutable {
        rt.run([&sums](tf::Subflow& sf) mutable {
          auto A = sf.emplace([&sums]() mutable { ++sums; });
          auto B = sf.emplace([&sums]() mutable { ++sums; });
          auto C = sf.emplace([&sums]() mutable { ++sums; });
          A.precede(B, C);
        });
      }
    },

    /* subflow has the following dependency
     *
     *     |--> B--| 
     *     |       v
     *  A--|       D
     *     |       ^
     *     |--> C--|
     *
     */
    tf::Pipe{
      tf::PipeType::SERIAL, [&sums](tf::Pipeflow&, tf::Runtime& rt) mutable {
        rt.run([&sums](tf::Subflow& sf) mutable {
          auto A = sf.emplace([&sums]() mutable { ++sums; });
          auto B = sf.emplace([&sums]() mutable { ++sums; });
          auto C = sf.emplace([&sums]() mutable { ++sums; });
          auto D = sf.emplace([&sums]() mutable { ++sums; });
          A.precede(B, C);
          D.succeed(B, C);
        });
      }
    },

    /* subflow has the following dependency
     *
     *       |--> C 
     *       |       
     *  A--> B       
     *       |       
     *       |--> D 
     *
     */
    tf::Pipe{
      tf::PipeType::PARALLEL, [&sums](tf::Pipeflow&, tf::Runtime& rt) mutable {
        rt.run([&sums](tf::Subflow& sf) mutable {
          auto A = sf.emplace([&sums]() mutable { ++sums; });
          auto B = sf.emplace([&sums]() mutable { ++sums; });
          auto C = sf.emplace([&sums]() mutable { ++sums; });
          auto D = sf.emplace([&sums]() mutable { ++sums; });
          A.precede(B);
          B.precede(C, D);
        });
      }
    },

    /* subflow has the following dependency
     *
     *     |--> B--|   |--> E
     *     |       v   |
     *  A--|       D --| 
     *     |       ^   |
     *     |--> C--|   |--> F
     *
     */
    tf::Pipe{
      tf::PipeType::SERIAL, [&sums](tf::Pipeflow&, tf::Runtime& rt) mutable {
        rt.run([&sums](tf::Subflow& sf) mutable {
          auto A = sf.emplace([&sums]() mutable { ++sums; });
          auto B = sf.emplace([&sums]() mutable { ++sums; });
          auto C = sf.emplace([&sums]() mutable { ++sums; });
          auto D = sf.emplace([&sums]() mutable { ++sums; });
          auto E = sf.emplace([&sums]() mutable { ++sums; });
          auto F = sf.emplace([&sums]() mutable { ++sums; });
          A.precede(B, C);
          D.succeed(B, C);
          D.precede(E, F);
        });
      }
    },

    /* subflow has the following dependency
     *
     *  A --> B --> C --> D -->  E
     *
     */
    tf::Pipe{
      tf::PipeType::PARALLEL, [&sums](tf::Pipeflow&, tf::Runtime& rt) mutable {
        rt.run([&sums](tf::Subflow& sf) mutable {
          auto A = sf.emplace([&sums]() mutable { ++sums; });
          auto B = sf.emplace([&sums]() mutable { ++sums; });
          auto C = sf.emplace([&sums]() mutable { ++sums; });
          auto D = sf.emplace([&sums]() mutable { ++sums; });
          auto E = sf.emplace([&sums]() mutable { ++sums; });
          A.precede(B);
          B.precede(C);
          C.precede(D);
          D.precede(E);
        });
      }
    },

    /* subflow has the following dependency
     *    
     *        |-----------|
     *        |           v
     *  A --> B --> C --> D -->  E
     *              |            ^
     *              |------------|
     *
     */
    tf::Pipe{
      tf::PipeType::SERIAL, [&sums](tf::Pipeflow&, tf::Runtime& rt) mutable {
        rt.run([&sums](tf::Subflow& sf) mutable {
          auto A = sf.emplace([&sums]() mutable { ++sums; });
          auto B = sf.emplace([&sums]() mutable { ++sums; });
          auto C = sf.emplace([&sums]() mutable { ++sums; });
          auto D = sf.emplace([&sums]() mutable { ++sums; });
          auto E = sf.emplace([&sums]() mutable { ++sums; });
          A.precede(B);
          B.precede(C, D);
          C.precede(D, E);
          D.precede(E);
        });
      }
    },

    /* subflow has the following dependency
     *    
     *  |-----------|
     *  |           v
     *  A --> B --> C --> D 
     *  |                 ^
     *  |-----------------|
     *
     */
    tf::Pipe{
      tf::PipeType::PARALLEL, [&sums](tf::Pipeflow&, tf::Runtime& rt) mutable {
        rt.run([&sums](tf::Subflow& sf) mutable {
          auto A = sf.emplace([&sums]() mutable { ++sums; });
          auto B = sf.emplace([&sums]() mutable { ++sums; });
          auto C = sf.emplace([&sums]() mutable { ++sums; });
          auto D = sf.emplace([&sums]() mutable { ++sums; });
          A.precede(B, C, D);
          B.precede(C);
          C.precede(D);
        });
      }
    }
  );

  taskflow.composed_of(pl).name("pipeline");
  executor.run(taskflow).wait();

  //taskflow.dump(std::cout);
  // there are 31 spawned subtasks in total
  REQUIRE(sums == 31*max_tokens);
}

/*
TEST_CASE("Pipeline(SPSPSPSP).Runtime.Irregular.Subflow.1thread" * doctest::timeout(300)){
  pipeline_spspspsp_runtime_irregular_subflow(1);
}

TEST_CASE("Pipeline(SPSPSPSP).Runtime.Irregular.Subflow.2threads" * doctest::timeout(300)){
  pipeline_spspspsp_runtime_irregular_subflow(2);
}

TEST_CASE("Pipeline(SPSPSPSP).Runtime.Irregular.Subflow.3threads" * doctest::timeout(300)){
  pipeline_spspspsp_runtime_irregular_subflow(3);
}

TEST_CASE("Pipeline(SPSPSPSP).Runtime.Irregular.Subflow.4threads" * doctest::timeout(300)){
  pipeline_spspspsp_runtime_irregular_subflow(4);
}

TEST_CASE("Pipeline(SPSPSPSP).Runtime.Irregular.Subflow.5threads" * doctest::timeout(300)){
  pipeline_spspspsp_runtime_irregular_subflow(5);
}

TEST_CASE("Pipeline(SPSPSPSP).Runtime.Irregular.Subflow.6threads" * doctest::timeout(300)){
  pipeline_spspspsp_runtime_irregular_subflow(6);
}

TEST_CASE("Pipeline(SPSPSPSP).Runtime.Irregular.Subflow.7threads" * doctest::timeout(300)){
  pipeline_spspspsp_runtime_irregular_subflow(7);
}

TEST_CASE("Pipeline(SPSPSPSP).Runtime.Irregular.Subflow.8threads" * doctest::timeout(300)){
  pipeline_spspspsp_runtime_irregular_subflow(8);
}
*/
