#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>

// ----------------------------------------------------------------------------
// Scalable Pipeline
// ----------------------------------------------------------------------------

void scalable_pipeline(size_t num_lines, size_t num_pipes) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  size_t N = 0;

  std::vector< tf::Pipe<std::function<void(tf::Pipeflow&)>> > pipes;
  std::vector< int > data(num_lines, -1);

  for(size_t i=0; i<num_pipes; i++) {
    pipes.emplace_back(tf::PipeType::SERIAL, [&](tf::Pipeflow& pf) mutable { 

      switch(pf.pipe()) {
        case 0:
          if(pf.token() == 1111) {
            pf.stop();
            return;
          }
          data[pf.line()] = num_pipes * pf.token();
        break;

        default: {
          ++data[pf.line()];
        }
        break;
      }
      //printf("data[%zu]=%d\n", pf.line(), data[pf.line()]);
      REQUIRE(data[pf.line()] == (pf.token() * num_pipes + pf.pipe()));
      if(pf.pipe() == num_pipes - 1) {
        N++;
      }
    });
  }
  
  tf::ScalablePipeline spl(num_lines, pipes.begin(), pipes.end()); 
  taskflow.composed_of(spl);
  executor.run(taskflow).wait();  
  
  REQUIRE(N == 1111);
}

TEST_CASE("ScalablePipeline" * doctest::timeout(300)) {
  for(size_t L=1; L<=10; L++) {
    for(size_t P=1; P<=10; P++) {
      scalable_pipeline(L, P);
    }
  }
}

// ----------------------------------------------------------------------------
// Scalable Pipeline using Reset
// ----------------------------------------------------------------------------

void scalable_pipeline_reset(size_t num_lines, size_t num_pipes) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  size_t N = 0;

  std::vector< tf::Pipe<std::function<void(tf::Pipeflow&)>> > pipes;
  std::vector< int > data(num_lines, -1);
  
  tf::ScalablePipeline<typename decltype(pipes)::iterator> spl(num_lines); 

  auto init = taskflow.emplace([&](){
    for(size_t i=0; i<num_pipes; i++) {
      pipes.emplace_back(tf::PipeType::SERIAL, [&](tf::Pipeflow& pf) mutable { 

        switch(pf.pipe()) {
          case 0:
            if(pf.token() == 1111) {
              pf.stop();
              return;
            }
            data[pf.line()] = num_pipes * pf.token();
          break;

          default: {
            ++data[pf.line()];
          }
          break;
        }
        //printf("data[%zu]=%d\n", pf.line(), data[pf.line()]);
        REQUIRE(data[pf.line()] == (pf.token() * num_pipes + pf.pipe()));

        if(pf.pipe() == num_pipes - 1) {
          N++;
        }
      });
    }
    spl.reset(pipes.begin(), pipes.end());
  });
  
  auto pipeline = taskflow.composed_of(spl);
  pipeline.succeed(init);
  executor.run(taskflow).wait();  
  
  REQUIRE(N == 1111);
}

TEST_CASE("ScalablePipeline.Reset" * doctest::timeout(300)) {
  for(size_t L=1; L<=10; L++) {
    for(size_t P=1; P<=10; P++) {
      scalable_pipeline_reset(L, P);
    }
  }
}

// ----------------------------------------------------------------------------
// Scalable Pipeline using Iterative Reset
// ----------------------------------------------------------------------------

void scalable_pipeline_iterative_reset(size_t num_lines, size_t num_pipes) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  size_t N = 0;

  std::vector< tf::Pipe<std::function<void(tf::Pipeflow&)>> > pipes;
  std::vector< int > data(num_lines, -1);
  
  tf::ScalablePipeline<typename decltype(pipes)::iterator> spl(num_lines); 

  auto init = taskflow.emplace([&](){
    for(size_t i=0; i<num_pipes; i++) {
      pipes.emplace_back(tf::PipeType::SERIAL, [&](tf::Pipeflow& pf) mutable { 

        switch(pf.pipe()) {
          case 0:
            if(pf.token() == 1111) {
              pf.stop();
              return;
            }
            data[pf.line()] = num_pipes * pf.token();
          break;

          default: {
            ++data[pf.line()];
          }
          break;
        }
        //printf("data[%zu]=%d\n", pf.line(), data[pf.line()]);
        REQUIRE(data[pf.line()] == (pf.token() * num_pipes + pf.pipe()));

        if(pf.pipe() == num_pipes - 1) {
          N++;
        }
      });
    }
    spl.reset(pipes.begin(), pipes.end());
  });

  auto cond = taskflow.emplace([&, i=0]()mutable{
    REQUIRE(N == 1111*(i+1));
    spl.reset();
    return (i++ < 3) ? 0 : -1;
  });
  
  auto pipeline = taskflow.composed_of(spl);
  pipeline.succeed(init)
          .precede(cond);
  cond.precede(pipeline);
  executor.run(taskflow).wait();  
}

TEST_CASE("ScalablePipeline.IterativeReset" * doctest::timeout(300)) {
  for(size_t L=1; L<=10; L++) {
    for(size_t P=1; P<=10; P++) {
      scalable_pipeline_iterative_reset(L, P);
    }
  }
}









