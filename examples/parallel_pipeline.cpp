// This program demonstrates how to create a pipeline scheduling framework.

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>

int main() {

  tf::Taskflow taskflow("pipeline");
  tf::Executor executor;

  const size_t num_lines = 4;
  const size_t num_pipes = 3;

  // custom dataflow storage
  std::array<std::array<int, num_pipes>, num_lines> mybuffer;

  // the pipeline consists of three pipes (serial-parallel-serial)
  // and up to four concurrent scheduling tokens
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, [&mybuffer](tf::Pipeflow& pf) {
      // generate only 5 scheduling tokens
      if(pf.token() == 5) {
        pf.stop();
      }
      // save the result of this pipe into the buffer
      else {
        printf("stage 1: input token = %zu\n", pf.token());
        mybuffer[pf.line()][pf.pipe()] = pf.token();
      }
    }},

    tf::Pipe{tf::PipeType::PARALLEL, [&mybuffer](tf::Pipeflow& pf) {
      printf(
        "stage 2: input mybuffer[%zu][%zu] = %d\n", 
        pf.line(), pf.pipe() - 1, mybuffer[pf.line()][pf.pipe() - 1]
      );
      // propagate the previous result to this pipe by adding one
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe()-1] + 1;
    }},

    tf::Pipe{tf::PipeType::SERIAL, [&mybuffer](tf::Pipeflow& pf) {
      printf(
        "stage 3: input mybuffer[%zu][%zu] = %d\n", 
        pf.line(), pf.pipe() - 1, mybuffer[pf.line()][pf.pipe() - 1]
      );
      // propagate the previous result to this pipe by adding one
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe()-1] + 1;
    }}
  );
  
  // build the pipeline graph using composition
  tf::Task init = taskflow.emplace([](){ std::cout << "ready\n"; })
                          .name("starting pipeline");
  tf::Task task = taskflow.composed_of(pl)
                          .name("pipeline");
  tf::Task stop = taskflow.emplace([](){ std::cout << "stopped\n"; })
                          .name("pipeline stopped");

  // create task dependency
  init.precede(task);
  task.precede(stop);
  
  // dump the pipeline graph structure (with composition)
  taskflow.dump(std::cout);

  // run the pipeline
  executor.run(taskflow).wait();
  
  return 0;
}
