// This program demonstrates how to create a pipeline scheduling framework
// that propagates a series of integers and adds one to the result at each
// stage.
//
// The pipeline has the following structure:
//
// o -> o -> o
// |         |
// v         v
// o -> o -> o
// |         |
// v         v
// o -> o -> o
// |         |
// v         v
// o -> o -> o

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>

int main() {

  tf::Taskflow taskflow("pipeline");
  tf::Executor executor;

  const size_t num_lines = 4;

  // custom data storage
  std::array<size_t, num_lines> buffer;

  // the pipeline consists of three pipes (serial-parallel-serial)
  // and up to four concurrent scheduling tokens
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, [&buffer](tf::Pipeflow& pf) {
      // generate only 5 scheduling tokens
      if(pf.token() == 5) {
        pf.stop();
      }
      // save the result of this pipe into the buffer
      else {
        printf("stage 1: input token = %zu\n", pf.token());
        buffer[pf.line()] = pf.token();
      }
    }},

    tf::Pipe{tf::PipeType::PARALLEL, [&buffer](tf::Pipeflow& pf) {
      printf(
        "stage 2: input buffer[%zu] = %zu\n", pf.line(), buffer[pf.line()]
      );
      // propagate the previous result to this pipe and increment
      // it by one
      buffer[pf.line()] = buffer[pf.line()] + 1;
    }},

    tf::Pipe{tf::PipeType::SERIAL, [&buffer](tf::Pipeflow& pf) {
      printf(
        "stage 3: input buffer[%zu] = %zu\n", pf.line(), buffer[pf.line()]
      );
      // propagate the previous result to this pipe and increment
      // it by one
      buffer[pf.line()] = buffer[pf.line()] + 1;
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
