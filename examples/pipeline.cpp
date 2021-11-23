#include <taskflow/taskflow.hpp>
#include <taskflow/pipeline.hpp>

// TODO: you can use printf to avoid interleaving strings

int main() {

  tf::Taskflow taskflow;
  tf::Executor executor;

  const size_t num_lines = 4;
  const size_t num_pipes = 2;

  // user-defined buffer to hold the data
  std::array<std::array<int, num_pipes>, num_lines> mybuffer;

  // the pipeline is consisted of three pipes, 
  // first is a serial pipe, second is a parallel pipe, the third is a serial pipe
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, [i = 0, &mybuffer](auto& pf) mutable {
      printf("first stage i = %d\n", i);
      if(i++ == 5) {
        pf.stop();
      }
      else {
        // save the result of this pipe into the buffer of next pipe
        mybuffer[pf.line()][pf.pipe()] = -11;
      }
    }},

    tf::Pipe{tf::PipeType::PARALLEL, [&mybuffer](auto& pf){
      printf(
        "second stage input mybuffer[%zu][%zu] = %d\n", 
        pf.line(), pf.pipe(), mybuffer[pf.line()][pf.pipe() - 1]
      );
      
      // save the result of this pipe into the buffer of next pipe
      mybuffer[pf.line()][pf.pipe()] = 2;

    }},

    tf::Pipe{tf::PipeType::SERIAL, [&mybuffer](auto& pf){
      printf(
        "third  stage input mybuffer[%zu][%zu] = %d\n", 
        pf.line(), pf.pipe(), mybuffer[pf.line()][pf.pipe() - 1]
      );
    }}
  );

  taskflow.pipeline(pl);
  //taskflow.dump(std::cout);
  executor.run(taskflow).wait();

  //executor.run(taskflow).wait();
  return 0;
}












