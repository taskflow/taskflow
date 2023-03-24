// This program demonstrates how to create a pipeline scheduling framework
// that describes a generalized token dependencies,
// propagates a series of integers, and adds one to the result at each
// stage, using a range of pipes provided by the application.
//
// The pipeline has the following structure:
//
// o -> o -> o
// |    |    |
// v    v    v
// o -> o -> o
// |    |    |
// v    v    v
// o -> o -> o
// |    |    |
// v    v    v
// o -> o -> o
//
// Then, the program resets the pipeline to a new range of five pipes.
//
// o -> o -> o -> o -> o
// |    |    |    |    |
// v    v    v    v    v
// o -> o -> o -> o -> o
// |    |    |    |    |
// v    v    v    v    v
// o -> o -> o -> o -> o
// |    |    |    |    |
// v    v    v    v    v
// o -> o -> o -> o -> o




// The scheduling token has the following dependencies:
//    ___________
//   |           |
//   V _____     |
//   |     |     | 
//   |     V     | 
// 1 2 3 4 5 6 7 8 9 10 
//         ^   |   |
//         |___|   |
//         ^       | 
//         |_______|
//
// 2 is deferred by 8
// 5 is dieferred by 2, 7, and 9

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>

int main() {

  tf::Taskflow taskflow("pipeline");
  tf::Executor executor;

  const size_t num_lines = 4;

  // define the pipe callable
  auto pipe_callable = [] (tf::Pipeflow& pf) mutable {
    switch(pf.pipe()) {
      // first stage generates only 15 scheduling tokens
      // and describes the token dependencies
      case 0: {
        if(pf.token() == 15) {
          pf.stop();
        }
        else {
          if (pf.token() == 5) {
            switch(pf.num_deferrals()) {
              case 0:
                pf.defer(2);
                printf("1st-time: Token %zu is deferred by 2\n", pf.token());
                pf.defer(7);
                printf("1st-time: Token %zu is deferred by 7\n", pf.token());
              break;

              case 1:
                pf.defer(9);
                printf("snd-time: Token %zu is deferred by 9\n", pf.token());
              break;

              case 2:
                printf("3rd-time: Tokens 2, 7 and 9 resolved dependencies for token %zu\n", pf.token());
              break;
            }
          }

          else if (pf.token() == 2) {
            switch(pf.num_deferrals()) {
              case 0:
                pf.defer(8);
                printf("1st-time: Token %zu is deferred by 8\n", pf.token());
              break;
              
              case 1:
                printf("2nd-time: Token 8 resolved dependencies for token %zu\n", pf.token());
              break;
            }
          }
          
          else {
            printf("stage 1: Non-deferred token %zu\n", pf.token());
          }
        }
      }
      break;

      // other stages prints input token 
      default: {
        printf("stage %zu : input token %zu (deferrals=%zu)\n",
               pf.pipe()+1, pf.token(), pf.num_deferrals());
      }
      break;
    }
  };

  // create a vector of three pipes
  std::vector< tf::Pipe<std::function<void(tf::Pipeflow&)>> > pipes;

  for(size_t i=0; i<3; i++) {
    pipes.emplace_back(tf::PipeType::SERIAL, pipe_callable);
  }

  // create a pipeline of four parallel lines using the given vector of pipes
  tf::ScalablePipeline<decltype(pipes)::iterator> pl(num_lines, pipes.begin(), pipes.end());

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

  // reset the pipeline to a new range of five pipes and starts from
  // the initial state (i.e., token counts from zero)
  for(size_t i=0; i<2; i++) {
    pipes.emplace_back(tf::PipeType::SERIAL, pipe_callable);
  }
  pl.reset(pipes.begin(), pipes.end());

  executor.run(taskflow).wait();

  return 0;
}



