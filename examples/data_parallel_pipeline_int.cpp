#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>

int main() {

  // data flow => void -> int -> std::string -> float -> void 
  tf::Taskflow taskflow("pipeline");
  tf::Executor executor;

  // const size_t num_lines = 4;

  // custom data storage
  // std::array<int, num_lines> buffer;

  tf::DataPipeline pl(4,
    tf::DataPipe<tf::Pipeflow&, int>{tf::PipeType::SERIAL, [](tf::Pipeflow& pf) -> int{
      if(pf.token() == 5) {
        pf.stop();
      }
      else {
        return pf.token();
      }
    }},

    tf::DataPipe<int, int>{tf::PipeType::SERIAL, [](int input) -> int {
      return input + 100;
    }},

    tf::DataPipe<int, int>{tf::PipeType::SERIAL, [](int input) -> int {
      return input + 100;
    }},

    tf::DataPipe<int, void>{tf::PipeType::SERIAL, [](int input) -> void {
      std::cout << "done with " << input << std::endl;
    }}
  );

  // build the pipeline graph using composition
  taskflow.composed_of(pl).name("pipeline");

  // dump the pipeline graph structure (with composition)
  taskflow.dump(std::cout);

  // run the pipeline
  executor.run(taskflow).wait();

  return 0;
}