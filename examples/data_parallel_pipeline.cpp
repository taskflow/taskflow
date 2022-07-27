#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/data_pipeline.hpp>

int main() {

  // data flow => void -> int -> std::string -> float -> void 
  tf::Taskflow taskflow("pipeline");
  tf::Executor executor;

  const size_t num_lines = 3;

  tf::DataPipeline pl(num_lines,
    tf::make_datapipe<void, int>(tf::PipeType::SERIAL, [&](tf::Pipeflow& pf) -> int{
      if(pf.token() == 5) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),

    tf::make_datapipe<int, std::string>(tf::PipeType::SERIAL, [](int& input, tf::Pipeflow& pf) {
      return std::to_string(input + 100);
    }),

    tf::make_datapipe<std::string, void>(tf::PipeType::SERIAL, [](std::string& input) {
      std::cout << input << std::endl;
    })
  );

  // build the pipeline graph using composition
  taskflow.composed_of(pl).name("pipeline");

  // dump the pipeline graph structure (with composition)
  taskflow.dump(std::cout);

  // run the pipeline
  executor.run(taskflow).wait();

  return 0;
}
