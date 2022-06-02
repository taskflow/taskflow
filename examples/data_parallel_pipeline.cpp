#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>

int main() {

  // data flow => void -> int -> std::string -> float -> void 
  tf::Taskflow taskflow("pipeline");
  tf::Executor executor;

  const size_t num_lines = 4;

  // custom data storage
  //std::variant<int, std::string, float> mydata[num_lines];

  tf::DataPipeline pl(num_lines,
    tf::DataPipe<void, int>{tf::PipeType::SERIAL, [&](tf::Pipeflow& pf) -> int{
      if(pf.token() == 5) {
        pf.stop();
      }
      else {
        return rand();
      }
    }},

    tf::DataPipe<int, std::string>{tf::PipeType::SERIAL, [](int input) -> std::string {
      // ??? which line is this???
      return std::to_string(input + 100);
    }},

    tf::DataPipe<std::string, float>{tf::PipeType::SERIAL, [](std::string input) {
      return std::stoi(input) + rand()*0.5f;
    }},

    tf::DataPipe<float, void>{tf::PipeType::SERIAL, [](float input) {
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
