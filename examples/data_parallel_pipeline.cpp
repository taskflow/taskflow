#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>

int main() {

  // data flow => void -> int -> std::string -> float -> void 
  tf::Taskflow taskflow("pipeline");
  tf::Executor executor;

  const size_t num_lines = 3;

  // custom data storage
  // std::array<int, num_lines> buffer;

  tf::DataPipeline pl(num_lines,
    tf::make_datapipe<void, int>(tf::PipeType::SERIAL, [&](tf::Pipeflow& pf) -> int{
      if(pf.token() == 5) {
        pf.stop();
      }
      else {
        return pf.token();
      }
    }),

    tf::make_datapipe<int, std::string>(tf::PipeType::SERIAL, [](int& input, tf::Pipeflow& pf) {
      // printf("%d, %d\n", pf.line(), pf.pipe());
      return std::to_string(input + 100);
    }),

    tf::make_datapipe<std::string, void>(tf::PipeType::SERIAL, [](std::string& input) {
      std::cout << input << std::endl;
      // return std::stoi(input) + rand()*0.5f;
    })

    // tf::DataPipe<float, void>{tf::PipeType::SERIAL, [](float input) {
    //   std::cout << "done with " << input << std::endl;
    // }}
  );

  // build the pipeline graph using composition
  taskflow.composed_of(pl).name("pipeline");

  // dump the pipeline graph structure (with composition)
  taskflow.dump(std::cout);

  // run the pipeline
  executor.run(taskflow).wait();

  return 0;
}
