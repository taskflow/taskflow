#include <taskflow/taskflow.hpp>
#include <taskflow/pipeline.hpp>

int main()
{
  tf::Taskflow taskflow;
  tf::Executor executor;

  // std::variant as the data type
  using data_type = std::variant<std::monostate, int, float>;

  auto pl = tf::make_pipeline<data_type>(4,
    tf::Filter{tf::FilterType::SERIAL, [i=0](auto& df) mutable {
      if(i++==5) {
        df.stop();
      }
      else {
        df.at_output() = -11;
      }
    }},
    tf::Filter{tf::FilterType::PARALLEL, [](auto& df){
      std::cout << "second stage input = " << std::get<int>(df.at_input()) << std::endl;
      df.at_output() = 1.2f;
    }},
    tf::Filter{tf::FilterType::SERIAL, [](auto& df){
      std::cout << "third stage input = " << std::get<float>(df.at_input()) << std::endl;
    }}
  );

  taskflow.pipeline(pl);
  taskflow.dump(std::cout);

  executor.run(taskflow).wait();

  return 0;
}












