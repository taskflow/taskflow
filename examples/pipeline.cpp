#include <taskflow/taskflow.hpp>
#include <taskflow/pipeline.hpp>

int main()
{

  tf::Taskflow taskflow;
  tf::Executor executor;

  // std::variant as the data type
  using data_type = std::variant<std::monostate, int, float>;

  auto pl = tf::make_pipeline<data_type, 4>(
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

  //
  // std::any as the data type
  //auto pl = make_pipeline<std::any>(4,
  //  Filter{FilterType::SERIAL, [i=0](auto& df) mutable {
  //    if(i++==5) {
  //      df.stop();
  //    }
  //    else {
  //      df.at_output() = -11;
  //    }
  //  }},
  //  Filter{FilterType::PARALLEL, [](auto& df){
  //    std::cout << "second stage input = " << std::any_cast<int>(df.at_input()) << std::endl;
  //    df.at_output() = 1.2f;
  //  }},
  //  Filter{FilterType::SERIAL, [](auto& df){
  //    std::cout << "third stage input = " << std::any_cast<float>(df.at_input()) << std::endl;
  //  }}
  //);
  
  //// int as the data type
  //auto pl = make_pipeline<int>(4,
  //  Filter{FilterType::SERIAL, [i=3] (auto& df) mutable {
  //    if(i-- == 0) {
  //      df.stop();
  //    }
  //    else {
  //      printf("stage 1 input = %d\n", df.input());
  //      df.at_output() = -11;
  //    }
  //  }},
  //  Filter{FilterType::PARALLEL, [] (auto& df) {
  //    printf("stage 2 input = %d\n", df.at_input());
  //    df.at_output() = 7;
  //  }},
  //  Filter{FilterType::SERIAL, [] (auto& df) {
  //    printf("stage 3 input = %d\n", df.at_input());
  //  }}
  //);

  // Pipeline pl(4,
  //   make_filter<void, int>(FilterType::SERIAL, [i=0](auto&& df) mutable {
  //     std::cout << "first stage " << i << std::endl;
  //     if(i++ == 5) {
  //       df.stop();
  //     }
  //     else {
  //       df.output = -11;
  //     }
  //   }),
  //   make_filter<int, float> (FilterType::PARALLEL, [](auto& df){ 
  //     df.output = 1.2f;
  //     std::cout << "second stage\n";
  //   }),
  //   make_filter<float, void>(FilterType::SERIAL,   [](auto&){  
  //     std::cout << "third stage\n"; 
  //   })
  // );
   
  //tf::Executor executor;
  //tf::Taskflow taskflow;
  //pl.make_taskflow(taskflow);
  //executor.run(taskflow).wait();

  return 0;
}












