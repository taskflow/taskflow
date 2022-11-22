#include "deferred_pipeline.hpp"
#include <CLI11.hpp>
#include <cstdlib>
#include <cassert>

int main(int argc, char* argv[]) {

  CLI::App app{"Parallel Deferred Pipeline"};

  unsigned num_threads {1};
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned num_rounds {1};
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");

  std::string model = "tf";
  app.add_option("-m,--model", model, "model name pthread|tf (default=tf)")
    ->check([] (const std::string& m) {
      if(m != "pthread" && m != "tf") {
        return "model name should be \"pthread\" or \"tf\"";
      }
      return "";
    });

  std::string vtype = "1";
  app.add_option("-v, --vtype", vtype, "x264 video types (default=1). Check deferred_pipeline.hpp for detailed frame patterns.")
    ->check([] (const std::string& v) {
      if(v != "1" && v != "2") {
        return "video types should be \"1\" or \"2\"";
      }
      return "";
    });

   
  CLI11_PARSE(app, argc, argv);
  
  std::cout << "model="       << model       << ' '
            << "num_threads=" << num_threads << ' '
            << "num_rounds="  << num_rounds  << ' '
            << "video_type="  << vtype        << ' '
            << std::endl;

  std::cout << std::setw(12) << "Size"
            << std::setw(12) << "Runtime"
            << '\n';

  size_t shift = 1;
  size_t max_shift = 21;

  for(size_t i = (1 << shift); i <= (1 << max_shift); i*=2) {

    double runtime {0.0};

    for(unsigned j = 0; j < num_rounds; ++j) {
      
      if(model == "tf") {
        runtime += measure_time_taskflow(num_threads, vtype, i).count();
      }
      else if(model == "pthread") {
        runtime += measure_time_pthread(num_threads, vtype, i).count();
      }
    }

    std::cout << std::setw(12) << i
              << std::setw(12) << runtime / num_rounds / 1e3
              << std::endl;

  }
}
