#include <CLI11.hpp>
#include "embarrassing_parallelism.hpp"

void embarrassing_parallelism(
  const std::string& model,
  const unsigned num_threads,
  const unsigned num_rounds
  ) {

  std::cout << std::setw(12) << "size"
            << std::setw(12) << "runtime"
            << std::endl;

  for(size_t num_tasks=1; num_tasks<=(1<<20); num_tasks <<= 1) {

    double runtime {0.0};

    for(unsigned j=0; j<num_rounds; ++j) {
      if(model == "tf") {
        runtime += measure_time_taskflow(num_threads, num_tasks).count();
      }
      else if(model == "tbb") {
        runtime += measure_time_tbb(num_threads, num_tasks).count();
      }
      else if(model == "omp") {
        runtime += measure_time_omp(num_threads, num_tasks).count();
      }
      else if(model == "seq") {
        runtime += measure_time_seq(num_tasks).count();
      }
      else assert(false);
    }

    std::cout << std::setw(12) << num_tasks
              << std::setw(12) << runtime / num_rounds / 1e3
              << std::endl;
  }
}

int main(int argc, char* argv[]) {

  CLI::App app{"EmbarrassingParallelism"};

  unsigned num_threads {1};
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned num_rounds {1};
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");

  std::string model = "tf";
  app.add_option("-m,--model", model, "model name tbb|omp|tf (default=tf)")
     ->check([] (const std::string& m) {
        if(m != "tbb" && m != "omp" && m != "tf" && m!= "seq") {
          return "model name should be \"tbb\", \"seq\", \"omp\", or \"tf\"";
        }
        return "";
     });

  CLI11_PARSE(app, argc, argv);

  std::cout << "model=" << model << ' '
            << "num_threads=" << num_threads << ' '
            << "num_rounds=" << num_rounds << ' '
            << std::endl;

  embarrassing_parallelism(model, num_threads, num_rounds);

  return 0;
}


