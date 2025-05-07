#include "integrate.hpp"
#include <CLI11.hpp>

void bench_integrate(
  const std::string& model,
  const size_t num_threads,
  const size_t num_rounds,
  const size_t max_value
  ) {

  std::cout << std::setw(12) << "size"
            << std::setw(12) << "runtime"
            << std::endl;

  for(size_t v=0; v<=max_value; v+=100) {

    double runtime {0.0};

    for(unsigned j=0; j<num_rounds; ++j) {
      if(model == "tf") {
        runtime += measure_time_taskflow(num_threads, v).count();
      }
      else if(model == "omp") {
        runtime += measure_time_omp(num_threads, v).count();
      }
      else if(model == "tbb") {
        runtime += measure_time_tbb(num_threads, v).count();
      }
      
      else assert(false);
    }

    std::cout << std::setw(12) << v
              << std::setw(12) << runtime / num_rounds / 1e3
              << std::endl;
  }
}

int main(int argc, char* argv[]) {

  CLI::App app{"Integrate"};

  size_t num_threads {1};
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");
  
  size_t max_value {2000};
  app.add_option("-v,--value", max_value, "max x-value (default=100)");

  size_t num_rounds {1};
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");

  std::string model = "tf";
  app.add_option("-m,--model", model, "model name omp|tf|tbb (default=tf)")
     ->check([] (const std::string& m) {
        if(m != "omp" && m != "tf" && m != "tbb") {
          return "model name should be \"omp\", \"tbb\", or \"tf\"";
        }
        return "";
     });

  CLI11_PARSE(app, argc, argv);

  std::cout << "model="       << model << ' '
            << "num_threads=" << num_threads << ' '
            << "num_rounds="  << num_rounds << ' '
            << "max_value="   << max_value << ' '
            << std::endl;

  bench_integrate(model, num_threads, num_rounds, max_value);

  return 0;
}


