// The skynet benchmark as described here:
// https://github.com/atemerev/skynet

#include "skynet.hpp"
#include <CLI11.hpp>

void bench_skynet(
  const std::string& model,
  const unsigned num_threads,
  const unsigned num_rounds
  ) {

  std::cout << std::setw(12) << "size"
            << std::setw(12) << "runtime"
            << std::endl;

  for(size_t MaxDepth=1; MaxDepth<=8; MaxDepth++) { 
    double runtime {0.0};

    for(unsigned j=0; j<num_rounds; ++j) {
      if(model == "tf") {
        runtime += measure_time_taskflow(num_threads, MaxDepth).count();
      }
      else if(model == "tbb") {
        runtime += measure_time_tbb(num_threads, MaxDepth).count();
      }
      else if(model == "omp") {
        runtime += measure_time_omp(num_threads, MaxDepth).count(); 
      }
      else assert(false);
    }

    std::cout << std::setw(12) << MaxDepth
              << std::setw(12) << runtime / num_rounds / 1e3
              << std::endl;
  }
}

int main(int argc, char* argv[]) {

  CLI::App app{"Skynet"};

  unsigned num_threads {1};
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");
  
  unsigned num_rounds {1};
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");

  std::string model = "tf";
  app.add_option("-m,--model", model, "model name tf|tbb|omp (default=tf)")
     ->check([] (const std::string& m) {
        if(m != "tf" && m != "tbb" && m != "omp") {
          return "model name should be \"tbb\", \"omp\", or \"tf\"";
        }
        return "";
     });


  CLI11_PARSE(app, argc, argv);

  std::cout << "model="       << model << ' '
            << "num_threads=" << num_threads << ' '
            << "num_rounds="  << num_rounds << ' '
            << std::endl;

  bench_skynet(model, num_threads, num_rounds);

  return 0;
}


