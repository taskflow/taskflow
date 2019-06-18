#include "linear_chain.hpp"
#include <CLI11.hpp>

void linear_chain(
  const std::string& model,
  const size_t log_length,
  const unsigned num_threads, 
  const unsigned num_rounds
  ) {

  std::cout << std::setw(12) << "length"
            << std::setw(12) << "runtime"
            << std::endl;
  
  for(size_t i=1; i<=log_length; ++i) {

    size_t L = 1 << i;

    double runtime {0.0};

    for(unsigned j=0; j<num_rounds; ++j) {
      if(model == "tf") {
        runtime += measure_time_taskflow(L, num_threads).count();
      }
      else if(model == "tbb") {
        runtime += measure_time_tbb(L, num_threads).count();
      }
      else if(model == "omp") {
        runtime += measure_time_omp(L, num_threads).count();
      }
      else assert(false);
    }

    std::cout << std::setw(12) << L
              << std::setw(12) << runtime / num_rounds / 1e3
              << std::endl;
  }
}

int main(int argc, char* argv[]) {

  CLI::App app{"LinearChain"};

  unsigned num_threads {1}; 
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned num_rounds {1};  
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");
  
  size_t log_length {25};  
  app.add_option("-l,--log_length", log_length, "length in log scale (default=25)");

  std::string model = "tf";
  app.add_option("-m,--model", model, "model name tbb|omp|tf (default=tf)")
     ->check([] (const std::string& m) {
        if(m != "tbb" && m != "tf" && m != "omp") {
          return "model name should be \"tbb\", \"omp\", or \"tf\"";
        }
        return "";
     });

  CLI11_PARSE(app, argc, argv);
   
  std::cout << "model=" << model << ' '
            << "num_threads=" << num_threads << ' '
            << "num_rounds=" << num_rounds << ' '
            << std::endl;

  linear_chain(model, log_length, num_threads, num_rounds);

  return 0;
}


