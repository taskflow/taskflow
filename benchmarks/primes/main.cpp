#include "primes.hpp"
#include <CLI11.hpp>

void bench_primes(
  const std::string& model,
  const unsigned num_threads,
  const unsigned num_rounds
  ) {

  std::cout << std::setw(12) << "size"
            << std::setw(12) << "runtime"
            << std::endl;

  double runtime {0.0};

  for (size_t p = 10; p <= primes_limit; p*=10) {

    for(unsigned j=0; j<num_rounds; ++j) {
      if(model == "tf") {
        runtime += measure_time_taskflow(num_threads, p).count();
      }
      else if(model == "omp") {
        runtime += measure_time_omp(num_threads, p).count();
      }
      else if(model == "tbb") {
        runtime += measure_time_tbb(num_threads, p).count();
      }
      
      else assert(false);
    }

    std::cout << std::setw(12) << p 
              << std::setw(12) << runtime / num_rounds / 1e3
              << std::endl;
  }  
}

int main(int argc, char* argv[]) {

  CLI::App app{"Primes"};

  unsigned num_threads {1};
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");
  
  unsigned num_rounds {1};
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
            << "primes_limit="   << primes_limit << ' '
            << std::endl;

  bench_primes(model, num_threads, num_rounds);

  return 0;
}


