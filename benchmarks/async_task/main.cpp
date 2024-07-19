#include "async_task.hpp"
#include <CLI11.hpp>

void bench_async_task(
  const std::string& model,
  const unsigned num_threads,
  const unsigned num_rounds
  ) {

  std::cout << std::setw(12) << "size"
            << std::setw(12) << "runtime"
            << std::endl;

  for(int S=1; S<=2097152; S<<=1) {

    double runtime {0.0};

    for(unsigned j=0; j<num_rounds; ++j) {
      if(model == "tf") {
        runtime += measure_time_taskflow(num_threads, S).count();
      }
      else if(model == "std") {
        runtime += measure_time_std(num_threads, S).count();
      }
      else if(model == "omp") {
        runtime += measure_time_omp(num_threads, S).count();
      }
      else assert(false);
    }

    std::cout << std::setw(12) << S
              << std::setw(12) << runtime / num_rounds / 1e3
              << std::endl;
  }
}

int main(int argc, char* argv[]) {

  CLI::App app{"AsyncTask"};

  unsigned num_threads {1};
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned num_rounds {1};
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");

  std::string model = "tf";
  app.add_option("-m,--model", model, "model name std|omp|tf (default=tf)")
     ->check([] (const std::string& m) {
        if(m != "std" && m != "omp" && m != "tf") {
          return "model name should be \"std\", \"omp\", or \"tf\"";
        }
        return "";
     });

  CLI11_PARSE(app, argc, argv);

  std::cout << "model=" << model << ' '
            << "num_threads=" << num_threads << ' '
            << "num_rounds=" << num_rounds << ' '
            << std::endl;

  bench_async_task(model, num_threads, num_rounds);

  return 0;
}


