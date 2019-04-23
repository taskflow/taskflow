#include <thread>
#include <iomanip>
#include <CLI11.hpp>
#include "dnn.hpp"

// Function: measure_time_taskflow
std::chrono::milliseconds measure_time_taskflow(
  unsigned num_epochs,
  unsigned num_threads
) {
  auto dnn {build_dnn(num_epochs)};
  auto t1 = std::chrono::high_resolution_clock::now();
  run_taskflow(dnn, num_threads);
  auto t2 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
}

// Function: measure_time_omp
std::chrono::milliseconds measure_time_omp(
  unsigned num_epochs,
  unsigned num_threads
) {
  auto dnn {build_dnn(num_epochs)};
  auto t1 = std::chrono::high_resolution_clock::now();
  run_omp(dnn, num_threads);
  auto t2 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
}

// Function: measure_time_tbb
std::chrono::milliseconds measure_time_tbb(
  unsigned num_epochs,
  unsigned num_threads
) {
  auto dnn {build_dnn(num_epochs)};
  auto t1 = std::chrono::high_resolution_clock::now();
  run_tbb(dnn, num_threads);
  auto t2 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
}

// Function: main
int main(int argc, char *argv[]){

  CLI::App app{"MNIST"};

  unsigned num_threads {1}; 
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned num_epochs {10}; 
  app.add_option("-e,--num_epochs", num_epochs, "number of epochs (default=10)");

  unsigned num_rounds {1};  
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");

  std::string model = "tf";
  app.add_option("-m,--model", model, "model name (tbb|omp|tf(default))");

  CLI11_PARSE(app, argc, argv);

  double runtime  {0.0};

  for(unsigned i=0; i<num_rounds; i++) {
    if(model == "tf") {
      runtime += measure_time_taskflow(num_epochs, num_threads).count();
    }
    else if(model == "tbb") {
      runtime += measure_time_tbb(num_epochs, num_threads).count();
    }
    else if(model == "omp") {
      runtime += measure_time_omp(num_epochs, num_threads).count();
    }
    else {
      std::cout << "Unsupported model = " << model << '\n';
      break;
    }
  }

  std::cout << model << '=' << runtime / num_rounds / 1e3
            << " threads=" << num_threads 
            << " epochs=" << num_epochs 
            << " rounds=" << num_rounds 
            << std::endl;

  return EXIT_SUCCESS;
}

