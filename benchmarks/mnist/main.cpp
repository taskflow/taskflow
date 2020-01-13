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

// Procedure
void mnist(
  const std::string& model,
  const unsigned min_epochs,
  const unsigned max_epochs,
  const unsigned num_threads, 
  const unsigned num_rounds
) {
  
  std::cout << std::setw(12) << "epochs"
            << std::setw(12) << "runtime"
            << std::endl;

  for(unsigned epochs=min_epochs; epochs <= max_epochs; epochs += 10) {

    double runtime  {0.0};

    for(unsigned i=0; i<num_rounds; i++) {
    
      if(model == "tf") {
        runtime += measure_time_taskflow(epochs, num_threads).count();
      }
      else if(model == "tbb") {
        runtime += measure_time_tbb(epochs, num_threads).count();
      }
      else if(model == "omp") {
        runtime += measure_time_omp(epochs, num_threads).count();
      }
      else assert(false);

      std::cout << std::setw(12) << epochs
                << std::setw(12) << runtime / num_rounds / 1e3
                << std::endl;
    }
  }

}

// Function: main
int main(int argc, char *argv[]){

  CLI::App app{"DNN Training on MNIST Dataset"};

  unsigned num_threads {1}; 
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned max_epochs {100}; 
  app.add_option("-E,--max_epochs", max_epochs, "max number of epochs (default=100)");
  
  unsigned min_epochs {10}; 
  app.add_option("-e,--min_epochs", min_epochs, "min number of epochs (default=10)");

  unsigned num_rounds {1};  
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");

  std::string model = "tf";
  app.add_option("-m,--model", model, "model name tbb|omp|tf (default=tf)")
     ->check([] (const std::string& m) {
        if(m != "tbb" && m != "omp" && m != "tf") {
          return "model name should be \"tbb\", \"omp\", or \"tf\"";
        }
        return "";
     });

  CLI11_PARSE(app, argc, argv);
    
  std::cout << "model=" << model << ' '
            << "num_threads=" << num_threads << ' '
            << "num_rounds=" << num_rounds << ' '
            << "min_epochs=" << min_epochs << ' '
            << "max_epochs=" << max_epochs << ' '
            << std::endl;

  mnist(model, min_epochs, max_epochs, num_threads, num_rounds);
  
  return EXIT_SUCCESS;
}




