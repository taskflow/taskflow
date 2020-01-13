#include "binary_tree.hpp"
#include <CLI11.hpp>

void binary_tree(
  const std::string& model,
  const size_t num_layers,
  const unsigned num_threads, 
  const unsigned num_rounds
  ) {

  std::cout << std::setw(12) << "size"
            << std::setw(12) << "runtime"
            << std::endl;
  
  for(size_t i=1; i<=num_layers; ++i) {

    double runtime {0.0};

    for(unsigned j=0; j<num_rounds; ++j) {
      if(model == "tf") {
        runtime += measure_time_taskflow(i, num_threads).count();
      }
      else if(model == "tbb") {
        runtime += measure_time_tbb(i, num_threads).count();
      }
      else if(model == "omp") {
        runtime += measure_time_omp(i, num_threads).count();
      }
      else assert(false);
    }

    std::cout << std::setw(12) << (1 << i)
              << std::setw(12) << runtime / num_rounds / 1e3
              << std::endl;
  }
}

int main(int argc, char* argv[]) {

  CLI::App app{"BinaryTree"};

  unsigned num_threads {1}; 
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned num_rounds {1};  
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");
  
  size_t num_layers {25};  
  app.add_option("-l,--num_layers", num_layers, "number of layers (default=25)");

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

  binary_tree(model, num_layers, num_threads, num_rounds);

  return 0;
}


