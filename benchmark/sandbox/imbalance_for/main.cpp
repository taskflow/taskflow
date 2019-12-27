#include <CLI11.hpp>
#include "sparse.hpp"

void imbalance(
  const std::string& model,
  const unsigned num_threads, 
  const unsigned num_rounds
  ) {

  double runtime {0.0};

  std::string fname {"TF19.mtx"};
  read_matrix(fname.data());
  toCSR();

  for(unsigned j=0; j<num_rounds; ++j) {
    if(model == "tf") {
      runtime += measure_time_taskflow(num_threads).count();
    }
    else if(model == "tbb") {
      runtime += measure_time_tbb(num_threads).count();
    }
    else if(model == "omp") {
      runtime += measure_time_omp(num_threads).count();
    }
    else assert(false);
  }

  clear();
  
  std::cout << std::setw(12) << runtime / num_rounds / 1e3
            << std::endl;
}

int main(int argc, char* argv[]) {

 CLI::App app{"Imbalance workload using number of non-zeros in sparse matrix"};

  unsigned num_threads {1}; 
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned num_rounds {1};  
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");
  
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

  if(std::ifstream input("./TF19.mtx"); !input.good()) {
    std::cout << "Please unzip the file TF19.mtx.zip first!\n";
    return 0;
  }

  imbalance(model, num_threads, num_rounds); 
}
