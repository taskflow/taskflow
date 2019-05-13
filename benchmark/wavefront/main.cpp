#include "matrix.hpp"
#include <CLI11.hpp>

void framework_wavefront(
  const std::string& model,
  const unsigned num_threads, 
  const unsigned num_rounds
  ) {
  
  const int repeat [] = {1, 5, 10, 100};

  std::cout << std::setw(12) << "size"
            << std::setw(12) << repeat[0]
            << std::setw(12) << repeat[1]
            << std::setw(12) << repeat[2]
            << std::setw(12) << repeat[3]
            << std::endl;

  for(int S=32; S<=4096; S += 128) {
    M = N = S;
    B = 8;
    MB = (M/B) + (M%B>0);
    NB = (N/B) + (N%B>0);

    std::cout << std::setw(12) << MB*NB;

    for(int k=0; k<4; k++) {

      double runtime {0.0};

      init_matrix();

      for(unsigned j=0; j<num_rounds; ++j) {
        if(model == "tf") {
          runtime += measure_time_taskflow(num_threads, repeat[k]).count();
        }
        else if(model == "tbb") {
          runtime += measure_time_tbb(num_threads, repeat[k]).count();
        }
        else if(model == "omp") {
          runtime += measure_time_omp(num_threads, repeat[k]).count();
        }
        else assert(false);
      }

      destroy_matrix();
    
      std::cout << std::setw(12) 
                << std::setprecision (2) << std::fixed
                << runtime / num_rounds / 1e3;
    }
    std::cout << std::endl;
  }
}


void wavefront(
  const std::string& model,
  const unsigned num_threads, 
  const unsigned num_rounds
  ) {

  std::cout << std::setw(12) << "size"
            << std::setw(12) << "runtime"
            << std::endl;
  
  for(int S=32; S<=4096; S += 128) {

    M = N = S;
    B = 8;
    MB = (M/B) + (M%B>0);
    NB = (N/B) + (N%B>0);
  
    double runtime {0.0};

    init_matrix();

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

    destroy_matrix();
    
    std::cout << std::setw(12) << MB*NB
              << std::setw(12) << runtime / num_rounds / 1e3
              << std::endl;
  }
}

int main(int argc, char* argv[]) {

  CLI::App app{"Wavefront"};

  unsigned num_threads {1}; 
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned num_rounds {1};  
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");

  bool use_framework {false};
  app.add_flag("-f", use_framework, "enable framework run"); 

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
            << "use_framework=" << std::boolalpha << use_framework << ' '
            << std::endl;

  if(use_framework) {
    framework_wavefront(model, num_threads, num_rounds);
  }
  else {
    wavefront(model, num_threads, num_rounds);
  }

  return 0;
}


