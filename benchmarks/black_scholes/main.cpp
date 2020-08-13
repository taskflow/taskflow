// Copyright (c) 2007 Intel Corp.

// Black-Scholes
// Analytical method for calculating European Options
// 
// Reference Source: Options, Futures, and Other Derivatives, 3rd Edition, Prentice 
// Hall, John C. Hull,
//
// Modified from 
//   https://parsec.cs.princeton.edu/

#include "common.hpp"
#include <CLI11.hpp>

// extern global variables
OptionData *optdata = nullptr;
float *prices = nullptr;
int numOptions = 0;
int NUM_RUNS = 0;
int*    otype = nullptr;
float* sptprice = nullptr;
float* strike = nullptr;
float* rate = nullptr;
float* volatility = nullptr;
float* otime = nullptr;
int numError = 0;
float* BUFFER = nullptr;
int* BUFFER2 = nullptr;

void black_scholes(
  const std::string& model,
  const unsigned num_threads,
  const unsigned num_rounds
) {
  
  std::cout << std::setw(12) << "size"
            << std::setw(12) << "runtime"
            << std::endl;

  for(size_t N = 1000; N<=10000; N+=1000) {

    generate_options(N);

    double runtime {0.0};

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
    
    destroy_options();

    std::cout << std::setw(12) << N
              << std::setw(12) << runtime / num_rounds / 1e3
              << std::endl;
  }
}


int main (int argc, char *argv[]) {

  CLI::App app{"Option pricing with Black-Scholes Partial Differential Equation"};

  unsigned num_threads {1}; 
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned num_rounds {1};  
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");
  
  bool cmp_seq {false};
  app.add_option("-s,--seq", cmp_seq, "compare with sequential (default=false)");

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

  black_scholes(model, num_threads, num_rounds);

  // Compare with sequential version to check correctness 
  //if(cmp_seq) {
  //  auto seq_prices = static_cast<FPTYPE*>(malloc(numOptions*sizeof(FPTYPE)));
  //  bs_seq(seq_prices);
  //  for(auto i=0; i<numOptions; i++) {
  //    assert(std::fabs(seq_prices[i] - prices[i]) < 0.01);
  //  }
  //  std::cout << "Results are consistent with the sequential version\n";
  //  free(seq_prices);
  //}

#ifdef ERR_CHK
    printf("Num Errors: %d\n", numError);
#endif

  return 0;
}

