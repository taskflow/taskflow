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

void black_scholes(
  const std::string& model,
  const unsigned num_threads,
  const unsigned num_rounds
  ) {

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

  std::cout << runtime/ num_rounds / 1e3 << std::endl;
}


int main (int argc, char *argv[]) {

  CLI::App app{"Option pricing with Black-Scholes Partial Differential Equation"};

  unsigned num_threads {1}; 
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned num_rounds {1};  
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");
  
  bool cmp_seq {false};
  app.add_option("-s,--seq", cmp_seq, "compare with sequential (default=false)");

  std::string option_file;
  app.add_option("-i,--input", option_file, "require an option file")->required()->check(CLI::ExistingFile);

  std::string output_file;
  app.add_option("-o,--output", output_file, "output file");

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
            << "cmp_sequential=" << cmp_seq << ' '
            << "input_file=" << option_file << ' '
            << "output_file=" << output_file << ' '
            << std::endl;

  if(!parse_options(option_file)) {
    return 0;
  }

  black_scholes(model, num_threads, num_rounds);

  if(!output_file.empty()) {
    dump(output_file);
  }

  // Compare with sequential version to check correctness 
  if(cmp_seq) {
    auto seq_prices = static_cast<FPTYPE*>(malloc(numOptions*sizeof(FPTYPE)));
    bs_seq(seq_prices);
    for(auto i=0; i<numOptions; i++) {
      assert(std::fabs(seq_prices[i] - prices[i]) < 0.01);
    }
    std::cout << "Results are consistent with the sequential version\n";
    free(seq_prices);
  }

#ifdef ERR_CHK
    printf("Num Errors: %d\n", numError);
#endif

  free(data);
  free(prices);
  free(BUFFER);
  free(BUFFER2);
  return 0;
}

