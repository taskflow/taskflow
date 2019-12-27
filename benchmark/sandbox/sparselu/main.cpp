//	14 APR 2011 Lukasz G. Szafaryn

#include <chrono>
#include <iostream>
#include <string>

#include "CLI11.hpp"
#include "sparselu.h"


//#define BOTS_APP_INIT float **SEQ,**BENCH;

#define KERNEL_INIT sparselu_init(&BENCH,"benchmark");
#define KERNEL_CALL sparselu_par_call(BENCH);
#define KERNEL_FINI sparselu_fini(BENCH,"benchmark");

#define KERNEL_SEQ_INIT sparselu_init(&SEQ,"serial");
#define KERNEL_SEQ_CALL sparselu_seq_call(SEQ);
#define KERNEL_SEQ_FINI sparselu_fini(SEQ,"serial");

#define BOTS_APP_CHECK_USES_SEQ_RESULT
#define KERNEL_CHECK sparselu_check(SEQ,BENCH);


int main(	int argc, char *argv []) {

  // A molecular dynamics application that calculates the potential and relocation of particles within a large 3D space.
  CLI::App app{"SparseLU"};

  // num_threads should be roughly equal to NUMBER_PAR_PER_BOX for best performance
  unsigned num_threads {1};
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  app.add_option("-s,--matrix_size", matrix_size, "matrix size (default=256)");
  app.add_option("-b,--submatrix_size", submatrix_size, "submatrix size (default=16)");

  std::string model = "tf";
  app.add_option("-m,--model", model, "model name tbb|omp|tf (default=tf)")
    ->check([] (const std::string& m) {
      if(m != "tbb" && m != "omp" && m != "tf") {
        return "model name should be \"tbb\", \"omp\", or \"tf\"";
      }
      return "";
  });

  CLI11_PARSE(app, argc, argv);

	// Print configuration
  std::cout << "model=" << model << ' '
            << "num_threads=" << num_threads << ' '
            << "matrix_size=" << matrix_size << ' '
            << "submatrix_size=" << submatrix_size << ' '
            << std::endl;

  float** BENCH; 
  {
    std::string name {"BENCHMARK"};
    sparselu_init(&BENCH, name.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    sparselu_par_call(BENCH); 
    //sparselu_tf_for(BENCH);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Runtime: " <<std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1e3 << " ms\n";
  }


  float** SEQ;
  {
    std::string name {"serial"};
    sparselu_init(&SEQ, name.data());
    sparselu_seq_call(SEQ);
  }

  sparselu_check(SEQ,BENCH);
  return 0;																		
}
