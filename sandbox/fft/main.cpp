//	14 APR 2011 Lukasz G. Szafaryn

#include <chrono>
#include <iostream>


#include "CLI11.hpp"
#include "fft.h"

#define BOTS_APP_NAME "FFT"
#define BOTS_APP_PARAMETERS_DESC "Size=%d"
#define BOTS_APP_PARAMETERS_LIST ,bots_arg_size

#define BOTS_APP_USES_ARG_SIZE
#define BOTS_APP_DEF_ARG_SIZE 32*1024*1024
#define BOTS_APP_DESC_ARG_SIZE "Matrix Size"

#define BOTS_APP_INIT int i;\
     COMPLEX *in, *out1=NULL, *out2=NULL;\
     in = (COMPLEX *)malloc(bots_arg_size * sizeof(COMPLEX));\

#define KERNEL_INIT\
     out1 = (COMPLEX *)malloc(bots_arg_size * sizeof(COMPLEX));\
     for (i = 0; i < bots_arg_size; ++i) {\
          c_re(in[i]) = 1.0;\
          c_im(in[i]) = 1.0;\
     }
#define KERNEL_CALL fft(bots_arg_size, in, out1);
#define KERNEL_FINI 

#define KERNEL_SEQ_INIT\
     out2 = (COMPLEX *)malloc(bots_arg_size * sizeof(COMPLEX));\
     for (i = 0; i < bots_arg_size; ++i) {\
          c_re(in[i]) = 1.0;\
          c_im(in[i]) = 1.0;\
     }
#define KERNEL_SEQ_CALL fft_seq(bots_arg_size, in, out2);
#define KERNEL_SEQ_FINI

#define BOTS_APP_CHECK_USES_SEQ_RESULT
#define KERNEL_CHECK test_correctness(bots_arg_size, out1, out2)



int main(	int argc, char *argv []) {

  // A molecular dynamics application that calculates the potential and relocation of particles within a large 3D space.
  CLI::App app{"FFT"};

  // num_threads should be roughly equal to NUMBER_PAR_PER_BOX for best performance
  unsigned num_threads {1};
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned num_boxes {1};
  app.add_option("-b,--boxes1d", num_boxes, "number of threads (default=1)");

  int num_values {1024};
  app.add_option("-v,--num_values", num_values, "number of values (default=1024)");

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
            << "num_values=" << num_values << ' '
            << std::endl;

  int i;
  COMPLEX *in, *out1=NULL, *out2=NULL;
  in = (COMPLEX *)malloc(num_values * sizeof(COMPLEX));
  out1 = (COMPLEX *)malloc(num_values * sizeof(COMPLEX));

  for (i = 0; i < num_values; ++i) {
    c_re(in[i]) = 1.0;
    c_im(in[i]) = 1.0;
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  //fft(num_values, in, out1); 
  fft_tf(num_values, in, out1);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1e3 << " ms\n";


  {
    out2 = (COMPLEX *)malloc(num_values * sizeof(COMPLEX));
    for (i = 0; i < num_values; ++i) {
      c_re(in[i]) = 1.0;
      c_im(in[i]) = 1.0;
    }
    fft_seq(num_values, in, out2);
    test_correctness(num_values, out1, out2);
  }

	return 0;																		
}
