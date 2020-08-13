#include <CLI11.hpp>
#include "mandel.hpp"

// Reference:
//   https://en.wikipedia.org/wiki/Mandelbrot_set#Generalizations
//   https://people.sc.fsu.edu/~jburkardt/cpp_src/mandelbrot_openmp/mandelbrot_openmp.html
//   http://courses.cms.caltech.edu/cs11/material/dgc/lab4/ 
//   https://github.com/gasparian/mandelbrot_cpp/blob/master/mandelbrot.cpp
//   https://csl.name/post/mandelbrot-rendering/ 
//   http://linas.org/art-gallery/escape/smooth.html

int H = 1000; 
int W = 1000; 
unsigned char* RGB = nullptr;

void mandelbrot(
  const std::string& model,
  const unsigned num_threads, 
  const unsigned num_rounds
  ) {

  std::cout << std::setw(12) << "size"
            << std::setw(12) << "runtime"
            << std::endl;
 
  for(int N = 100; N<=1000; N+=100) {

    W = N;
    H = N;
  
    double runtime {0.0};
    RGB = static_cast<unsigned char *>(malloc (W * H * 3 * sizeof(unsigned char)));

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

    std::cout << std::setw(12) << N
              << std::setw(12) << runtime / num_rounds / 1e3
              << std::endl;

    //dump_tga(W, H, RGB, "mandelbrot_set.tga");
    free(RGB);
  }
}


int main(int argc, char* argv[]) {

  CLI::App app{"Mandelbrot set"};

  unsigned num_threads {1}; 
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

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
            << std::endl; 

  mandelbrot(model, num_threads, num_rounds);
}


