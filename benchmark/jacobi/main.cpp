#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <CLI11.hpp>
#include <omp.h>
#include <string>

#include "poisson.hpp"

#define min(a, b) ((a<b)?a:b)
#define max(a, b) ((a>b)?a:b)


int comp (const void * elem1, const void * elem2) {
  double f = *((double*)elem1);
  double s = *((double*)elem2);
  if (f > s) return  1;
  if (f < s) return -1;
  return 0;
}


// This benchmarks is modified from KaStORS benchmarks:
//   Evaluation of the OpenMP dependent tasks with the KASTORS benchmark suite
int main(int argc, char* argv[]) {
  struct user_parameters params;
  memset(&params, 0, sizeof(params));

  CLI::App app{"Jacobi"};
  int num_threads = 1;
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  params.titer = 1;
  app.add_option("-i,--num_iterations", params.titer, "number of iterations (default=1)");

  params.niter = 1;
  app.add_option("-r,--num_repeats", params.niter, "number of repeats (default=1)");

  params.matrix_size = 8192;
  app.add_option("-n,--matrix_size", params.matrix_size, "matrix size (default=8192)");

  params.blocksize = 128;
  app.add_option("-b,--block_size", params.blocksize, "block size (default=128)");

  params.check = 1; // Turn on check by default


  std::string model = "tf";
  app.add_option("-m,--model", model, "model name seq|tbb|omp|tf (default=tf)")
    ->check([] (const std::string& m) {
    if(m != "seq" && m != "tbb" && m != "tf" && m != "omp") {
       return "model name should be \"seq\", \"tbb\", \"omp\", or \"tf\"";
    }
    return "";
  });

  CLI11_PARSE(app, argc, argv);

  double mean = 0.0;
  double meansqr = 0.0;
  double min_ = DBL_MAX;
  double max_ = -1;
  double* all_times = (double*)malloc(sizeof(double) * params.niter); 

  for (int i=0; i<params.niter; ++i) {
    double cur_time = run(&params, num_threads, model)/1e3;
    all_times[i] = cur_time;
    mean += cur_time;
    min_ = min(min_, cur_time);
    max_ = max(max_, cur_time);
    meansqr += cur_time * cur_time;
  }

  mean /= params.niter;
  meansqr /= params.niter;
  double stddev = sqrt(meansqr - mean * mean);

  qsort(all_times, params.niter, sizeof(double), comp);
  double median = all_times[params.niter / 2];

  free(all_times);

  std::cout << "Mode : " << model << "\n";
  printf("Size : %d\n", params.matrix_size);
  printf("Blocksize : %d\n", params.blocksize);
  printf("Threads : %d\n", num_threads); 
  printf("Repeat : %d\n", params.niter);
  printf("Iterations : %d\n", params.titer);

  printf("avg : %lf :: std : %lf :: min : %lf :: max : %lf :: median : %lf\n",
      mean, stddev, min_, max_, median);
  printf("Check : %s\n", (params.succeed)? ((params.succeed > 1)?"not implemented":"success"):"fail");

  return 0;
}
