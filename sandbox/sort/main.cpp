//	14 APR 2011 Lukasz G. Szafaryn

#include <chrono>
#include <iostream>
#include <algorithm>

#include "CLI11.hpp"
#include "sort.hpp"

int main(	int argc, char *argv []) {

  // A molecular dynamics application that calculates the potential and relocation of particles within a large 3D space.
  CLI::App app{"FFT"};

  // num_threads should be roughly equal to NUMBER_PAR_PER_BOX for best performance
  unsigned num_threads {1};
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  int num_values {1024}; // 33554432 this is the value set by bots
  app.add_option("-v,--num_values", num_values, "number of values must be power of 2 (default=1024)");

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

  sort_init();

  auto t1 = std::chrono::high_resolution_clock::now();
  //std::sort(array, array+array_size);
  sort_par();
  //sort_par_tf();
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1e3 << " ms\n";

  sort_verify();

	return 0;																		
}
