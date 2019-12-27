//	14 APR 2011 Lukasz G. Szafaryn

#include "CLI11.hpp"
#include "lavamd.hpp"


void lavamd(
  const std::string& model,
  const unsigned num_threads,
  const unsigned num_rounds
  ) {

  double runtime {0.0};

  for(unsigned j=0; j<num_rounds; ++j) {
    init();
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
    clear();
  }

  std::cout << runtime/ num_rounds / 1e3 << std::endl;
}


int main(	int argc, char *argv []) {

  // A molecular dynamics application that calculates the potential and relocation of particles within a large 3D space.
  CLI::App app{"LavaMD"};

  // num_threads should be roughly equal to NUMBER_PAR_PER_BOX for best performance
  unsigned num_threads {1};
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned num_boxes {1};
  app.add_option("-b,--boxes1d", num_boxes, "number of threads (default=1)");

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

  dim_cpu.cores_arg = num_threads;
  dim_cpu.boxes1d_arg = num_boxes;

	// Print configuration
  std::cout << "model=" << model << ' '
            << "num_threads=" << num_threads << ' '
            << "num_rounds=" << num_rounds << ' '
            << "num_boxes=" << num_boxes << ' '
            << std::endl;

  lavamd(model, num_threads, num_rounds);

	return 0;																		
}
