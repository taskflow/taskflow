#include <thread>
#include <iomanip>
#include "dnn.hpp"

// Function: measure_time_taskflow
std::chrono::milliseconds measure_time_taskflow(
  unsigned num_epochs,
  unsigned num_threads
) {
  auto dnn {build_dnn(num_epochs)};
  auto t1 = std::chrono::high_resolution_clock::now();
  run_taskflow(dnn, num_threads);
  auto t2 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
}

// Function: measure_time_omp
std::chrono::milliseconds measure_time_omp(
  unsigned num_epochs,
  unsigned num_threads
) {
  auto dnn {build_dnn(num_epochs)};
  auto t1 = std::chrono::high_resolution_clock::now();
  run_omp(dnn, num_threads);
  auto t2 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
}

// Function: measure_time_tbb
std::chrono::milliseconds measure_time_tbb(
  unsigned num_epochs,
  unsigned num_threads
) {
  auto dnn {build_dnn(num_epochs)};
  auto t1 = std::chrono::high_resolution_clock::now();
  run_tbb(dnn, num_threads);
  auto t2 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
}

// Function: main
int main(int argc, char *argv[]){
  
  unsigned num_threads = std::thread::hardware_concurrency();

  if(argc > 1) {
    num_threads = std::atoi(argv[1]);
  }

  int rounds {2};

  std::cout << std::setw(12) << "# epochs"
            << std::setw(12) << "OpenMP"
            << std::setw(12) << "TBB"
            << std::setw(12) << "Taskflow"
            << std::setw(12) << "speedup1"
            << std::setw(12) << "speedup2"
            << '\n';


  for(int epoch=10; epoch<=100; epoch+=10) {
    
    double omp_time {0.0};
    double tbb_time {0.0};
    double tf_time  {0.0};

    for(int j=0; j<rounds; ++j) {
      //omp_time += measure_time_omp(epoch, num_threads).count();
      //tbb_time += measure_time_tbb(epoch, num_threads).count();
      tf_time  += measure_time_taskflow(epoch, num_threads).count();
    }
    
    std::cout << std::setw(12) << epoch 
              << std::setw(12) << omp_time / rounds / 1e3
              << std::setw(12) << tbb_time / rounds / 1e3 
              << std::setw(12) << tf_time  / rounds / 1e3 
              << std::setw(12) << omp_time / tf_time
              << std::setw(12) << tbb_time / tf_time
              << std::endl;
  }

  return EXIT_SUCCESS;
}

