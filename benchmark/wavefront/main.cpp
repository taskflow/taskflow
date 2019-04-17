#include "matrix.hpp"

void framework_wavefront(unsigned num_threads) {
  int rounds {5};
  const int repeat [] = {1, 5, 10, 100};

  std::cout << std::setw(12) << "# blocks"
            << std::setw(12) << "Repeat"
            << std::setw(12) << "OpenMP"
            << std::setw(12) << "TBB"
            << std::setw(12) << "Taskflow"
            << std::setw(12) << "speedup1"
            << std::setw(12) << "speedup2"
            << '\n';
  std::cout << std::string(100, '=') << std::endl; 
  
  const std::string line (100, '-');

  for(int S=32; S<=4096; S += 128) {
    for(int k=0; k<4; k++) {
      M = N = S;
      B = 8;
      MB = (M/B) + (M%B>0);
      NB = (N/B) + (N%B>0);
  
      double omp_time {0.0};
      double tbb_time {0.0};
      double tf_time  {0.0};

      init_matrix();

      for(int j=0; j<rounds; ++j) {
        omp_time += measure_time_omp(num_threads, repeat[k]).count();
        tbb_time += measure_time_tbb(num_threads, repeat[k]).count();
        tf_time  += measure_time_taskflow(num_threads, repeat[k]).count();
      }

      destroy_matrix();
      
      std::cout << std::setw(12) << MB*NB
                << std::setw(12) << repeat[k]
                << std::setw(12) << omp_time / rounds / 1e3
                << std::setw(12) << tbb_time / rounds / 1e3 
                << std::setw(12) << tf_time  / rounds / 1e3 
                << std::setw(12) << omp_time / tf_time
                << std::setw(12) << tbb_time / tf_time
                << std::endl;
    }
    std::cout << line << std::endl;
  }
}

void taskflow_wavefront(unsigned num_threads) {
  int rounds {5};

  std::cout << std::setw(12) << "# blocks"
            << std::setw(12) << "OpenMP"
            << std::setw(12) << "TBB"
            << std::setw(12) << "Taskflow"
            << std::setw(12) << "speedup1"
            << std::setw(12) << "speedup2"
            << '\n';
  
  for(int S=32; S<=4096; S += 128) {

    M = N = S;
    B = 8;
    MB = (M/B) + (M%B>0);
    NB = (N/B) + (N%B>0);
  
    double omp_time {0.0};
    double tbb_time {0.0};
    double tf_time  {0.0};

    init_matrix();

    for(int j=0; j<rounds; ++j) {
      omp_time += measure_time_omp(num_threads).count();
      tbb_time += measure_time_tbb(num_threads).count();
      tf_time  += measure_time_taskflow(num_threads).count();
    }

    destroy_matrix();
    
    std::cout << std::setw(12) << MB*NB
              << std::setw(12) << omp_time / rounds / 1e3
              << std::setw(12) << tbb_time / rounds / 1e3 
              << std::setw(12) << tf_time  / rounds / 1e3 
              << std::setw(12) << omp_time / tf_time
              << std::setw(12) << tbb_time / tf_time
              << std::endl;
  }
}

int main(int argc, char* argv[]) {

  unsigned num_threads = std::thread::hardware_concurrency();

  if(argc > 1) {
    num_threads = std::atoi(argv[1]);
  }

  if(argc == 2) {
    taskflow_wavefront(num_threads);
  }
  else {
    framework_wavefront(num_threads);
  }

  return 0;
}


