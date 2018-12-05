#include "matrix.hpp"

int main() {

  double omp_time {0.0};
  double tbb_time {0.0};
  double tf_time  {0.0};
  int rounds {5};

  std::cout << std::setw(12) << "# blocks"
            << std::setw(12) << "OpenMP"
            << std::setw(12) << "TBB"
            << std::setw(12) << "Taskflow"
            << std::endl;
  
  for(int S=32; S<=4096; S += 32) {

    M = N = S;
    B = 8;
    MB = (M/B) + (M%B>0);
    NB = (N/B) + (N%B>0);

    init_matrix();

    for(int j=0; j<rounds; ++j) {
      omp_time += measure_time_omp().count();
      tbb_time += measure_time_tbb().count();
      tf_time  += measure_time_taskflow().count();
    }

    destroy_matrix();
    
    std::cout << std::setw(12) << MB*NB
              << std::setw(12) << omp_time / rounds / 1000.0
              << std::setw(12) << tbb_time / rounds / 1000.0
              << std::setw(12) << tf_time  / rounds / 1000.0
              << std::endl;
  }

  return 0;
}
