#include "levelgraph.hpp"

int main(int argc, char* argv[]) {

  unsigned num_threads = std::thread::hardware_concurrency();

  if(argc > 1) {
    num_threads = std::atoi(argv[1]);
  }

  const int width {12};
  const int rounds {5};
  const int repeat [] = {1, 5, 10, 100};
  
  std::cout << std::setw(width) << "|V|+|E|"
            << std::setw(width) << "Repeat"
            << std::setw(width) << "OMP"
            << std::setw(width) << "TBB"
            << std::setw(width) << "TF"
            << std::setw(width) << "speedup1"
            << std::setw(width) << "speedup2"
            << '\n';
  std::cout << std::string(100, '=') << std::endl; 

  std::cout.precision(3);

  const std::string line (100, '-');

  for(int i=1; i<=451; i += 15) {

    for(int k=0; k<4; k++) {
      double omp_time {0.0};
      double tbb_time {0.0};
      double tf_time  {0.0};

      LevelGraph graph(i, i);

      if(k == 0) {
        std::cout << std::setw(width) << graph.graph_size();
      }
      else {
        std::cout << std::setw(width) << " ";
      }
      
      for(int j=0; j<rounds; ++j) {
        omp_time += measure_time_omp(graph, num_threads, repeat[k]).count();

        tbb_time += measure_time_tbb(graph, num_threads, repeat[k]).count();

        tf_time  += measure_time_taskflow(graph, num_threads, repeat[k]).count();
      }
              
      std::cout << std::setw(width) << repeat[k] << std::fixed
                << std::setw(width) << omp_time / rounds / 1e3 << std::fixed
                << std::setw(width) << tbb_time / rounds / 1e3 << std::fixed
                << std::setw(width) << tf_time  / rounds / 1e3 << std::fixed
                << std::setw(width) << omp_time / tf_time  << std::fixed
                << std::setw(width) << tbb_time / tf_time  << std::fixed
                << std::endl;
    }
    std::cout << line << std::endl;
  }
}


