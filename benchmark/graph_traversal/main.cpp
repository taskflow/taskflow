#include "levelgraph.hpp"

int main(int argc, char* argv[]) {

  unsigned num_threads = std::thread::hardware_concurrency();

  if(argc > 1) {
    num_threads = std::atoi(argv[1]);
  }

  double omp_time {0.0};
  double tbb_time {0.0};
  double tf_time  {0.0};
  int rounds {5};
  
  std::cout << std::setw(12) << "|V|+|E|"
            << std::setw(12) << "OpenMP"
            << std::setw(12) << "TBB"
            << std::setw(12) << "Taskflow"
            << std::setw(12) << "speedup1"
            << std::setw(12) << "speedup2"
            << '\n';

  for(int i=1; i<=451; i += 15) {

    LevelGraph graph(i, i);
    
    for(int j=0; j<rounds; ++j) {
      omp_time += measure_time_omp(graph, num_threads).count();
      graph.clear_graph();

      tbb_time += measure_time_tbb(graph, num_threads).count();
      graph.clear_graph();

      tf_time  += measure_time_taskflow(graph, num_threads).count();
      graph.clear_graph();
    }

    std::cout << std::setw(12) << graph.graph_size() 
              << std::setw(12) << omp_time / rounds / 1e6
              << std::setw(12) << tbb_time / rounds / 1e6
              << std::setw(12) << tf_time  / rounds / 1e6
              << std::setw(12) << omp_time / tf_time
              << std::setw(12) << tbb_time / tf_time
              << std::endl;
  }
}


