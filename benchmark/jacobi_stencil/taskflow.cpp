#include "matrix.hpp"
#include <taskflow/taskflow.hpp> 

// wavefront computing
void wavefront_taskflow(unsigned num_threads, unsigned num_runs) {

  tf::Taskflow tf{num_threads};

  tf::Framework F;
  std::vector<std::vector<tf::Task>> node(MB);

  for(auto &n : node){
    for(int i=0; i<NB; i++){
      n.emplace_back(F.placeholder());
    }
  }
  
  matrix[M-1][N-1] = 0;
  for( int i=MB; --i>=0; ) {
    for( int j=NB; --j>=0; ) {
      node[i][j].work(
        [=]() {
          jacobi_stencil(i, j);
        }
      );

      if(j+1 < NB) node[i][j].precede(node[i][j+1]);
      if(i+1 < MB) node[i][j].precede(node[i+1][j]);
    }
  }

  tf.run_n(F, num_runs).get();
}

std::chrono::microseconds measure_time_taskflow(unsigned num_threads, unsigned num_runs) {
  auto beg = std::chrono::high_resolution_clock::now();
  wavefront_taskflow(num_threads, num_runs);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
}

/*// main function
int main(int argc, char *argv[]) {

  init_matrix();

  auto beg = std::chrono::high_resolution_clock::now();
  wavefront();
  auto end = std::chrono::high_resolution_clock::now();
  
  std::cout << "Taskflow wavefront elapsed time: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count() 
            << " ms\n"
            << "result: " << matrix[M-1][N-1] << std::endl;

  destroy_matrix();

  return 0;
}
*/



