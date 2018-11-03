#include "matrix.hpp"
#include <taskflow/taskflow.hpp> 

// wavefront computing
void wavefront_taskflow() {

  tf::Taskflow tf;

  std::vector<std::vector<tf::Task>> node(MB);

  for(auto &n : node){
    for(size_t i=0; i<NB; i++){
      n.emplace_back(tf.placeholder());
    }
  }
  
  matrix[M-1][N-1] = 0;
  for( int i=MB; --i>=0; ) {
    for( int j=NB; --j>=0; ) {
      node[i][j].work(
        [i=i, j=j]() {
          int start_i = i*B;
          int end_i = (i*B+B > M) ? M : i*B+B;
          int start_j = j*B;
          int end_j = (j*B+B > N) ? N : j*B+B;
          for ( int ii = start_i; ii < end_i; ++ii ) {
            for ( int jj = start_j; jj < end_j; ++jj ) {
              double v0 = ii == 0 ? 0 : matrix[ii-1][jj];
              double v1 = jj == 0 ? 0 : matrix[ii][jj-1];
              matrix[ii][jj] = ii==0 && jj==0 ? 1 : calc(v0,v1);
            }
          }
        }
      );
      if(j+1 < NB) node[i][j].precede(node[i][j+1]);
      if(i+1 < MB) node[i][j].precede(node[i+1][j]);
    }
  }

  tf.wait_for_all();
}

std::chrono::microseconds measure_time_taskflow() {
  auto beg = std::chrono::high_resolution_clock::now();
  wavefront_taskflow();
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



