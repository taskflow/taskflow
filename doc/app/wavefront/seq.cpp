#include "matrix.hpp"

// the computation inside each block
void block_computation(int i, int j){

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

// wavefront computation
void wavefront_seq() {
  for( int i=0; i<MB; ++i){
    for( int j=0; j<NB; ++j) {
      block_computation(i, j);
    }
  }
}

std::chrono::microseconds measure_time_seq() {
  auto beg = std::chrono::high_resolution_clock::now();
  wavefront_seq();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
}

/*// main function
int main(int argc, char *argv[]) {

  init_matrix();

  auto beg = std::chrono::high_resolution_clock::now();
  wavefront();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "sequential program elapsed time: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count() 
            << " ms\n"
            << "result: " << matrix[M-1][N-1] << '\n';

  destroy_matrix();

  return 0;
} */

