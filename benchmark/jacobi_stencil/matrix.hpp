#include <algorithm> // for std::max
#include <cassert>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>
#include <random>
#include <cmath>

// https://www.nersc.gov/users/software/programming-models/openmp/openmp-tasking/openmp-tasking-example-jacobi/

inline int M, N;
inline int B;
inline int MB;
inline int NB;

inline double **matrix {nullptr};


// initialize the matrix
inline void init_matrix(){
  matrix = new double *[M];
  for ( int i = 0; i < M; ++i ) matrix[i] = new double [N];
  for(int i=0; i<M; ++i){
    for(int j=0; j<N ; ++j){
      matrix[i][j] = i*N + j;
    }   
  }
}

// destroy the matrix
inline void destroy_matrix() {
  for ( int i = 0; i < M; ++i ) {
    delete [] matrix[i];
  }
  delete [] matrix;
}

//computation given block row index i, block col index j
inline void jacobi_stencil(int i, int j){

  int start_i = i*B;
  int end_i = (i*B+B > M) ? M : i*B+B;
  int start_j = j*B;
  int end_j = (j*B+B > N) ? N : j*B+B;

  for ( int ii = start_i; ii < end_i; ++ii ) {
    for ( int jj = start_j; jj < end_j; ++jj ) {
      matrix[ii][jj] +=  ii == 0   ? 0.0 : matrix[ii-1][jj];
      matrix[ii][jj] +=  ii >= M-1 ? 0.0 : matrix[ii+1][jj];
      matrix[ii][jj] +=  jj == 0   ? 0.0 : matrix[ii][jj-1];
      matrix[ii][jj] +=  jj >= N-1 ? 0.0 : matrix[ii][jj+1];
      matrix[ii][jj] *= 0.25;
    }
  }
}


std::chrono::microseconds measure_time_taskflow(unsigned, unsigned=1);
std::chrono::microseconds measure_time_omp(unsigned, unsigned=1);
std::chrono::microseconds measure_time_tbb(unsigned, unsigned=1);


