#include <algorithm> // for std::max
#include <cassert>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>
#include <random>
#include <cmath>

extern int M, N;
extern int B;
extern int MB;
extern int NB;
extern double **matrix;

// nominal operations
inline double calc(double v0, double v1) {
  return (v0 == v1) ? std::pow(v0/v1, 4.0f) : std::max(v0,v1);
  //return std::max(v0, v1);
}

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
inline int block_computation(int i, int j){
  // When testing taskflow
  return i + j;

  //int start_i = i*B;
  //int end_i = (i*B+B > M) ? M : i*B+B;
  //int start_j = j*B;
  //int end_j = (j*B+B > N) ? N : j*B+B;
  //for ( int ii = start_i; ii < end_i; ++ii ) {
  //  for ( int jj = start_j; jj < end_j; ++jj ) {
  //    double v0 = ii == 0 ? 0 : matrix[ii-1][jj];
  //    double v1 = jj == 0 ? 0 : matrix[ii][jj-1];
  //    matrix[ii][jj] = ii==0 && jj==0 ? 1 : calc(v0,v1);
  //  }
  //}
}


//computation given block row index i, block col index j
inline void framework_computation(int i, int j){
  // When testing framework 
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
      //matrix[ii][jj] = matrix[ii][jj];
    }   
  }
}



std::chrono::microseconds measure_time_taskflow(unsigned);
std::chrono::microseconds measure_time_omp(unsigned);
std::chrono::microseconds measure_time_tbb(unsigned);

std::chrono::microseconds measure_time_taskflow(unsigned, unsigned);
std::chrono::microseconds measure_time_omp(unsigned, unsigned);
std::chrono::microseconds measure_time_tbb(unsigned, unsigned);

