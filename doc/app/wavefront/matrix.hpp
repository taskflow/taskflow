#include <algorithm> // for std::max
#include <cassert>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>
#include <random>
#include <cmath>

inline int M, N;
inline int B;
inline int MB;
inline int NB;

inline double **matrix {nullptr};

// nominal operations
inline double calc(double v0, double v1) {
  //return (v0 == v1) ? std::pow(v0/v1, 4.0f) : std::max(v0,v1);
  return std::max(v0, v1);
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


std::chrono::microseconds measure_time_taskflow();
std::chrono::microseconds measure_time_omp();
std::chrono::microseconds measure_time_tbb();


