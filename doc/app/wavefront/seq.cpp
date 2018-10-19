#include <algorithm> // for std::max
#include <cstdio>
#include <chrono>
#include <iostream>
#include <thread>
#include <random>
#include <cmath>

int M=40000, N=40000;
int B = 160;
int MB = (M/B) + (M%B>0);
int NB = (N/B) + (N%B>0);

double **value;

inline double calc(double v0, double v1) {
  if(v0 == v1){
    return std::pow(v0/v1, 4.0f);
  }
  else
    return std::max(v0,v1);
}

void init_data(){
  value = new double *[M];
  for ( int i = 0; i < M; ++i ) value[i] = new double [N];
  for(int i=0; i<M; ++i){
    for(int j=0; j<N ; ++j){
      value[i][j] = i*N + j;
    }   
  }
}

int main(int argc, char *argv[]) {

  init_data();

  auto beg = std::chrono::high_resolution_clock::now();
  {
    for( int i=0; i<MB; ++i){
      for( int j=0; j<NB; ++j) {
        int start_i = i*B;
        int end_i = (i*B+B > M) ? M : i*B+B;
        int start_j = j*B;
        int end_j = (j*B+B > N) ? N : j*B+B;
        for ( int ii = start_i; ii < end_i; ++ii ) {
          for ( int jj = start_j; jj < end_j; ++jj ) {
            double v0 = (ii == 0) ? 0 : value[ii-1][jj];
            double v1 = (jj == 0) ? 0 : value[ii][jj-1];
            value[ii][jj] = (ii==0 && jj==0) ? 1 : calc(v0,v1);
          }
        }
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Seq: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count() << " ms, Result = "
            << value[M-1][N-1] << '\n';

  for ( int i = 0; i < M; ++i ) delete [] value[i];
  delete [] value;
  return 0;
}

