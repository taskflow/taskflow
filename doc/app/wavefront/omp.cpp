#include <cstdio>
#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <tuple>
#include <chrono>
#include <cmath>

#include <omp.h>

int M= 40000, N=40000;
int B = 160;
int MB = (M/B) + (M%B>0);
int NB = (N/B) + (N%B>0);

int **D;       //dependency matrix

double **value;

inline double calc(double v0, double v1) {
  if(v0 == v1)
    return std::pow(v0/v1, 4.0f);
  else
    return std::max(v0,v1);
}

//computation given block row index i, block col index j
inline void block_computation(int i, int j){

  int start_i = i*B;
  int end_i = (i*B+B > M) ? M : i*B+B;
  int start_j = j*B;
  int end_j = (j*B+B > N) ? N : j*B+B;

  for ( int ii = start_i; ii < end_i; ++ii ) {
    for ( int jj = start_j; jj < end_j; ++jj ) {
      double v0 = ii == 0 ? 0 : value[ii-1][jj];
      double v1 = jj == 0 ? 0 : value[ii][jj-1];
      value[ii][jj] = ii==0 && jj==0 ? 1 : calc(v0,v1);
    }
  }

}


void RunGraph() {
  
  omp_set_num_threads(4); 

  #pragma omp parallel
  {
    #pragma omp single
    {
	    value[M-1][N-1] = 0;
	    for( int k=1; k <= 2*MB-1; k++) {
        int i, j;
        if(k <= MB){
          i = k-1;
          j = 0;
        }
        else{
          //assume matrix is square
          i = MB-1;
          j = k-MB;
        }       
        
        for(; (k <= MB && i>=0) || (k > MB && j <= NB-1) ; i--, j++){

          if(i > 0 && j > 0){
            #pragma omp task depend(in:D[i-1][j], D[i][j-1]) depend(out:D[i][j]) firstprivate(i, j)
              block_computation(i, j); 
          }
          //top left corner
          else if(i == 0 && j == 0){
            #pragma omp task depend(out:D[i][j]) firstprivate(i, j)
              block_computation(i, j); 
          }	
          //top edge	
          else if(j+1 <= NB && i == 0 && j > 0){
            #pragma omp task depend(in:D[i][j-1]) depend(out:D[i][j]) firstprivate(i, j)
              block_computation(i, j); 
          }
          //left edge
          else if(i+1 <= MB && i > 0 && j == 0){
            #pragma omp task depend(in:D[i-1][j]) depend(out:D[i][j]) firstprivate(i, j)
              block_computation(i, j); 
          }
          //bottom right corner
          else if(i == MB-1 && j == NB-1){
            #pragma omp task depend(in:D[i-1][j] ,D[i][j-1]) firstprivate(i, j)
              block_computation(i, j); 
          }
          else{
            std::cout << "There is some case not covered!!!" << std::endl;
          }
                

        }

       // #pragma omp taskwait 

	    }
    }
  }
}

int main(int argc, char *argv[]) {

  D = new int *[MB];
  for(int i=0; i<MB; ++i) D[i] = new int [NB];
  for(int i=0; i<MB; ++i){
    for(int j=0; j<NB; ++j){
      D[i][j] = 0;
    }
  }

  value = new double *[M];
  for(int i = 0; i<M; ++i) value[i] = new double [N];
	for(int i=0; i<M; ++i){
		for(int j=0; j<N ; ++j){
			value[i][j] = i*N + j;
		}
	}

  auto beg = std::chrono::high_resolution_clock::now();
	
  RunGraph();

  auto end = std::chrono::high_resolution_clock::now();
	std::cout << "OMP: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count() << " ms, ";
  std::cout << "The result is:" << value[M-1][N-1] << std::endl;

  for ( int i = 0; i < M; ++i ) delete [] value[i];
  delete [] value;

  for ( int i = 0; i < MB; ++i ) delete [] D[i];
  delete [] D;

  return 0;
}




