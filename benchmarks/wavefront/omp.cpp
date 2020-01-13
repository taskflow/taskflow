#include "matrix.hpp"

#include <omp.h>

int **D {nullptr};

// wavefront computation
void wavefront_omp(unsigned num_threads) {
  
  // set up the dependency matrix
  D = new int *[MB];
  for(int i=0; i<MB; ++i) D[i] = new int [NB];
  for(int i=0; i<MB; ++i){
    for(int j=0; j<NB; ++j){
      D[i][j] = 0;
    }
  }
  
  omp_set_num_threads(num_threads);

  #pragma omp parallel
  {
    #pragma omp single
    {
      matrix[M-1][N-1] = 0;
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
            assert(false);
          }
        }
      }
      #pragma omp taskwait
    }
  }
  
  for ( int i = 0; i < MB; ++i ) delete [] D[i];
  delete [] D;
}

std::chrono::microseconds measure_time_omp(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  wavefront_omp(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
