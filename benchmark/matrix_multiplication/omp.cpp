#include "matrix_multiplication.hpp"
#include <omp.h>

// matrix_multiplication_omp
// reference: https://computing.llnl.gov/tutorials/openMP/samples/C/omp_mm.c
void matrix_multiplication_omp(unsigned nthreads) {
  
  omp_set_num_threads(nthreads);

  int i, j, k;

  #pragma omp parallel shared(a, b, c, nthreads) private(i, j, k)
  {

    #pragma omp for schedule (static)
    for (i=0; i<N; i++)
      for (j=0; j<N; j++)
        a[i][j]= i+j;

    #pragma omp for schedule (static)
    for (i=0; i<N; i++)
      for (j=0; j<N; j++)
        b[i][j]= i*j;

    #pragma omp for schedule (static)
    for (i=0; i<N; i++)
      for (j=0; j<N; j++)
        c[i][j]= 0;

    #pragma omp for schedule (static)
    for (i=0; i<N; i++) {
      for(j=0; j<N; j++) {
        for (k=0; k<N; k++) {
          c[i][j] += a[i][k] * b[k][j];
        }
      }
    }
  }
  
  //std::cout << reduce_sum() << std::endl;
}

std::chrono::microseconds measure_time_omp(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  matrix_multiplication_omp(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
