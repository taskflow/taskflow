#include "parallel_pipeline.hpp"
#include <omp.h>

// linear_chain_omp
void linear_chain_omp(size_t length, unsigned num_threads) {

  size_t counter = 0;
  size_t *D = new size_t [length]; 

  #pragma omp parallel num_threads(num_threads)
  {
    #pragma omp single
    {
      for(size_t i=0; i<length; ++i) {

        if(i==0) {
          #pragma omp task firstprivate(i) depend(out:D[i])
          {
            ++counter;
          }
        }
        else {
          #pragma omp task firstprivate(i) depend(out:D[i]) depend(in:D[i-1])
          {
            ++counter;
          }
        }
      }
    }
  }

  assert(counter == length);

  delete [] D;
}

std::chrono::microseconds measure_time_omp(
  std::string pipes,
  unsigned length,
  unsigned num_threads,
  size_t size
) {
  auto beg = std::chrono::high_resolution_clock::now();
  //linear_chain_omp(length, num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


