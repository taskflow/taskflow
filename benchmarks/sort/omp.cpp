#include "sort.hpp"
#include <omp.h>

// OMP parallel sort
// https://cw.fel.cvut.cz/old/_media/courses/b4m35pag/lab6_slides_advanced_openmp.pdf
template <typename V>
void mergeSortRecursive(V& v, size_t left, size_t right) {
  if (left < right) {
    if (right-left >= 32) {
      size_t mid = (left+right)/2;
      #pragma omp taskgroup
      {
        #pragma omp task shared(v) untied if(right-left >= (1<<14))
        mergeSortRecursive(v, left, mid);
        #pragma omp task shared(v) untied if(right-left >= (1<<14))
        mergeSortRecursive(v, mid+1, right);
        #pragma omp taskyield
      }
      std::inplace_merge(v.begin()+left, v.begin()+mid+1, v.begin()+right+1);
    } 
    else {
      std::sort(v.begin()+left, v.begin()+right+1);
    }
  }
}

// sort_omp
void sort_omp(size_t nthreads) {
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp single
    mergeSortRecursive(vec, 0, vec.size()-1);
  }
}

std::chrono::microseconds measure_time_omp(size_t num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  sort_omp(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

