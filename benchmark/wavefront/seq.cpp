#include "matrix.hpp"

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

