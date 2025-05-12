#include <omp.h>
#include "skynet.hpp"

size_t skynet_one_omp(size_t BaseNum, size_t Depth, size_t MaxDepth) {

  if (Depth == MaxDepth) {
    return BaseNum;
  }

  size_t depthOffset = 1;
  for (size_t i = 0; i < MaxDepth - Depth - 1; ++i) {
    depthOffset *= 10;
  }

  std::array<size_t, 10> results = {};
  
  for (size_t i = 0; i < 10; ++i) {
    #pragma omp task firstprivate(i) shared(results)
    {
      results[i] = skynet_one_omp(BaseNum + depthOffset * i, Depth + 1, MaxDepth);
    };
  }
  
  #pragma omp taskwait

  size_t count = 0;
  for (size_t idx = 0; idx < 10; ++idx) {
    count += results[idx];
  }
  return count;
}

std::chrono::microseconds measure_time_omp(size_t num_threads, size_t MaxDepth) {
  auto beg = std::chrono::high_resolution_clock::now();
  omp_set_num_threads(num_threads);
  #pragma omp parallel
  {
    #pragma omp single
    skynet_one_omp(0, 0, MaxDepth);
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


