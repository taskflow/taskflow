// The skynet benchmark as described here:
// https://github.com/atemerev/skynet

#include <tbb/task_group.h>
#include <tbb/global_control.h>
#include "skynet.hpp"

size_t skynet_one_tbb(size_t BaseNum, size_t Depth, size_t MaxDepth) {

  if (Depth == MaxDepth) {
    return BaseNum;
  }

  size_t depthOffset = 1;
  for (size_t i = 0; i < MaxDepth - Depth - 1; ++i) {
    depthOffset *= 10;
  }

  std::array<size_t, 10> results;

  tbb::task_group tg;
  for (size_t i = 0; i < 10; ++i) {
    tg.run([=, &results]() {
      results[i] = skynet_one_tbb(BaseNum + depthOffset * i, Depth + 1, MaxDepth);
    });
  }
  tg.wait();

  size_t count = 0;
  for (size_t idx = 0; idx < 10; ++idx) {
    count += results[idx];
  }
  return count;
}

std::chrono::microseconds measure_time_tbb(size_t num_threads, size_t MaxDepth) {
  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_threads
  );
  auto beg = std::chrono::high_resolution_clock::now();
  skynet_one_tbb(0, 0, MaxDepth);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


