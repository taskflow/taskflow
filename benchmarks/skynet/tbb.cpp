// The skynet benchmark as described here:
// https://github.com/atemerev/skynet

#include <tbb/tbb.h>

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include "skynet.hpp"

template <size_t DepthMax>
size_t skynet_one(size_t BaseNum, size_t Depth) {

  if (Depth == DepthMax) {
    return BaseNum;
  }

  size_t depthOffset = 1;
  for (size_t i = 0; i < DepthMax - Depth - 1; ++i) {
    depthOffset *= 10;
  }

  std::array<size_t, 10> results;

  tbb::task_group tg;
  for (size_t i = 0; i < 10; ++i) {
    tg.run([=, &results, idx = i]() {
      results[idx] =
        skynet_one<DepthMax>(BaseNum + depthOffset * idx, Depth + 1);
    });
  }
  tg.wait();

  size_t count = 0;
  for (size_t idx = 0; idx < 10; ++idx) {
    count += results[idx];
  }
  return count;
}


template <size_t Depth = 8>
void loop_skynet() {
  skynet_one<Depth>(0,0);
}


std::chrono::microseconds measure_time_tbb(size_t num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  tbb::task_arena arena(num_threads);
  arena.execute(loop_skynet<8>);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


