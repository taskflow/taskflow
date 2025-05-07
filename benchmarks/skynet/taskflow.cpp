// The skynet benchmark as described here:
// https://github.com/atemerev/skynet
#include <taskflow/taskflow.hpp>
#include "skynet.hpp"

size_t skynet_one_tf(tf::Runtime& rt, size_t BaseNum, size_t Depth, size_t MaxDepth) {

  if (Depth == MaxDepth) {
    return BaseNum;
  }

  size_t depthOffset = 1;
  for (size_t i = 0; i < MaxDepth - Depth - 1; ++i) {
    depthOffset *= 10;
  }

  std::array<size_t, 10> results;

  for (size_t i = 0; i < 10; ++i) {
    rt.silent_async([=, &results](tf::Runtime& s) {
      results[i] = skynet_one_tf(s, BaseNum + depthOffset * i, Depth + 1, MaxDepth);
    });
  }
  rt.corun_all();

  size_t count = 0;
  for (size_t idx = 0; idx < 10; ++idx) {
    count += results[idx];
  }
  return count;
}

void skynet(size_t num_threads, size_t MaxDepth) {
  static tf::Executor executor(num_threads);
  executor.async([=](tf::Runtime& rt) { skynet_one_tf(rt, 0, 0, MaxDepth); }).wait();
}

std::chrono::microseconds measure_time_taskflow(size_t num_threads, size_t MaxDepth) {
  auto beg = std::chrono::high_resolution_clock::now();
  skynet(num_threads, MaxDepth);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


