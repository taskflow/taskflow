// The skynet benchmark as described here:
// https://github.com/atemerev/skynet


#include <taskflow/taskflow.hpp>
#include "skynet.hpp"


template <size_t DepthMax>
size_t skynet_one(tf::Runtime& rt, size_t BaseNum, size_t Depth) {
  if (Depth == DepthMax) {
    return BaseNum;
  }
  size_t depthOffset = 1;
  for (size_t i = 0; i < DepthMax - Depth - 1; ++i) {
    depthOffset *= 10;
  }

  std::array<size_t, 10> results;

  for (size_t i = 0; i < 10; ++i) {
    rt.silent_async([=, &results, idx = i](tf::Runtime& s) {
      results[idx] =
        skynet_one<DepthMax>(s, BaseNum + depthOffset * idx, Depth + 1);
    });
  }
  rt.corun_all();

  size_t count = 0;
  for (size_t idx = 0; idx < 10; ++idx) {
    count += results[idx];
  }
  return count;
}

template <size_t DepthMax>
void skynet(tf::Executor& executor) {

  tf::Taskflow taskflow;
  size_t count;

  taskflow.emplace([&](tf::Runtime& rt) {
    count = skynet_one<DepthMax>(rt, 0, 0);
  });

  executor.run(taskflow).wait();

}

std::chrono::microseconds measure_time_taskflow(size_t num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();

  tf::Executor executor(num_threads);
  
  skynet<8>(executor);
  
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


