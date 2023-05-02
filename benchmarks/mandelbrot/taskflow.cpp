#include "mandel.hpp"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

void mandelbrot_taskflow(unsigned num_threads, int d = D) {

  tf::Executor executor {num_threads};
  tf::Taskflow taskflow;

  taskflow.for_each_index(0, H, 1, [&](int i){
    for(int j=0; j<W; j++) {
      auto xy = scale_xy(i, j);
      auto value = escape_time(xy.first, xy.second, d);
      auto k = 3 * ( j * W + i );
      std::tie(RGB[k], RGB[k+1], RGB[k+2]) = get_color(value);
    }
  });

  executor.run(taskflow).wait();
}

std::chrono::microseconds measure_time_taskflow(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  mandelbrot_taskflow(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

