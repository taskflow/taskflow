#include "mandel.hpp"
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

void mandelbrot_tbb(unsigned num_threads, int d = D) {
  tbb::task_scheduler_init init(num_threads);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, H), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i=r.begin();i!=r.end();++i) {
      for(int j=0; j<W; j++) {

        auto [xx, yy] = scale_xy(i, j);
        auto value = escape_time(xx, yy, d);

        auto k = 3*(j*W + i);

        auto [r, g, b] = get_color(value);

        RGB[k]   = r;
        RGB[k+1] = g;
        RGB[k+2] = b;
      }
    }
  });
}

std::chrono::microseconds measure_time_tbb(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  mandelbrot_tbb(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
