#include "mandel.hpp"
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

void mandelbrot_tbb(unsigned num_threads, int d = D) {
  
  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_threads
  );

  tbb::parallel_for(0, H, 1, [&](int i) {
    for(int j=0; j<W; j++) {
      auto xy = scale_xy(i, j);
      auto value = escape_time(xy.first, xy.second, d);
      auto k = 3 * ( j * W + i );
      std::tie(RGB[k], RGB[k+1], RGB[k+2]) = get_color(value);
    }
  });
}

std::chrono::microseconds measure_time_tbb(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  mandelbrot_tbb(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
