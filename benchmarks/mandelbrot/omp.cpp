#include "mandel.hpp"
#include <omp.h>

void mandelbrot_omp(unsigned num_threads, int d = D) {
  omp_set_num_threads(num_threads);
  int i,j ;

  # pragma omp parallel shared (d) private (i, j)
  {
    # pragma omp for schedule(dynamic, 1)
    for(i=0; i<H ;i ++) {
      for(j=0; j<W; j++) {

        auto xy = scale_xy(i, j);
        auto value = escape_time(xy.first, xy.second, d);
        auto k = 3*(j*W + i);
        std::tie(RGB[k], RGB[k+1], RGB[k+2]) = get_color(value);
      }
    }
  }
}

std::chrono::microseconds measure_time_omp(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  mandelbrot_omp(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
