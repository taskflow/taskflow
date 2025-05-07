#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>
#include "integrate.hpp"

auto integrate_omp(double x1, double y1, double x2, double y2, double area) -> double {

  double half = (x2 - x1) / 2;
  double x0 = x1 + half;
  double y0 = fn(x0);

  double area_x1x0 = (y1 + y0) / 2 * half;
  double area_x0x2 = (y0 + y2) / 2 * half;
  double area_x1x2 = area_x1x0 + area_x0x2;

  if (area_x1x2 - area < epsilon && area - area_x1x2 < epsilon) {
    return area_x1x2;
  }

  #pragma omp task
  area_x1x0 = integrate_omp(x1, y1, x0, y0, area_x1x0);

  #pragma omp task
  area_x0x2 = integrate_omp(x0, y0, x2, y2, area_x0x2);

  #pragma omp taskwait

  return area_x1x0 + area_x0x2;
}


std::chrono::microseconds measure_time_omp(size_t num_threads, size_t max_value) {

  auto beg = std::chrono::high_resolution_clock::now();
 
  #pragma omp parallel num_threads(num_threads)
  {
    #pragma omp single
    {
      integrate_omp(0, fn(0), max_value, fn(max_value), 0.0);
    }
  }
  
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
