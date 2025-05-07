#include <algorithm>
#include <vector>
#include <exception>
#include <iostream>
#include <numeric>
#include <taskflow/taskflow.hpp>
#include "integrate.hpp"


tf::Executor& get_executor() {
  static tf::Executor executor;
  return executor;
}

// integrate computation 
auto spawn_async(double x1, double y1, double x2, double y2, double area, tf::Runtime& rt) {
  double half = (x2 - x1) / 2.0;
  double x0 = x1 + half;
  double y0 = fn(x0);

  double area_x1x0 = (y1 + y0) / 2 * half;
  double area_x0x2 = (y0 + y2) / 2 * half;
  double area_x1x2 = area_x1x0 + area_x0x2;
  
  if (area_x1x2 - area < epsilon && area - area_x1x2 < epsilon) {
    return area_x1x2;
  }

  rt.silent_async([x1,y1,x0,y0,&area_x1x0](tf::Runtime& rt1){
    area_x1x0 = spawn_async(x1, y1, x0, y0, area_x1x0, rt1);
  });
  
  area_x0x2 = spawn_async(x0, y0, x2, y2, area_x0x2, rt);

  // use corun to avoid blocking the worker from waiting the two children tasks to finish
  rt.corun();

  return area_x1x0 + area_x0x2;
}


auto integrate_taskflow(size_t num_threads, double x1, double y1, double x2, double y2) {

  double area = 0.0;
  static tf::Executor executor(num_threads);

  executor.async([x1, y1, x2, y2, &area](tf::Runtime& rt){
    area = spawn_async(x1, y1, x2, y2, 0.0, rt);
  }).get();

  return area;
}

std::chrono::microseconds measure_time_taskflow(size_t num_threads, size_t max_value) {

  auto beg = std::chrono::high_resolution_clock::now();
  integrate_taskflow(num_threads, 0, fn(0), max_value, fn(max_value));
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
