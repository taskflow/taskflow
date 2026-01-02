#include <algorithm>
#include <vector>
#include <exception>
#include <iostream>
#include <numeric>
#include <taskflow/taskflow.hpp>
#include "integrate.hpp"


// integrate computation 
auto spawn_async(double x1, double y1, double x2, double y2, double area, tf::Executor& exe) {
  double half = (x2 - x1) / 2.0;
  double x0 = x1 + half;
  double y0 = fn(x0);

  double area_x1x0 = (y1 + y0) / 2 * half;
  double area_x0x2 = (y0 + y2) / 2 * half;
  double area_x1x2 = area_x1x0 + area_x0x2;
  
  if (area_x1x2 - area < epsilon && area - area_x1x2 < epsilon) {
    return area_x1x2;
  }

  auto tg = exe.task_group();

  tg.silent_async([x1, y1, x0, y0, &area_x1x0, &exe](){
    area_x1x0 = spawn_async(x1, y1, x0, y0, area_x1x0, exe);
  });
  
  area_x0x2 = spawn_async(x0, y0, x2, y2, area_x0x2, exe);

  // use corun to avoid blocking the worker from waiting the two children tasks to finish
  tg.corun();

  return area_x1x0 + area_x0x2;
}


auto integrate_taskflow(size_t num_threads, double x1, double y1, double x2, double y2) {
  static tf::Executor executor(num_threads);
  return executor.async([x1, y1, x2, y2](){
    return spawn_async(x1, y1, x2, y2, 0.0, executor);
  }).get();
}

std::chrono::microseconds measure_time_taskflow(size_t num_threads, size_t max_value) {

  auto beg = std::chrono::high_resolution_clock::now();
  integrate_taskflow(num_threads, 0, fn(0), max_value, fn(max_value));
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
