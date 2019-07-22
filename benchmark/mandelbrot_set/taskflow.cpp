#include "mandel.hpp"
#include <taskflow/taskflow.hpp> 

//void run_taskflow_parallel_for(int num_threads, int d = D) {
//
//  tf::Taskflow flow;
//
//  flow.parallel_for(0, h, 1, [&](int i){
//    for(int j=0; j<w; j++) {
//
//      auto [xx, yy] = scale_xy(i, j);
//      auto value = escape_time(xx, yy, d);
//
//      auto k = 3 * ( j * w + i );
//
//      auto [r, g, b] = get_color(value);
//
//      rgb[k]   = r;
//      rgb[k+1] = g;
//      rgb[k+2] = b;
//    }
//  });
//
//  tf::Executor executor {num_threads};
//  executor.run(flow).wait();
//} 


void mandelbrot_taskflow(unsigned num_threads, int d = D) {

  tf::Taskflow flow;

  for(int i=0; i<H; i++) {
    flow.emplace([&, i=i](){
      for(int j=0; j<W; j++) {

        auto [xx, yy] = scale_xy(i, j);
        auto value = escape_time(xx, yy, d);

        auto k = 3*(j*W + i);

        auto [r, g, b] = get_color(value);

        RGB[k]   = r;
        RGB[k+1] = g;
        RGB[k+2] = b;
      }
    });
  }

  tf::Executor executor {num_threads};
  executor.run(flow).wait();
}

std::chrono::microseconds measure_time_taskflow(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  mandelbrot_taskflow(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

