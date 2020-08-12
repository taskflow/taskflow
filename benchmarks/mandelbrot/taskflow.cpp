#include "mandel.hpp"
#include <taskflow/taskflow.hpp> 

void mandelbrot_taskflow(unsigned num_threads, int d = D) {

  tf::Executor executor {num_threads};
  tf::Taskflow flow;

  flow.parallel_for_dynamic(0, H, 1, [&](int i){
    for(int j=0; j<W; j++) {

      auto [xx, yy] = scale_xy(i, j);
      auto value = escape_time(xx, yy, d);

      auto k = 3 * ( j * W + i );

      auto [r, g, b] = get_color(value);

      RGB[k]   = r;
      RGB[k+1] = g;
      RGB[k+2] = b;
    }
  }, 1);

  executor.run(flow).wait();
} 


//void mandelbrot_taskflow(unsigned num_threads, int d = D) {
//
//  tf::Taskflow flow;
//
//  for(int i=0; i<H; i++) {
//    flow.emplace([&, i=i](){
//      for(int j=0; j<W; j++) {
//
//        auto [xx, yy] = scale_xy(i, j);
//        auto value = escape_time(xx, yy, d);
//
//        auto k = 3*(j*W + i);
//
//        auto [r, g, b] = get_color(value);
//
//        RGB[k]   = r;
//        RGB[k+1] = g;
//        RGB[k+2] = b;
//      }
//    });
//  }
//
//  tf::Executor executor {num_threads};
//  executor.run(flow).wait();
//}

std::chrono::microseconds measure_time_taskflow(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  mandelbrot_taskflow(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

