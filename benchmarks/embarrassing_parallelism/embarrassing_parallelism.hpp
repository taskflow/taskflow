#include <chrono>
#include <cmath>
#include <cassert>

std::chrono::microseconds measure_time_taskflow(unsigned, size_t);
std::chrono::microseconds measure_time_omp(unsigned, size_t);
std::chrono::microseconds measure_time_tbb(unsigned, size_t);
std::chrono::microseconds measure_time_seq(size_t);

inline double dummy(size_t i) {
  volatile double sink = 0.0;

  // Mix i so the compiler cannot constant-fold
  double x = static_cast<double>((i * 1315423911u) ^ (i >> 3)) + 1.0;

  // Do a small but non-trivial amount of FP work
  for(int k = 0; k < 32; ++k) {
    x = std::sin(x) * 1.0000001 + std::cos(x);
    sink = sink + x;
  }

  return sink;
}
