#pragma once

#include <algorithm> // for std::max
#include <cassert>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>
#include <random>
#include <cmath>
#include <vector>

std::chrono::microseconds measure_time_taskflow(size_t, size_t);
std::chrono::microseconds measure_time_omp(size_t, size_t);
std::chrono::microseconds measure_time_tbb(size_t, size_t);

inline constexpr int n = 10000;
inline constexpr double epsilon = 1.0e-9;

inline constexpr auto fn(double x) -> double { return (x * x + 1.0) * x; }

inline constexpr auto integral_fn(double a, double b) -> double {

  constexpr auto indefinite = [](double x) {
    return 0.25 * x * x * (x * x + 2);
  };

  return indefinite(b) - indefinite(a);
}

