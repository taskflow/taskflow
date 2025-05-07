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


inline constexpr size_t primes_limit = 10000000;
inline constexpr size_t primes_chunk = 10;

/**
 * @brief See https://en.wikipedia.org/wiki/Primality_test
 */
inline auto is_prime = [](size_t n) -> bool {
  //
  if (n == 2 || n == 3) {
    return true;
  }

  if (n <= 1 || n % 2 == 0 || n % 3 == 0) {
    return false;
  }

  for (int i = 5; i * i <= n; i += 6) {
    if (n % i == 0 || n % (i + 2) == 0) {
      return false;
    }
  }

  return true;
};

