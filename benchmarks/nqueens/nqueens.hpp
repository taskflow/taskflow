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

inline std::vector<unsigned long long int> answers{0,1,0,0,2,10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512,95815104,666090624};

inline auto queens_ok(int n, char* a) -> bool {

  for (int i = 0; i < n; i++) {

    char p = a[i];

    for (int j = i + 1; j < n; j++) {
      if (char q = a[j]; q == p || q == p - (j - i) || q == p + (j - i)) {
        return false;
      }
    }
  }
  return true;
}
