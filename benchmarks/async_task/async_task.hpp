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

inline void func() {}

std::chrono::microseconds measure_time_taskflow(unsigned, size_t);
std::chrono::microseconds measure_time_omp(unsigned, size_t);
std::chrono::microseconds measure_time_std(unsigned, size_t);
