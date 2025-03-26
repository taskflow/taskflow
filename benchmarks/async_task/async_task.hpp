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

inline void func(std::atomic<size_t>& counter) { counter.fetch_add(1, std::memory_order_relaxed); }

std::chrono::microseconds measure_time_taskflow(unsigned, size_t);
std::chrono::microseconds measure_time_omp(unsigned, size_t);
std::chrono::microseconds measure_time_tbb(unsigned, size_t);
std::chrono::microseconds measure_time_std(unsigned, size_t);
