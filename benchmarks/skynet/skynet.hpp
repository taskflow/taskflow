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
#include <array>

std::chrono::microseconds measure_time_taskflow(size_t, size_t);
std::chrono::microseconds measure_time_tbb(size_t, size_t);
std::chrono::microseconds measure_time_omp(size_t, size_t);


