#include <algorithm> 
#include <cassert>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>
#include <random>
#include <cmath>
#include <atomic>

std::chrono::microseconds measure_time_taskflow(size_t, unsigned);
std::chrono::microseconds measure_time_tbb(size_t, unsigned);
std::chrono::microseconds measure_time_omp(size_t, unsigned);

