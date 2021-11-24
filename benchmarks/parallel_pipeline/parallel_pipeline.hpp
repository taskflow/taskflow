#include <chrono>
#include <string>
#include <cassert>

std::chrono::microseconds measure_time_taskflow(std::string, unsigned, unsigned, size_t);
std::chrono::microseconds measure_time_tbb(std::string, unsigned, unsigned, size_t);
std::chrono::microseconds measure_time_omp(std::string, unsigned, unsigned, size_t);

