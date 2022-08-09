#include <chrono>
#include <string>
#include <cassert>
#include <thread>

inline void work() {
  std::this_thread::sleep_for(std::chrono::microseconds(10));
}

std::chrono::microseconds measure_time_taskflow(std::string, unsigned, unsigned, size_t);
std::chrono::microseconds measure_time_tbb(std::string, unsigned, unsigned, size_t);
std::chrono::microseconds measure_time_omp(std::string, unsigned, unsigned, size_t);

