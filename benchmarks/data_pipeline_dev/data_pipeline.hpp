#include <chrono>
#include <string>
#include <cassert>
#include <thread>
#include <cmath>
#include <time.h>
#include <algorithm>

inline void work_int(int& input) {
  for (int i = 0; i < 100; i++) {
    input = cos(input) * input + 1;
    input = (int)std::pow(input, 5) % 2147483647;
  }
}

inline void work_string(std::string& input) {
  for (int i = 0; i < 50; i++) {
    input = std::to_string(std::stoi(input) + 1);
  }
}

std::chrono::microseconds measure_time_normal(std::string, unsigned, unsigned, size_t, std::string);
std::chrono::microseconds measure_time_efficient(std::string, unsigned, unsigned, size_t, std::string);

