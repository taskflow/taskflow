#include <chrono>
#include <string>
#include <cassert>
#include <thread>
#include <algorithm>
#include <cmath>
#include <vector>

// inline void work() {
//   std::this_thread::sleep_for(std::chrono::microseconds(10));
// }

inline void work_int(int& input) {
  for (int i = 0; i < 100; i++) {
    input = cos(input) * input + 1;
    input = (int)std::pow(input, 5) % 2147483647;
  }
}

inline void work_vector(std::vector<int>& input) {
  for (int i = 0; i < 100; i++) {
    input[0] = cos(input[0]) * input[0] + 1;
    input[0] = (int)std::pow(input[0], 5) % 2147483647;
  }
}

inline void work_float(float& input) {
  for (int i = 0; i < 100; i++) {
    input = cos(input) * input + 1;
    input = std::pow(input, 4);
  }
}

inline void work_string(std::string& input) {
  for (int i = 0; i < 50; i++) {
    input = std::to_string(std::stoi(input) + 1);
  }
}

std::chrono::microseconds measure_time_taskflow(std::string, unsigned, unsigned, size_t);
std::chrono::microseconds measure_time_tbb(std::string, unsigned, unsigned, size_t);

