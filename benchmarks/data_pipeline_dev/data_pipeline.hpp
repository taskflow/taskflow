#include <chrono>
#include <string>
#include <cassert>
#include <thread>
#include <cmath>
#include <time.h>

inline void work(int& input) {
  // srand(time(NULL));
  for (int i = 0; i < 10; i++) {

    input = cos(input) * input + 1;
    input = (int)std::pow(input, 5) % 2147483647;

    // if (rand() % 2 == 0) {
    //   input = cos(input) * input + 1;
    //   input = std::pow(input, 5);
    // } else {
    //   input = sin(input) * input + 1;
    //   input = std::pow(input, 3);
    // }
  }
  
}

std::chrono::microseconds measure_time_normal(std::string, unsigned, unsigned, size_t);
std::chrono::microseconds measure_time_efficient(std::string, unsigned, unsigned, size_t);

