#include "data_pipeline.hpp"
#include <tbb/pipeline.h>
#include <tbb/tick_count.h>
#include <tbb/tbb_allocator.h>
#include <tbb/global_control.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

namespace {

size_t i = 0;

// Filter for one filter only
class MyFunc {
public:
  MyFunc(size_t size) : s(size) {}

  ~MyFunc(){}

  size_t s;

  void operator()(tbb::flow_control& fc) const {
    int retval = 0;
    if (i++ == s) {
      fc.stop();
    }
    else {
      retval = retval + 1;
    }
  }
};

// Filter 1
class MyInputFunc {
public:
  MyInputFunc(size_t size) : s(size) {}

  ~MyInputFunc(){}

  size_t s;

  int operator()(tbb::flow_control& fc) const{
    int retval = 0;
    if (i++ == s) {
      fc.stop();
      return -1;
    }
    else {
      return retval + 1;
    }
  }
};

auto int2string = [](int input) -> std::string {
  work_int(input);
  return std::to_string(input);
};

auto string2int = [](std::string input) -> int {
  work_string(input);
  return std::stoi(input);
};

auto int2float = [](int input) -> float {
  work_int(input);
  return input * 1.0;
};

auto float2int = [](float input) -> int {
  work_float(input);
  return (int)input;
};

auto int2vector = [](int input) -> std::vector<int> {
  work_int(input);
  return std::vector<int>{input};
};

auto vector2int = [](std::vector<int> input) -> int {
  work_vector(input);
  return input[0];
};

auto int2int = [](int input) {
  work_int(input);
  return input;
};

auto int2void = [](int input) {  work_int(input); };

// parallel_pipeline_tbb_1_pipe
void parallel_pipeline_tbb_1_pipe(unsigned num_lines, size_t size) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, void>(
      tbb::filter::serial_in_order, MyFunc(size))
  );
}

// parallel_pipeline_tbb_2_pipes
void parallel_pipeline_tbb_2_pipes(std::string pipes, unsigned num_lines, size_t size) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(
      tbb::filter::serial_in_order, MyInputFunc(size))  &
    tbb::make_filter<int, void>(
      pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2void)
  );
}

// parallel_pipeline_tbb_3_pipes
void parallel_pipeline_tbb_3_pipes(std::string pipes, unsigned num_lines, size_t size) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(
      tbb::filter::serial_in_order, MyInputFunc(size))  &
    tbb::make_filter<int, int>(
      pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2int) &
    tbb::make_filter<int, void>(
      pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2void)
  );
}

// parallel_pipeline_tbb_4_pipes
void parallel_pipeline_tbb_4_pipes(std::string pipes, unsigned num_lines, size_t size) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(
      tbb::filter::serial_in_order, MyInputFunc(size))  &
    tbb::make_filter<int, float>(
      pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, void>(
      pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2void)
  );
}

// parallel_pipeline_tbb_5_pipes
void parallel_pipeline_tbb_5_pipes(std::string pipes, unsigned num_lines, size_t size) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(
      tbb::filter::serial_in_order, MyInputFunc(size))  &
    tbb::make_filter<int, float>(
      pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, int>(
      pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2int) &
    tbb::make_filter<int, void>(
      pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2void)
  );
}

// parallel_pipeline_tbb_6_pipes
void parallel_pipeline_tbb_6_pipes(std::string pipes, unsigned num_lines, size_t size) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(
      tbb::filter::serial_in_order, MyInputFunc(size))  &
    tbb::make_filter<int, float>(
      pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, std::string>(
      pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2string) &
    tbb::make_filter<std::string, int>(
      pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, string2int) &
    tbb::make_filter<int, void>(
      pipes[5] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2void)
  );
}

// parallel_pipeline_tbb_7_pipes
void parallel_pipeline_tbb_7_pipes(std::string pipes, unsigned num_lines, size_t size) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(
      tbb::filter::serial_in_order, MyInputFunc(size))  &
    tbb::make_filter<int, float>(
      pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, std::string>(
      pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2string) &
    tbb::make_filter<std::string, int>(
      pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, string2int) &
    tbb::make_filter<int, int>(
      pipes[5] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2int) &
    tbb::make_filter<int, void>(
      pipes[6] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2void)
  );
}

// parallel_pipeline_tbb_8_pipes
void parallel_pipeline_tbb_8_pipes(std::string pipes, unsigned num_lines, size_t size) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(
      tbb::filter::serial_in_order, MyInputFunc(size))  &
    tbb::make_filter<int, float>(
      pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, std::string>(
      pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2string) &
    tbb::make_filter<std::string, int>(
      pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, string2int) &
    tbb::make_filter<int, std::vector<int> >(
      pipes[5] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2vector) &
    tbb::make_filter<std::vector<int>, int>(
      pipes[6] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, vector2int) &
    tbb::make_filter<int, void>(
      pipes[7] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2void)
  );
}

// parallel_pipeline_tbb_9_pipes
void parallel_pipeline_tbb_9_pipes(std::string pipes, unsigned num_lines, size_t size) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(
      tbb::filter::serial_in_order, MyInputFunc(size))  &
    tbb::make_filter<int, float>(
      pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, std::string>(
      pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2string) &
    tbb::make_filter<std::string, int>(
      pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, string2int) &
    tbb::make_filter<int, std::vector<int> >(
      pipes[5] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2vector) &
    tbb::make_filter<std::vector<int>, int>(
      pipes[6] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, vector2int) &
    tbb::make_filter<int, int>(
      pipes[7] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2int) &
    tbb::make_filter<int, void>(
      pipes[8] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2void)
  );
}

// parallel_pipeline_tbb_10_pipes
void parallel_pipeline_tbb_10_pipes(std::string pipes, unsigned num_lines, size_t size) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(
      tbb::filter::serial_in_order, MyInputFunc(size))  &
    tbb::make_filter<int, float>(
      pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, std::string>(
      pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2string) &
    tbb::make_filter<std::string, int>(
      pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, string2int) &
    tbb::make_filter<int, std::vector<int> >(
      pipes[5] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2vector) &
    tbb::make_filter<std::vector<int>, int>(
      pipes[6] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, vector2int) &
    tbb::make_filter<int, float>(
      pipes[7] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[8] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, void>(
      pipes[9] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2void)
  );
}

// parallel_pipeline_tbb_11_pipes
void parallel_pipeline_tbb_11_pipes(std::string pipes, unsigned num_lines, size_t size) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(
      tbb::filter::serial_in_order, MyInputFunc(size))  &
    tbb::make_filter<int, float>(
      pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, std::string>(
      pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2string) &
    tbb::make_filter<std::string, int>(
      pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, string2int) &
    tbb::make_filter<int, std::vector<int> >(
      pipes[5] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2vector) &
    tbb::make_filter<std::vector<int>, int>(
      pipes[6] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, vector2int) &
    tbb::make_filter<int, float>(
      pipes[7] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[8] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, int>(
      pipes[9] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2int) &
    tbb::make_filter<int, void>(
      pipes[10] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2void)
  );
}

// parallel_pipeline_tbb_12_pipes
void parallel_pipeline_tbb_12_pipes(std::string pipes, unsigned num_lines, size_t size) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(
      tbb::filter::serial_in_order, MyInputFunc(size))  &
    tbb::make_filter<int, float>(
      pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, std::string>(
      pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2string) &
    tbb::make_filter<std::string, int>(
      pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, string2int) &
    tbb::make_filter<int, std::vector<int> >(
      pipes[5] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2vector) &
    tbb::make_filter<std::vector<int>, int>(
      pipes[6] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, vector2int) &
    tbb::make_filter<int, float>(
      pipes[7] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[8] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, std::string>(
      pipes[9] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2string) &
    tbb::make_filter<std::string, int>(
      pipes[10] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, string2int) &
    tbb::make_filter<int, void>(
      pipes[11] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2void)
  );
}

// parallel_pipeline_tbb_13_pipes
void parallel_pipeline_tbb_13_pipes(std::string pipes, unsigned num_lines, size_t size) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(
      tbb::filter::serial_in_order, MyInputFunc(size))  &
    tbb::make_filter<int, float>(
      pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, std::string>(
      pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2string) &
    tbb::make_filter<std::string, int>(
      pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, string2int) &
    tbb::make_filter<int, std::vector<int> >(
      pipes[5] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2vector) &
    tbb::make_filter<std::vector<int>, int>(
      pipes[6] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, vector2int) &
    tbb::make_filter<int, float>(
      pipes[7] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[8] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, std::string>(
      pipes[9] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2string) &
    tbb::make_filter<std::string, int>(
      pipes[10] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, string2int) &
    tbb::make_filter<int, int>(
      pipes[11] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2int) &
    tbb::make_filter<int, void>(
      pipes[12] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2void)
  );
}

// parallel_pipeline_tbb_14_pipes
void parallel_pipeline_tbb_14_pipes(std::string pipes, unsigned num_lines, size_t size) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(
      tbb::filter::serial_in_order, MyInputFunc(size))  &
    tbb::make_filter<int, float>(
      pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, std::string>(
      pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2string) &
    tbb::make_filter<std::string, int>(
      pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, string2int) &
    tbb::make_filter<int, std::vector<int> >(
      pipes[5] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2vector) &
    tbb::make_filter<std::vector<int>, int>(
      pipes[6] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, vector2int) &
    tbb::make_filter<int, float>(
      pipes[7] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[8] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, std::string>(
      pipes[9] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2string) &
    tbb::make_filter<std::string, int>(
      pipes[10] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, string2int) &
    tbb::make_filter<int, std::vector<int> >(
      pipes[11] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2vector) &
    tbb::make_filter<std::vector<int>, int>(
      pipes[12] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, vector2int) &
    tbb::make_filter<int, void>(
      pipes[13] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2void)
  );
}

// parallel_pipeline_tbb_15_pipes
void parallel_pipeline_tbb_15_pipes(std::string pipes, unsigned num_lines, size_t size) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(
      tbb::filter::serial_in_order, MyInputFunc(size))  &
    tbb::make_filter<int, float>(
      pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, std::string>(
      pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2string) &
    tbb::make_filter<std::string, int>(
      pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, string2int) &
    tbb::make_filter<int, std::vector<int> >(
      pipes[5] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2vector) &
    tbb::make_filter<std::vector<int>, int>(
      pipes[6] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, vector2int) &
    tbb::make_filter<int, float>(
      pipes[7] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[8] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, std::string>(
      pipes[9] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2string) &
    tbb::make_filter<std::string, int>(
      pipes[10] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, string2int) &
    tbb::make_filter<int, std::vector<int> >(
      pipes[11] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2vector) &
    tbb::make_filter<std::vector<int>, int>(
      pipes[12] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, vector2int) &
    tbb::make_filter<int, int>(
      pipes[13] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2int) &
    tbb::make_filter<int, void>(
      pipes[14] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2void)
  );
}

// parallel_pipeline_tbb_16_pipes
void parallel_pipeline_tbb_16_pipes(std::string pipes, unsigned num_lines, size_t size) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(
      tbb::filter::serial_in_order, MyInputFunc(size))  &
    tbb::make_filter<int, float>(
      pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, std::string>(
      pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2string) &
    tbb::make_filter<std::string, int>(
      pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, string2int) &
    tbb::make_filter<int, std::vector<int> >(
      pipes[5] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2vector) &
    tbb::make_filter<std::vector<int>, int>(
      pipes[6] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, vector2int) &
    tbb::make_filter<int, float>(
      pipes[7] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[8] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, std::string>(
      pipes[9] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2string) &
    tbb::make_filter<std::string, int>(
      pipes[10] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, string2int) &
    tbb::make_filter<int, std::vector<int> >(
      pipes[11] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2vector) &
    tbb::make_filter<std::vector<int>, int>(
      pipes[12] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, vector2int) &
    tbb::make_filter<int, float>(
      pipes[13] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2float) &
    tbb::make_filter<float, int>(
      pipes[14] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, float2int) &
    tbb::make_filter<int, void>(
      pipes[15] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, int2void)
  );
}

} // namespace

std::chrono::microseconds measure_time_tbb(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {
  //result.clear();
  //utility::thread_number_range threads( utility::get_default_num_threads, 0);
  tbb::global_control c(tbb::global_control::max_allowed_parallelism, num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();
  i = 0;
  switch(pipes.size()) {
    case 1:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_1_pipe(num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 2:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_2_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 3:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_3_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 4:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_4_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 5:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_5_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 6:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_6_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 7:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_7_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 8:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_8_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 9:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_9_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 10:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_10_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;
    case 11:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_11_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 12:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_12_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 13:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_13_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 14:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_14_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 15:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_15_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 16:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_16_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    default:
      throw std::runtime_error("can support only up to 16 pipes");
    break;
  }

  //std::ofstream outputfile;
  //outputfile.open("./tbb_result.txt", std::ofstream::app);
  //for (auto r:result) {
  //  outputfile << r << '\n';
  //}

  //std::ofstream outputfile;
  //outputfile.open("./build/benchmarks/tbb_time.csv", std::ofstream::app);
  //outputfile << num_threads << ','
  //           << num_lines   << ','
  //           << pipes       << ','
  //           << size        << ','
  //           << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count())/1e3
  //           << '\n';
  //outputfile.close();

  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
