#include "data_pipeline.hpp"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/data_pipeline.hpp>

namespace {

//my convert function
auto int2string = [](int& input) -> std::string {
  work_int(input);
  return std::to_string(input);
};

auto string2int = [](std::string& input) -> int {
  work_string(input);
  return std::stoi(input);
};

auto int2float = [](int& input) -> float {
  work_int(input);
  return input * 1.0;
};

auto float2int = [](float& input) -> int {
  work_float(input);
  return (int)input;
};

auto int2vector = [](int& input) -> std::vector<int> {
  work_int(input);
  return std::vector{input};
};

auto vector2int = [](std::vector<int>& input) -> int {
  work_vector(input);
  return input[0];
};

auto int2int = [](int& input) {
  work_int(input);
  return input;
};

auto int2void = [](int& input) {  work_int(input); };

tf::PipeType to_pipe_type(char t) {
  return t == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL;
}

// parallel_pipeline_taskflow_1_pipe
std::chrono::microseconds parallel_pipeline_taskflow_1_pipe(
  unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<tf::Pipeflow&, void>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> void{
      if(pf.token() == size) {
        pf.stop();
      }
    })
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_2_pipes
std::chrono::microseconds parallel_pipeline_taskflow_2_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[1]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_3_pipes
std::chrono::microseconds parallel_pipeline_taskflow_3_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[1]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[2]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_4_pipes
std::chrono::microseconds parallel_pipeline_taskflow_4_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[1]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[2]), float2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[3]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_5_pipes
std::chrono::microseconds parallel_pipeline_taskflow_5_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[1]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[2]), float2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[4]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_6_pipes
std::chrono::microseconds parallel_pipeline_taskflow_6_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[1]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[2]), float2int),
    tf::make_data_pipe<int, std::string>(to_pipe_type(pipes[3]), int2string),
    tf::make_data_pipe<std::string, int>(to_pipe_type(pipes[4]), string2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[5]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_7_pipes
std::chrono::microseconds parallel_pipeline_taskflow_7_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[1]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[2]), float2int),
    tf::make_data_pipe<int, std::string>(to_pipe_type(pipes[3]), int2string),
    tf::make_data_pipe<std::string, int>(to_pipe_type(pipes[4]), string2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[6]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_8_pipes
std::chrono::microseconds parallel_pipeline_taskflow_8_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[1]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[2]), float2int),
    tf::make_data_pipe<int, std::string>(to_pipe_type(pipes[3]), int2string),
    tf::make_data_pipe<std::string, int>(to_pipe_type(pipes[4]), string2int),
    tf::make_data_pipe<int, std::vector<int> >(to_pipe_type(pipes[5]), int2vector),
    tf::make_data_pipe<std::vector<int>, int>(to_pipe_type(pipes[6]), vector2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[7]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_9_pipes
std::chrono::microseconds parallel_pipeline_taskflow_9_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[1]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[2]), float2int),
    tf::make_data_pipe<int, std::string>(to_pipe_type(pipes[3]), int2string),
    tf::make_data_pipe<std::string, int>(to_pipe_type(pipes[4]), string2int),
    tf::make_data_pipe<int, std::vector<int> >(to_pipe_type(pipes[5]), int2vector),
    tf::make_data_pipe<std::vector<int>, int>(to_pipe_type(pipes[6]), vector2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[8]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_10_pipes
std::chrono::microseconds parallel_pipeline_taskflow_10_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[1]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[2]), float2int),
    tf::make_data_pipe<int, std::string>(to_pipe_type(pipes[3]), int2string),
    tf::make_data_pipe<std::string, int>(to_pipe_type(pipes[4]), string2int),
    tf::make_data_pipe<int, std::vector<int> >(to_pipe_type(pipes[5]), int2vector),
    tf::make_data_pipe<std::vector<int>, int>(to_pipe_type(pipes[6]), vector2int),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[7]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[8]), float2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[9]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


// parallel_pipeline_taskflow_11_pipes
std::chrono::microseconds parallel_pipeline_taskflow_11_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[1]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[2]), float2int),
    tf::make_data_pipe<int, std::string>(to_pipe_type(pipes[3]), int2string),
    tf::make_data_pipe<std::string, int>(to_pipe_type(pipes[4]), string2int),
    tf::make_data_pipe<int, std::vector<int> >(to_pipe_type(pipes[5]), int2vector),
    tf::make_data_pipe<std::vector<int>, int>(to_pipe_type(pipes[6]), vector2int),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[7]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[8]), float2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[9]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[10]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_12_pipes
std::chrono::microseconds parallel_pipeline_taskflow_12_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[1]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[2]), float2int),
    tf::make_data_pipe<int, std::string>(to_pipe_type(pipes[3]), int2string),
    tf::make_data_pipe<std::string, int>(to_pipe_type(pipes[4]), string2int),
    tf::make_data_pipe<int, std::vector<int> >(to_pipe_type(pipes[5]), int2vector),
    tf::make_data_pipe<std::vector<int>, int>(to_pipe_type(pipes[6]), vector2int),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[7]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[8]), float2int),
    tf::make_data_pipe<int, std::string>(to_pipe_type(pipes[9]), int2string),
    tf::make_data_pipe<std::string, int>(to_pipe_type(pipes[10]), string2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[11]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_13_pipes
std::chrono::microseconds parallel_pipeline_taskflow_13_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[1]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[2]), float2int),
    tf::make_data_pipe<int, std::string>(to_pipe_type(pipes[3]), int2string),
    tf::make_data_pipe<std::string, int>(to_pipe_type(pipes[4]), string2int),
    tf::make_data_pipe<int, std::vector<int> >(to_pipe_type(pipes[5]), int2vector),
    tf::make_data_pipe<std::vector<int>, int>(to_pipe_type(pipes[6]), vector2int),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[7]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[8]), float2int),
    tf::make_data_pipe<int, std::string>(to_pipe_type(pipes[9]), int2string),
    tf::make_data_pipe<std::string, int>(to_pipe_type(pipes[10]), string2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[11]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[12]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_14_pipes
std::chrono::microseconds parallel_pipeline_taskflow_14_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[1]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[2]), float2int),
    tf::make_data_pipe<int, std::string>(to_pipe_type(pipes[3]), int2string),
    tf::make_data_pipe<std::string, int>(to_pipe_type(pipes[4]), string2int),
    tf::make_data_pipe<int, std::vector<int> >(to_pipe_type(pipes[5]), int2vector),
    tf::make_data_pipe<std::vector<int>, int>(to_pipe_type(pipes[6]), vector2int),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[7]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[8]), float2int),
    tf::make_data_pipe<int, std::string>(to_pipe_type(pipes[9]), int2string),
    tf::make_data_pipe<std::string, int>(to_pipe_type(pipes[10]), string2int),
    tf::make_data_pipe<int, std::vector<int> >(to_pipe_type(pipes[11]), int2vector),
    tf::make_data_pipe<std::vector<int>, int>(to_pipe_type(pipes[12]), vector2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[13]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_15_pipes
std::chrono::microseconds parallel_pipeline_taskflow_15_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[1]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[2]), float2int),
    tf::make_data_pipe<int, std::string>(to_pipe_type(pipes[3]), int2string),
    tf::make_data_pipe<std::string, int>(to_pipe_type(pipes[4]), string2int),
    tf::make_data_pipe<int, std::vector<int> >(to_pipe_type(pipes[5]), int2vector),
    tf::make_data_pipe<std::vector<int>, int>(to_pipe_type(pipes[6]), vector2int),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[7]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[8]), float2int),
    tf::make_data_pipe<int, std::string>(to_pipe_type(pipes[9]), int2string),
    tf::make_data_pipe<std::string, int>(to_pipe_type(pipes[10]), string2int),
    tf::make_data_pipe<int, std::vector<int> >(to_pipe_type(pipes[11]), int2vector),
    tf::make_data_pipe<std::vector<int>, int>(to_pipe_type(pipes[12]), vector2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[13]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[14]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_16_pipes
std::chrono::microseconds parallel_pipeline_taskflow_16_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[1]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[2]), float2int),
    tf::make_data_pipe<int, std::string>(to_pipe_type(pipes[3]), int2string),
    tf::make_data_pipe<std::string, int>(to_pipe_type(pipes[4]), string2int),
    tf::make_data_pipe<int, std::vector<int> >(to_pipe_type(pipes[5]), int2vector),
    tf::make_data_pipe<std::vector<int>, int>(to_pipe_type(pipes[6]), vector2int),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[7]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[8]), float2int),
    tf::make_data_pipe<int, std::string>(to_pipe_type(pipes[9]), int2string),
    tf::make_data_pipe<std::string, int>(to_pipe_type(pipes[10]), string2int),
    tf::make_data_pipe<int, std::vector<int> >(to_pipe_type(pipes[11]), int2vector),
    tf::make_data_pipe<std::vector<int>, int>(to_pipe_type(pipes[12]), vector2int),
    tf::make_data_pipe<int, float>(to_pipe_type(pipes[13]), int2float),
    tf::make_data_pipe<float, int>(to_pipe_type(pipes[14]), float2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[15]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

} // namespace

std::chrono::microseconds measure_time_taskflow(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  std::chrono::microseconds elapsed;

  switch(pipes.size()) {
    case 1:
      elapsed = parallel_pipeline_taskflow_1_pipe(num_lines, num_threads, size);
      break;

    case 2:
      elapsed = parallel_pipeline_taskflow_2_pipes(pipes, num_lines, num_threads, size);
      break;

    case 3:
      elapsed = parallel_pipeline_taskflow_3_pipes(pipes, num_lines, num_threads, size);
      break;

    case 4:
      elapsed = parallel_pipeline_taskflow_4_pipes(pipes, num_lines, num_threads, size);
      break;

    case 5:
      elapsed = parallel_pipeline_taskflow_5_pipes(pipes, num_lines, num_threads, size);
      break;

    case 6:
      elapsed = parallel_pipeline_taskflow_6_pipes(pipes, num_lines, num_threads, size);
      break;

    case 7:
      elapsed = parallel_pipeline_taskflow_7_pipes(pipes, num_lines, num_threads, size);
      break;

    case 8:
      elapsed = parallel_pipeline_taskflow_8_pipes(pipes, num_lines, num_threads, size);
      break;

    case 9:
      elapsed = parallel_pipeline_taskflow_9_pipes(pipes, num_lines, num_threads, size);
      break;

    case 10:
      elapsed = parallel_pipeline_taskflow_10_pipes(pipes, num_lines, num_threads, size);
      break;

    case 11:
      elapsed = parallel_pipeline_taskflow_11_pipes(pipes, num_lines, num_threads, size);
      break;

    case 12:
      elapsed = parallel_pipeline_taskflow_12_pipes(pipes, num_lines, num_threads, size);
      break;

    case 13:
      elapsed = parallel_pipeline_taskflow_13_pipes(pipes, num_lines, num_threads, size);
      break;

    case 14:
      elapsed = parallel_pipeline_taskflow_14_pipes(pipes, num_lines, num_threads, size);
      break;

    case 15:
      elapsed = parallel_pipeline_taskflow_15_pipes(pipes, num_lines, num_threads, size);
      break;

    case 16:
      elapsed = parallel_pipeline_taskflow_16_pipes(pipes, num_lines, num_threads, size);
      break;

    default:
      throw std::runtime_error("can support only up to 16 pipes");
    break;
  }

  //std::ofstream outputfile;
  //outputfile.open("./build/benchmarks/tf_time.csv", std::ofstream::app);
  //outputfile << num_threads << ','
  //           << num_lines   << ','
  //           << pipes       << ','
  //           << size        << ','
  //           << elapsed.count()/1e3 << '\n';

  //outputfile.close();
  return elapsed;
}


