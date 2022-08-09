#include "data_pipeline.hpp"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/data_pipeline_dev.hpp>
#include <vector>
#include <string>

namespace normal{
//my convert function
auto int2int = [](int& input) {
  work_int(input);
  return input + 1;
};

auto string2string = [](std::string& input) {
  work_string(input);
  return input;
};

auto string2void = [](std::string& input) {  work_string(input); };

auto int2void = [](int& input) {  work_int(input); };

tf::PipeType to_pipe_type(char t) {
  return t == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL;
}

// normal::parallel_pipeline_taskflow_1_pipe
std::chrono::microseconds parallel_pipeline_taskflow_1_pipe_int(
  unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

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

// normal::parallel_pipeline_taskflow_2_pipes_int
std::chrono::microseconds parallel_pipeline_taskflow_2_pipes_int(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

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

// normal::parallel_pipeline_taskflow_3_pipes_int
std::chrono::microseconds parallel_pipeline_taskflow_3_pipes_int(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

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

// normal::parallel_pipeline_taskflow_4_pipes_int
std::chrono::microseconds parallel_pipeline_taskflow_4_pipes_int(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

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
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[3]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_5_pipes_int
std::chrono::microseconds parallel_pipeline_taskflow_5_pipes_int(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

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
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[4]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_6_pipes_int
std::chrono::microseconds parallel_pipeline_taskflow_6_pipes_int(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

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
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[5]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_7_pipes_int
std::chrono::microseconds parallel_pipeline_taskflow_7_pipes_int(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

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
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[6]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_8_pipes_int
std::chrono::microseconds parallel_pipeline_taskflow_8_pipes_int(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

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
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[7]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_9_pipes_int
std::chrono::microseconds parallel_pipeline_taskflow_9_pipes_int(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

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
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[8]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_10_pipes_int
std::chrono::microseconds parallel_pipeline_taskflow_10_pipes_int(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

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
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[8]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[9]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


// normal::parallel_pipeline_taskflow_11_pipes_int
std::chrono::microseconds parallel_pipeline_taskflow_11_pipes_int(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

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
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[8]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[9]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[10]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_12_pipes_int
std::chrono::microseconds parallel_pipeline_taskflow_12_pipes_int(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

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
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[8]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[9]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[10]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[11]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_13_pipes_int
std::chrono::microseconds parallel_pipeline_taskflow_13_pipes_int(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

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
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[8]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[9]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[10]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[11]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[12]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_14_pipes_int
std::chrono::microseconds parallel_pipeline_taskflow_14_pipes_int(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

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
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[8]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[9]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[10]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[11]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[12]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[13]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_15_pipes_int
std::chrono::microseconds parallel_pipeline_taskflow_15_pipes_int(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

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
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[8]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[9]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[10]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[11]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[12]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[13]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[14]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_16_pipes_int
std::chrono::microseconds parallel_pipeline_taskflow_16_pipes_int(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

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
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[8]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[9]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[10]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[11]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[12]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[13]), int2int),
    tf::make_data_pipe<int, int>(to_pipe_type(pipes[14]), int2int),
    tf::make_data_pipe<int, void>(to_pipe_type(pipes[15]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


// normal::parallel_pipeline_taskflow_1_pipe
std::chrono::microseconds parallel_pipeline_taskflow_1_pipe_string(
  unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, void>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> void{
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

// normal::parallel_pipeline_taskflow_2_pipes_string
std::chrono::microseconds parallel_pipeline_taskflow_2_pipes_string(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, std::string>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> std::string{
      if(pf.token() == size) {
        pf.stop();
        return "";
      }
      else {
        return std::to_string(pf.token());
      }
    }),
    tf::make_data_pipe<std::string, void>(to_pipe_type(pipes[1]), string2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_3_pipes_string
std::chrono::microseconds parallel_pipeline_taskflow_3_pipes_string(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, std::string>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> std::string{
      if(pf.token() == size) {
        pf.stop();
        return "";
      }
      else {
        return std::to_string(pf.token());
      }
    }),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[1]), string2string),
    tf::make_data_pipe<std::string, void>(to_pipe_type(pipes[2]), string2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_4_pipes_string
std::chrono::microseconds parallel_pipeline_taskflow_4_pipes_string(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, std::string>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> std::string{
      if(pf.token() == size) {
        pf.stop();
        return "";
      }
      else {
        return std::to_string(pf.token());
      }
    }),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[1]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[2]), string2string),
    tf::make_data_pipe<std::string, void>(to_pipe_type(pipes[3]), string2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_5_pipes_string
std::chrono::microseconds parallel_pipeline_taskflow_5_pipes_string(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, std::string>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> std::string{
      if(pf.token() == size) {
        pf.stop();
        return "";
      }
      else {
        return std::to_string(pf.token());
      }
    }),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[1]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[2]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[3]), string2string),
    tf::make_data_pipe<std::string, void>(to_pipe_type(pipes[4]), string2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_6_pipes_string
std::chrono::microseconds parallel_pipeline_taskflow_6_pipes_string(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, std::string>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> std::string{
      if(pf.token() == size) {
        pf.stop();
        return "";
      }
      else {
        return std::to_string(pf.token());
      }
    }),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[1]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[2]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[3]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[4]), string2string),
    tf::make_data_pipe<std::string, void>(to_pipe_type(pipes[5]), string2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_7_pipes_string
std::chrono::microseconds parallel_pipeline_taskflow_7_pipes_string(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, std::string>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> std::string{
      if(pf.token() == size) {
        pf.stop();
        return "";
      }
      else {
        return std::to_string(pf.token());
      }
    }),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[1]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[2]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[3]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[4]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[5]), string2string),
    tf::make_data_pipe<std::string, void>(to_pipe_type(pipes[6]), string2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_8_pipes_string
std::chrono::microseconds parallel_pipeline_taskflow_8_pipes_string(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, std::string>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> std::string{
      if(pf.token() == size) {
        pf.stop();
        return "";
      }
      else {
        return std::to_string(pf.token());
      }
    }),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[1]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[2]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[3]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[4]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[5]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[6]), string2string),
    tf::make_data_pipe<std::string, void>(to_pipe_type(pipes[7]), string2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_9_pipes_string
std::chrono::microseconds parallel_pipeline_taskflow_9_pipes_string(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, std::string>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> std::string{
      if(pf.token() == size) {
        pf.stop();
        return "";
      }
      else {
        return std::to_string(pf.token());
      }
    }),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[1]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[2]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[3]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[4]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[5]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[6]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[7]), string2string),
    tf::make_data_pipe<std::string, void>(to_pipe_type(pipes[8]), string2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_10_pipes_string
std::chrono::microseconds parallel_pipeline_taskflow_10_pipes_string(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, std::string>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> std::string{
      if(pf.token() == size) {
        pf.stop();
        return "";
      }
      else {
        return std::to_string(pf.token());
      }
    }),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[1]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[2]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[3]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[4]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[5]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[6]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[7]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[8]), string2string),
    tf::make_data_pipe<std::string, void>(to_pipe_type(pipes[9]), string2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


// normal::parallel_pipeline_taskflow_11_pipes_string
std::chrono::microseconds parallel_pipeline_taskflow_11_pipes_string(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, std::string>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> std::string{
      if(pf.token() == size) {
        pf.stop();
        return "";
      }
      else {
        return std::to_string(pf.token());
      }
    }),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[1]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[2]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[3]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[4]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[5]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[6]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[7]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[8]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[9]), string2string),
    tf::make_data_pipe<std::string, void>(to_pipe_type(pipes[10]), string2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_12_pipes_string
std::chrono::microseconds parallel_pipeline_taskflow_12_pipes_string(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, std::string>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> std::string{
      if(pf.token() == size) {
        pf.stop();
        return "";
      }
      else {
        return std::to_string(pf.token());
      }
    }),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[1]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[2]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[3]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[4]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[5]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[6]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[7]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[8]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[9]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[10]), string2string),
    tf::make_data_pipe<std::string, void>(to_pipe_type(pipes[11]), string2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_13_pipes_string
std::chrono::microseconds parallel_pipeline_taskflow_13_pipes_string(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, std::string>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> std::string{
      if(pf.token() == size) {
        pf.stop();
        return "";
      }
      else {
        return std::to_string(pf.token());
      }
    }),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[1]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[2]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[3]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[4]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[5]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[6]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[7]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[8]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[9]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[10]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[11]), string2string),
    tf::make_data_pipe<std::string, void>(to_pipe_type(pipes[12]), string2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_14_pipes_string
std::chrono::microseconds parallel_pipeline_taskflow_14_pipes_string(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, std::string>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> std::string{
      if(pf.token() == size) {
        pf.stop();
        return "";
      }
      else {
        return std::to_string(pf.token());
      }
    }),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[1]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[2]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[3]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[4]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[5]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[6]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[7]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[8]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[9]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[10]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[11]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[12]), string2string),
    tf::make_data_pipe<std::string, void>(to_pipe_type(pipes[13]), string2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_15_pipes_string
std::chrono::microseconds parallel_pipeline_taskflow_15_pipes_string(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, std::string>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> std::string{
      if(pf.token() == size) {
        pf.stop();
        return "";
      }
      else {
        return std::to_string(pf.token());
      }
    }),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[1]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[2]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[3]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[4]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[5]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[6]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[7]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[8]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[9]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[10]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[11]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[12]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[13]), string2string),
    tf::make_data_pipe<std::string, void>(to_pipe_type(pipes[14]), string2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// normal::parallel_pipeline_taskflow_16_pipes_string
std::chrono::microseconds parallel_pipeline_taskflow_16_pipes_string(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, std::string>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> std::string{
      if(pf.token() == size) {
        pf.stop();
        return "";
      }
      else {
        return std::to_string(pf.token());
      }
    }),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[1]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[2]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[3]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[4]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[5]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[6]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[7]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[8]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[9]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[10]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[11]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[12]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[13]), string2string),
    tf::make_data_pipe<std::string, std::string>(to_pipe_type(pipes[14]), string2string),
    tf::make_data_pipe<std::string, void>(to_pipe_type(pipes[15]), string2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
}

std::chrono::microseconds measure_time_normal(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size, std::string datatype) {

  std::chrono::microseconds elapsed;

  if (datatype == "int") {
    switch(pipes.size()) {
      case 1:
        elapsed = normal::parallel_pipeline_taskflow_1_pipe_int(num_lines, num_threads, size);
        break;

      case 2:
        elapsed = normal::parallel_pipeline_taskflow_2_pipes_int(pipes, num_lines, num_threads, size);
        break;

      case 3:
        elapsed = normal::parallel_pipeline_taskflow_3_pipes_int(pipes, num_lines, num_threads, size);
        break;

      case 4:
        elapsed = normal::parallel_pipeline_taskflow_4_pipes_int(pipes, num_lines, num_threads, size);
        break;

      case 5:
        elapsed = normal::parallel_pipeline_taskflow_5_pipes_int(pipes, num_lines, num_threads, size);
        break;

      case 6:
        elapsed = normal::parallel_pipeline_taskflow_6_pipes_int(pipes, num_lines, num_threads, size);
        break;

      case 7:
        elapsed = normal::parallel_pipeline_taskflow_7_pipes_int(pipes, num_lines, num_threads, size);
        break;

      case 8:
        elapsed = normal::parallel_pipeline_taskflow_8_pipes_int(pipes, num_lines, num_threads, size);
        break;

      case 9:
        elapsed = normal::parallel_pipeline_taskflow_9_pipes_int(pipes, num_lines, num_threads, size);
        break;

      case 10:
        elapsed = normal::parallel_pipeline_taskflow_10_pipes_int(pipes, num_lines, num_threads, size);
        break;

      case 11:
        elapsed = normal::parallel_pipeline_taskflow_11_pipes_int(pipes, num_lines, num_threads, size);
        break;

      case 12:
        elapsed = normal::parallel_pipeline_taskflow_12_pipes_int(pipes, num_lines, num_threads, size);
        break;

      case 13:
        elapsed = normal::parallel_pipeline_taskflow_13_pipes_int(pipes, num_lines, num_threads, size);
        break;

      case 14:
        elapsed = normal::parallel_pipeline_taskflow_14_pipes_int(pipes, num_lines, num_threads, size);
        break;

      case 15:
        elapsed = normal::parallel_pipeline_taskflow_15_pipes_int(pipes, num_lines, num_threads, size);
        break;

      case 16:
        elapsed = normal::parallel_pipeline_taskflow_16_pipes_int(pipes, num_lines, num_threads, size);
        break;

      default:
        throw std::runtime_error("can support only up to 16 pipes");
      break;
    }
  } else if (datatype =="string") {
     switch(pipes.size()) {
      case 1:
        elapsed = normal::parallel_pipeline_taskflow_1_pipe_string(num_lines, num_threads, size);
        break;

      case 2:
        elapsed = normal::parallel_pipeline_taskflow_2_pipes_string(pipes, num_lines, num_threads, size);
        break;

      case 3:
        elapsed = normal::parallel_pipeline_taskflow_3_pipes_string(pipes, num_lines, num_threads, size);
        break;

      case 4:
        elapsed = normal::parallel_pipeline_taskflow_4_pipes_string(pipes, num_lines, num_threads, size);
        break;

      case 5:
        elapsed = normal::parallel_pipeline_taskflow_5_pipes_string(pipes, num_lines, num_threads, size);
        break;

      case 6:
        elapsed = normal::parallel_pipeline_taskflow_6_pipes_string(pipes, num_lines, num_threads, size);
        break;

      case 7:
        elapsed = normal::parallel_pipeline_taskflow_7_pipes_string(pipes, num_lines, num_threads, size);
        break;

      case 8:
        elapsed = normal::parallel_pipeline_taskflow_8_pipes_string(pipes, num_lines, num_threads, size);
        break;

      case 9:
        elapsed = normal::parallel_pipeline_taskflow_9_pipes_string(pipes, num_lines, num_threads, size);
        break;

      case 10:
        elapsed = normal::parallel_pipeline_taskflow_10_pipes_string(pipes, num_lines, num_threads, size);
        break;

      case 11:
        elapsed = normal::parallel_pipeline_taskflow_11_pipes_string(pipes, num_lines, num_threads, size);
        break;

      case 12:
        elapsed = normal::parallel_pipeline_taskflow_12_pipes_string(pipes, num_lines, num_threads, size);
        break;

      case 13:
        elapsed = normal::parallel_pipeline_taskflow_13_pipes_string(pipes, num_lines, num_threads, size);
        break;

      case 14:
        elapsed = normal::parallel_pipeline_taskflow_14_pipes_string(pipes, num_lines, num_threads, size);
        break;

      case 15:
        elapsed = normal::parallel_pipeline_taskflow_15_pipes_string(pipes, num_lines, num_threads, size);
        break;

      case 16:
        elapsed = normal::parallel_pipeline_taskflow_16_pipes_string(pipes, num_lines, num_threads, size);
        break;

      default:
        throw std::runtime_error("can support only up to 16 pipes");
      break;   
    }
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
