#include "data_pipeline.hpp"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/data_pipeline_dev.hpp>
#include <vector>

namespace efficient{
//my convert function
auto int2int = [](int& input) {
  work();
  return input + 1;
};

auto int2void = [](int& input) {  work(); };

tf::PipeType to_pipe_type(char t) {
  return t == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL;
}

// efficient::parallel_pipeline_taskflow_1_pipe
std::chrono::microseconds parallel_pipeline_taskflow_1_pipe(
  unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline_aligned pl(num_lines,
    tf::make_datapipe<tf::Pipeflow&, void>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> void{
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

// efficient::parallel_pipeline_taskflow_2_pipes
std::chrono::microseconds parallel_pipeline_taskflow_2_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline_aligned pl(num_lines,
    tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_datapipe<int, void>(to_pipe_type(pipes[1]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// efficient::parallel_pipeline_taskflow_3_pipes
std::chrono::microseconds parallel_pipeline_taskflow_3_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline_aligned pl(num_lines,
    tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[1]), int2int),
    tf::make_datapipe<int, void>(to_pipe_type(pipes[2]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// efficient::parallel_pipeline_taskflow_4_pipes
std::chrono::microseconds parallel_pipeline_taskflow_4_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline_aligned pl(num_lines,
    tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[1]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_datapipe<int, void>(to_pipe_type(pipes[3]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// efficient::parallel_pipeline_taskflow_5_pipes
std::chrono::microseconds parallel_pipeline_taskflow_5_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline_aligned pl(num_lines,
    tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[1]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_datapipe<int, void>(to_pipe_type(pipes[4]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// efficient::parallel_pipeline_taskflow_6_pipes
std::chrono::microseconds parallel_pipeline_taskflow_6_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline_aligned pl(num_lines,
    tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[1]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_datapipe<int, void>(to_pipe_type(pipes[5]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// efficient::parallel_pipeline_taskflow_7_pipes
std::chrono::microseconds parallel_pipeline_taskflow_7_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline_aligned pl(num_lines,
    tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[1]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_datapipe<int, void>(to_pipe_type(pipes[6]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// efficient::parallel_pipeline_taskflow_8_pipes
std::chrono::microseconds parallel_pipeline_taskflow_8_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline_aligned pl(num_lines,
    tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[1]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_datapipe<int, void>(to_pipe_type(pipes[7]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// efficient::parallel_pipeline_taskflow_9_pipes
std::chrono::microseconds parallel_pipeline_taskflow_9_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline_aligned pl(num_lines,
    tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[1]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_datapipe<int, void>(to_pipe_type(pipes[8]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// efficient::parallel_pipeline_taskflow_10_pipes
std::chrono::microseconds parallel_pipeline_taskflow_10_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline_aligned pl(num_lines,
    tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[1]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[8]), int2int),
    tf::make_datapipe<int, void>(to_pipe_type(pipes[9]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


// efficient::parallel_pipeline_taskflow_11_pipes
std::chrono::microseconds parallel_pipeline_taskflow_11_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline_aligned pl(num_lines,
    tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[1]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[8]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[9]), int2int),
    tf::make_datapipe<int, void>(to_pipe_type(pipes[10]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// efficient::parallel_pipeline_taskflow_12_pipes
std::chrono::microseconds parallel_pipeline_taskflow_12_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline_aligned pl(num_lines,
    tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[1]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[8]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[9]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[10]), int2int),
    tf::make_datapipe<int, void>(to_pipe_type(pipes[11]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// efficient::parallel_pipeline_taskflow_13_pipes
std::chrono::microseconds parallel_pipeline_taskflow_13_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline_aligned pl(num_lines,
    tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[1]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[8]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[9]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[10]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[11]), int2int),
    tf::make_datapipe<int, void>(to_pipe_type(pipes[12]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// efficient::parallel_pipeline_taskflow_14_pipes
std::chrono::microseconds parallel_pipeline_taskflow_14_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline_aligned pl(num_lines,
    tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[1]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[8]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[9]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[10]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[11]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[12]), int2int),
    tf::make_datapipe<int, void>(to_pipe_type(pipes[13]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// efficient::parallel_pipeline_taskflow_15_pipes
std::chrono::microseconds parallel_pipeline_taskflow_15_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline_aligned pl(num_lines,
    tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[1]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[8]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[9]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[10]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[11]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[12]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[13]), int2int),
    tf::make_datapipe<int, void>(to_pipe_type(pipes[14]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// efficient::parallel_pipeline_taskflow_16_pipes
std::chrono::microseconds parallel_pipeline_taskflow_16_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::DataPipeline_aligned pl(num_lines,
    tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [size](tf::Pipeflow& pf) -> int{
      if(pf.token() == size) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[1]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[2]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[3]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[4]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[5]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[6]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[7]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[8]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[9]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[10]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[11]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[12]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[13]), int2int),
    tf::make_datapipe<int, int>(to_pipe_type(pipes[14]), int2int),
    tf::make_datapipe<int, void>(to_pipe_type(pipes[15]), int2void)
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
}

std::chrono::microseconds measure_time_efficient(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  std::chrono::microseconds elapsed;

  switch(pipes.size()) {
    case 1:
      elapsed = efficient::parallel_pipeline_taskflow_1_pipe(num_lines, num_threads, size);
      break;

    case 2:
      elapsed = efficient::parallel_pipeline_taskflow_2_pipes(pipes, num_lines, num_threads, size);
      break;

    case 3:
      elapsed = efficient::parallel_pipeline_taskflow_3_pipes(pipes, num_lines, num_threads, size);
      break;

    case 4:
      elapsed = efficient::parallel_pipeline_taskflow_4_pipes(pipes, num_lines, num_threads, size);
      break;

    case 5:
      elapsed = efficient::parallel_pipeline_taskflow_5_pipes(pipes, num_lines, num_threads, size);
      break;

    case 6:
      elapsed = efficient::parallel_pipeline_taskflow_6_pipes(pipes, num_lines, num_threads, size);
      break;

    case 7:
      elapsed = efficient::parallel_pipeline_taskflow_7_pipes(pipes, num_lines, num_threads, size);
      break;

    case 8:
      elapsed = efficient::parallel_pipeline_taskflow_8_pipes(pipes, num_lines, num_threads, size);
      break;

    case 9:
      elapsed = efficient::parallel_pipeline_taskflow_9_pipes(pipes, num_lines, num_threads, size);
      break;

    case 10:
      elapsed = efficient::parallel_pipeline_taskflow_10_pipes(pipes, num_lines, num_threads, size);
      break;

    case 11:
      elapsed = efficient::parallel_pipeline_taskflow_11_pipes(pipes, num_lines, num_threads, size);
      break;

    case 12:
      elapsed = efficient::parallel_pipeline_taskflow_12_pipes(pipes, num_lines, num_threads, size);
      break;

    case 13:
      elapsed = efficient::parallel_pipeline_taskflow_13_pipes(pipes, num_lines, num_threads, size);
      break;

    case 14:
      elapsed = efficient::parallel_pipeline_taskflow_14_pipes(pipes, num_lines, num_threads, size);
      break;

    case 15:
      elapsed = efficient::parallel_pipeline_taskflow_15_pipes(pipes, num_lines, num_threads, size);
      break;

    case 16:
      elapsed = efficient::parallel_pipeline_taskflow_16_pipes(pipes, num_lines, num_threads, size);
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

