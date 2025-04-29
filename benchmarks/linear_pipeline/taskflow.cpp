#include "linear_pipeline.hpp"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>

struct Input {
  size_t i;
  size_t size;
  void operator()(tf::Pipeflow& pf) {
    work();
    if (i++ == size) {
      pf.stop();
    }
  }
};

struct Filter {
  void operator()(tf::Pipeflow&) {
    work();
  }
};

tf::PipeType to_pipe_type(char t) {
  return t == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL;
}

// parallel_pipeline_taskflow_1_pipe
std::chrono::microseconds parallel_pipeline_taskflow_1_pipe(
  unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, size}}
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

  std::vector<std::array<int, 2>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, size}},
    tf::Pipe{to_pipe_type(pipes[1]), Filter{}}
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

  std::vector<std::array<int, 3>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, size}},
    tf::Pipe{to_pipe_type(pipes[1]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[2]), Filter{}}
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

  std::vector<std::array<int, 4>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, size}},
    tf::Pipe{to_pipe_type(pipes[1]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[2]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[3]), Filter{}}
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
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, size}},
    tf::Pipe{to_pipe_type(pipes[1]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[2]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[3]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[4]), Filter{}}
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
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, size}},
    tf::Pipe{to_pipe_type(pipes[1]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[2]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[3]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[4]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[5]), Filter{}}
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

  std::vector<std::array<int, 7>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, size}},
    tf::Pipe{to_pipe_type(pipes[1]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[2]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[3]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[4]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[5]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[6]), Filter{}}
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
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, size}},
    tf::Pipe{to_pipe_type(pipes[1]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[2]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[3]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[4]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[5]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[6]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[7]), Filter{}}
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
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, size}},
    tf::Pipe{to_pipe_type(pipes[1]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[2]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[3]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[4]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[5]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[6]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[7]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[8]), Filter{}}
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
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, size}},
    tf::Pipe{to_pipe_type(pipes[1]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[2]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[3]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[4]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[5]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[6]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[7]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[8]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[9]), Filter{}}
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
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, size}},
    tf::Pipe{to_pipe_type(pipes[1]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[2]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[3]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[4]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[5]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[6]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[7]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[8]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[9]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[10]), Filter{}}
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
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, size}},
    tf::Pipe{to_pipe_type(pipes[1]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[2]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[3]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[4]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[5]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[6]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[7]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[8]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[9]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[10]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[11]), Filter{}}
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
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, size}},
    tf::Pipe{to_pipe_type(pipes[1]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[2]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[3]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[4]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[5]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[6]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[7]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[8]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[9]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[10]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[11]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[12]), Filter{}}
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
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, size}},
    tf::Pipe{to_pipe_type(pipes[1]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[2]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[3]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[4]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[5]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[6]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[7]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[8]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[9]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[10]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[11]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[12]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[13]), Filter{}}
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
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, size}},
    tf::Pipe{to_pipe_type(pipes[1]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[2]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[3]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[4]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[5]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[6]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[7]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[8]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[9]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[10]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[11]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[12]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[13]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[14]), Filter{}}
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
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, size}},
    tf::Pipe{to_pipe_type(pipes[1]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[2]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[3]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[4]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[5]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[6]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[7]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[8]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[9]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[10]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[11]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[12]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[13]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[14]), Filter{}},
    tf::Pipe{to_pipe_type(pipes[15]), Filter{}}
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

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


