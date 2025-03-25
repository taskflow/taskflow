#include "levelgraph.hpp"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>
//#include "matrix_calculation.hpp"

struct Input {
  size_t lev;
  size_t len;
  std::vector<int>& mybuffer;
  LevelGraph& graph;

  void operator()(tf::Pipeflow& pf) {
    if (len == 0 && lev == graph.level()) {
      pf.stop();
    }
    else {
      mybuffer[pf.line()] = graph.node_at(lev, len).uid();
      int val = work(mybuffer[pf.line()]);
      graph.node_at(lev, len).set_value(val);

      if(len == graph.length()-1){
        ++lev;
        len = 0;
      }
      else {
        ++len;
      }
    }
  }
};

struct Filter {
  std::vector<int>& mybuffer;
  LevelGraph& graph;

  void operator()(tf::Pipeflow& pf) {

    int uid = mybuffer[pf.line()];
    int val = work(uid);
    size_t level = graph.level();
    size_t length = graph.length();

    graph.node_at(uid/level, uid%length).set_value(val);
  }
};

struct FilterFinal {
  std::vector<int>& mybuffer;
  LevelGraph& graph;

  void operator()(tf::Pipeflow& pf){
    int uid = mybuffer[pf.line()];
    int val = work(uid);

    int lev = uid/graph.level();
    int len = uid%graph.length();

    if (lev != 0) {
      int value_prev_level = 0;

      for (auto& in_edge : graph.node_at(lev, len)._in_edges) {
        value_prev_level += graph.node_at(lev-1, in_edge.first).get_value();
      }
      graph.node_at(lev, len).set_value(val + value_prev_level);
    }
    else {
      graph.node_at(lev, len).set_value(val);
    }
  }
};

// graph_pipeline_taskflow_1_pipe
std::chrono::microseconds graph_pipeline_taskflow_1_pipe(
  LevelGraph& graph, unsigned num_lines, unsigned num_threads) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  std::vector<int> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, 0, mybuffer, graph}} // 1st pipe
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// graph_pipeline_taskflow_2_pipes
std::chrono::microseconds graph_pipeline_taskflow_2_pipes(
  LevelGraph& graph, unsigned num_lines, unsigned num_threads) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  std::vector<int> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, 0, mybuffer, graph}}, // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, FilterFinal{mybuffer, graph}}  // 2nd pipe
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// graph_pipeline_taskflow_3_pipes
std::chrono::microseconds graph_pipeline_taskflow_3_pipes(
  LevelGraph& graph, unsigned num_lines, unsigned num_threads) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  std::vector<int> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, 0, mybuffer, graph}}, // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 2nd pipe
    tf::Pipe{tf::PipeType::SERIAL, FilterFinal{mybuffer, graph}}  // 3rd pipe
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// graph_pipeline_taskflow_4_pipes
std::chrono::microseconds graph_pipeline_taskflow_4_pipes(
  LevelGraph& graph, unsigned num_lines, unsigned num_threads) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  std::vector<int> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, 0, mybuffer, graph}}, // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 2nd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 3rd pipe
    tf::Pipe{tf::PipeType::SERIAL, FilterFinal{mybuffer, graph}}  // 4th pipe
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// graph_pipeline_taskflow_5_pipes
std::chrono::microseconds graph_pipeline_taskflow_5_pipes(
  LevelGraph& graph, unsigned num_lines, unsigned num_threads) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  std::vector<int> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, 0, mybuffer, graph}}, // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 2nd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 3rd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 4th pipe
    tf::Pipe{tf::PipeType::SERIAL, FilterFinal{mybuffer, graph}}  // 5th pipe
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// graph_pipeline_taskflow_6_pipes
std::chrono::microseconds graph_pipeline_taskflow_6_pipes(
  LevelGraph& graph, unsigned num_lines, unsigned num_threads) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  std::vector<int> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, 0, mybuffer, graph}}, // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 2nd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 3rd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 4th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 5th pipe
    tf::Pipe{tf::PipeType::SERIAL, FilterFinal{mybuffer, graph}}  // 6th pipe
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// graph_pipeline_taskflow_7_pipes
std::chrono::microseconds graph_pipeline_taskflow_7_pipes(
  LevelGraph& graph, unsigned num_lines, unsigned num_threads) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  std::vector<int> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, 0, mybuffer, graph}}, // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 2nd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 3rd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 4th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 5th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 6th pipe
    tf::Pipe{tf::PipeType::SERIAL, FilterFinal{mybuffer, graph}}  // 7th pipe
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// graph_pipeline_taskflow_8_pipes
std::chrono::microseconds graph_pipeline_taskflow_8_pipes(
  LevelGraph& graph, unsigned num_lines, unsigned num_threads) {
  //std::ofstream outputfile;
  //outputfile.open("./tf_result.txt", std::ofstream::app);

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  std::vector<int> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, 0, mybuffer, graph}}, // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 2nd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 3rd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 4th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 5th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 6th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 7th pipe
    tf::Pipe{tf::PipeType::SERIAL, FilterFinal{mybuffer, graph}}  // 8th pipe
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  //for(auto r:result) {
  //  outputfile << r << '\n';
  //}
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// graph_pipeline_taskflow_9_pipes
std::chrono::microseconds graph_pipeline_taskflow_9_pipes(
  LevelGraph& graph, unsigned num_lines, unsigned num_threads) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  std::vector<int> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, 0, mybuffer, graph}}, // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 2nd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 3rd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 4th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 5th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 6th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 7th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 8th pipe
    tf::Pipe{tf::PipeType::SERIAL, FilterFinal{mybuffer, graph}}  // 9th pipe
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// graph_pipeline_taskflow_10_pipes
std::chrono::microseconds graph_pipeline_taskflow_10_pipes(
  LevelGraph& graph, unsigned num_lines, unsigned num_threads) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  std::vector<int> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, 0, mybuffer, graph}}, // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 2nd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 3rd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 4th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 5th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 6th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 7th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 8th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 9th pipe
    tf::Pipe{tf::PipeType::SERIAL, FilterFinal{mybuffer, graph}}  // 10th pipe
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


// graph_pipeline_taskflow_11_pipes
std::chrono::microseconds graph_pipeline_taskflow_11_pipes(
  LevelGraph& graph, unsigned num_lines, unsigned num_threads) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  std::vector<int> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, 0, mybuffer, graph}}, // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 2nd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 3rd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 4th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 5th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 6th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 7th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 8th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 9th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 10th pipe
    tf::Pipe{tf::PipeType::SERIAL, FilterFinal{mybuffer, graph}}  // 11th pipe
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// graph_pipeline_taskflow_12_pipes
std::chrono::microseconds graph_pipeline_taskflow_12_pipes(
  LevelGraph& graph, unsigned num_lines, unsigned num_threads) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  std::vector<int> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, 0, mybuffer, graph}}, // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 2nd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 3rd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 4th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 5th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 6th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 7th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 8th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 9th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 10th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 11th pipe
    tf::Pipe{tf::PipeType::SERIAL, FilterFinal{mybuffer, graph}}  // 12th pipe
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// graph_pipeline_taskflow_13_pipes
std::chrono::microseconds graph_pipeline_taskflow_13_pipes(
  LevelGraph& graph, unsigned num_lines, unsigned num_threads) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  std::vector<int> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, 0, mybuffer, graph}}, // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 2nd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 3rd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 4th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 5th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 6th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 7th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 8th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 9th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 10th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 11th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 12th pipe
    tf::Pipe{tf::PipeType::SERIAL, FilterFinal{mybuffer, graph}}  // 13th pipe
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// graph_pipeline_taskflow_14_pipes
std::chrono::microseconds graph_pipeline_taskflow_14_pipes(
  LevelGraph& graph, unsigned num_lines, unsigned num_threads) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  std::vector<int> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, 0, mybuffer, graph}}, // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 2nd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 3rd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 4th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 5th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 6th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 7th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 8th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 9th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 10th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 11th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 12th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 13th pipe
    tf::Pipe{tf::PipeType::SERIAL, FilterFinal{mybuffer, graph}}  // 14th pipe
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// graph_pipeline_taskflow_15_pipes
std::chrono::microseconds graph_pipeline_taskflow_15_pipes(
  LevelGraph& graph, unsigned num_lines, unsigned num_threads) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  std::vector<int> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, 0, mybuffer, graph}}, // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 2nd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 3rd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 4th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 5th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 6th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 7th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 8th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 9th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 10th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 11th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 12th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 13th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 14th pipe
    tf::Pipe{tf::PipeType::SERIAL, FilterFinal{mybuffer, graph}}  // 15th pipe
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// graph_pipeline_taskflow_16_pipes
std::chrono::microseconds graph_pipeline_taskflow_16_pipes(
  LevelGraph& graph, unsigned num_lines, unsigned num_threads) {

  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  std::vector<int> mybuffer(num_lines);

  //std::ofstream outputfile;
  //outputfile.open("./tf_16_pipes.txt", std::ofstream::app);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    tf::Pipe{tf::PipeType::SERIAL, Input{0, 0, mybuffer, graph}}, // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 2nd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 3rd pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 4th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 5th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 6th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 7th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 8th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 9th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 10th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 11th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 12th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 13th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 14th pipe
    tf::Pipe{tf::PipeType::SERIAL, Filter{mybuffer, graph}},      // 15th pipe
    tf::Pipe{tf::PipeType::SERIAL, FilterFinal{mybuffer, graph}}  // 16th pipe
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  //outputfile.close();

  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

std::chrono::microseconds measure_time_taskflow(
  LevelGraph& graph, size_t pipes, unsigned num_lines, unsigned num_threads) {

  std::chrono::microseconds elapsed;

  switch(pipes) {
    case 1:
      elapsed = graph_pipeline_taskflow_1_pipe(graph, num_lines, num_threads);
    break;

    case 2:
      elapsed = graph_pipeline_taskflow_2_pipes(graph, num_lines, num_threads);
    break;

    case 3:
      elapsed = graph_pipeline_taskflow_3_pipes(graph, num_lines, num_threads);
    break;

    case 4:
      elapsed = graph_pipeline_taskflow_4_pipes(graph, num_lines, num_threads);
    break;

    case 5:
      elapsed = graph_pipeline_taskflow_5_pipes(graph, num_lines, num_threads);
    break;

    case 6:
      elapsed = graph_pipeline_taskflow_6_pipes(graph, num_lines, num_threads);
    break;

    case 7:
      elapsed = graph_pipeline_taskflow_7_pipes(graph, num_lines, num_threads);
    break;

    case 8:
      elapsed = graph_pipeline_taskflow_8_pipes(graph, num_lines, num_threads);
    break;

    case 9:
      elapsed = graph_pipeline_taskflow_9_pipes(graph, num_lines, num_threads);
    break;

    case 10:
      elapsed = graph_pipeline_taskflow_10_pipes(graph, num_lines, num_threads);
    break;

    case 11:
      elapsed = graph_pipeline_taskflow_11_pipes(graph, num_lines, num_threads);
    break;

    case 12:
      elapsed = graph_pipeline_taskflow_12_pipes(graph, num_lines, num_threads);
    break;

    case 13:
      elapsed = graph_pipeline_taskflow_13_pipes(graph, num_lines, num_threads);
    break;

    case 14:
      elapsed = graph_pipeline_taskflow_14_pipes(graph, num_lines, num_threads);
    break;

    case 15:
      elapsed = graph_pipeline_taskflow_15_pipes(graph, num_lines, num_threads);
    break;

    case 16:
      elapsed = graph_pipeline_taskflow_16_pipes(graph, num_lines, num_threads);
    break;

    default:
      throw std::runtime_error("can support only up to 16 pipes");
    break;
  }
  return elapsed;
}

