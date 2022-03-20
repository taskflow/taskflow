#include "levelgraph.hpp"
#include <tbb/pipeline.h>
#include <tbb/tick_count.h>
#include <tbb/tbb_allocator.h>
#include <tbb/global_control.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
//#include "matrix_calculation.hpp"

//#include "../../3rd-party/tbb/examples/common/utility/utility.h"
//#include "../../3rd-party/tbb/examples/common/utility/get_default_num_threads.h"

size_t length = 0;
size_t level = 0;
size_t len = 0;
size_t lev = 0;



// Filter for one filter only
class MyFunc {
public:
  MyFunc(LevelGraph& g) : graph(g) {}

  ~MyFunc(){}

  LevelGraph& graph;

  void operator()(tbb::flow_control& fc) const {
    if (len == 0 && lev == level) {
      fc.stop();
    }
    else {
      int uid = graph.node_at(lev, len).uid();

      int retval = work(uid);

      graph.node_at(uid/level, uid%length).set_value(retval);

      if (len == length-1) {
        ++lev;
        len = 0;
      }
      else {
        ++len;
      }
    }
  }
};

// Filter 1
class MyInputFunc {
public:
  MyInputFunc(LevelGraph& graph) : graph(graph) {}

  ~MyInputFunc(){}

  LevelGraph& graph;

  int operator()(tbb::flow_control& fc) const {
    int uid = 0;
    if (len == 0 && lev == level) {
      fc.stop();
      return -1;
    }
    else {
      uid = graph.node_at(lev, len).uid();

      work(uid);
      //std::cout << "len = " << len << ", lev = " << lev << ", uid = " << uid
      //          << ", val = " << graph.node_at(uid/level, uid%length).get_value()
      //          << ", uid/level = " << uid << '/' << level << " = " << uid/level
      //          << ' ' << uid%length << '\n';

      if (len == length-1) {
        ++lev;
        len = 0;
      }
      else {
        ++len;
      }
      return uid;
    }
  }
};

// Filter middle
class MyTransformFunc {
public:
  int operator()(int input) const {
    int val = input;
    work(val);
    return val;
  }
};

// Filter last
class MyOutputFunc {
public:
  MyOutputFunc(LevelGraph& g) : graph(g) {}

  LevelGraph& graph;

  void operator()(int input) const {
    int val = input;
    int retval = work(val);

    int local_lev = val/level;
    int local_len = val%length;

    if (local_lev != 0) {
      int value_prev_level = 0;

      for (auto& in_edge : graph.node_at(local_lev, local_len)._in_edges) {
        value_prev_level += graph.node_at(local_lev-1, in_edge.first).get_value();
      }
      graph.node_at(local_lev, local_len).set_value(retval + value_prev_level);
    }
    else {
      graph.node_at(local_lev, local_len).set_value(retval);
      //std::cout << "set value = " << retval << '\n';
      //std::cout << "get value = " << graph.node_at(local_lev, local_len).get_value() << '\n';
    }

    //std::ofstream outputfile;
    //outputfile.open("./tbb_result_.txt", std::ofstream::app);
    //outputfile << graph.node_at(local_lev, local_len).get_value() << '\n';
    //outputfile.close();
  }
};


// graph_pipeline_tbb_1_pipe
void graph_pipeline_tbb_1_pipe(LevelGraph& graph, unsigned num_lines) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, void>(
      tbb::filter::serial_in_order, MyFunc(graph))
  );
}

// graph_pipeline_tbb_2_pipes
void graph_pipeline_tbb_2_pipes(LevelGraph& graph, unsigned num_lines) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(tbb::filter::serial_in_order, MyInputFunc(graph))  & // 1st filter
    tbb::make_filter<int, void>(tbb::filter::serial_in_order, MyOutputFunc(graph))   // 2nd filter
  );
}

// graph_pipeline_tbb_3_pipes
void graph_pipeline_tbb_3_pipes(LevelGraph& graph, unsigned num_lines) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(tbb::filter::serial_in_order, MyInputFunc(graph))  & // 1st filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 2nd filter
    tbb::make_filter<int, void>(tbb::filter::serial_in_order, MyOutputFunc(graph))   // 3rd filter
  );
}

// graph_pipeline_tbb_4_pipes
void graph_pipeline_tbb_4_pipes(LevelGraph& graph, unsigned num_lines) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(tbb::filter::serial_in_order, MyInputFunc(graph))  & // 1st filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 2nd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 3rd filter
    tbb::make_filter<int, void>(tbb::filter::serial_in_order, MyOutputFunc(graph))   // 4th filter
  );
}

// graph_pipeline_tbb_5_pipes
void graph_pipeline_tbb_5_pipes(LevelGraph& graph, unsigned num_lines) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(tbb::filter::serial_in_order, MyInputFunc(graph))  & // 1st filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 2nd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 3rd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 4th filter
    tbb::make_filter<int, void>(tbb::filter::serial_in_order, MyOutputFunc(graph))   // 5th filter
  );
}

// graph_pipeline_tbb_6_pipes
void graph_pipeline_tbb_6_pipes(LevelGraph& graph, unsigned num_lines) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(tbb::filter::serial_in_order, MyInputFunc(graph))  & // 1st filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 2nd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 3rd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 4th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 5th filter
    tbb::make_filter<int, void>(tbb::filter::serial_in_order, MyOutputFunc(graph))   // 6th filter
  );
}

// graph_pipeline_tbb_7_pipes
void graph_pipeline_tbb_7_pipes(LevelGraph& graph, unsigned num_lines) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(tbb::filter::serial_in_order, MyInputFunc(graph))  & // 1st filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 2nd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 3rd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 4th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 5th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 6th filter
    tbb::make_filter<int, void>(tbb::filter::serial_in_order, MyOutputFunc(graph))   // 7th filter
  );
}

// graph_pipeline_tbb_8_pipes
void graph_pipeline_tbb_8_pipes(LevelGraph& graph, unsigned num_lines) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(tbb::filter::serial_in_order, MyInputFunc(graph))  & // 1st filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 2nd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 3rd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 4th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 5th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 6th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 7th filter
    tbb::make_filter<int, void>(tbb::filter::serial_in_order, MyOutputFunc(graph))   // 8th filter
  );
}

// graph_pipeline_tbb_9_pipes
void graph_pipeline_tbb_9_pipes(LevelGraph& graph, unsigned num_lines) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(tbb::filter::serial_in_order, MyInputFunc(graph))  & // 1st filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 2nd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 3rd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 4th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 5th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 6th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 7th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 8th filter
    tbb::make_filter<int, void>(tbb::filter::serial_in_order, MyOutputFunc(graph))   // 9th filter
  );
}

// graph_pipeline_tbb_10_pipes
void graph_pipeline_tbb_10_pipes(LevelGraph& graph, unsigned num_lines) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(tbb::filter::serial_in_order, MyInputFunc(graph))  & // 1st filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 2nd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 3rd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 4th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 5th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 6th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 7th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 8th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 9th filter
    tbb::make_filter<int, void>(tbb::filter::serial_in_order, MyOutputFunc(graph))   // 10th filter
  );
}

// graph_pipeline_tbb_11_pipes
void graph_pipeline_tbb_11_pipes(LevelGraph& graph, unsigned num_lines) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(tbb::filter::serial_in_order, MyInputFunc(graph))  & // 1st filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 2nd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 3rd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 4th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 5th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 6th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 7th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 8th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 9th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 10th filter
    tbb::make_filter<int, void>(tbb::filter::serial_in_order, MyOutputFunc(graph))   // 11th filter
  );
}

// graph_pipeline_tbb_12_pipes
void graph_pipeline_tbb_12_pipes(LevelGraph& graph, unsigned num_lines) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(tbb::filter::serial_in_order, MyInputFunc(graph))  & // 1st filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 2nd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 3rd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 4th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 5th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 6th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 7th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 8th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 9th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 10th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 11th filter
    tbb::make_filter<int, void>(tbb::filter::serial_in_order, MyOutputFunc(graph))   // 12th filter
  );
}

// graph_pipeline_tbb_13_pipes
void graph_pipeline_tbb_13_pipes(LevelGraph& graph, unsigned num_lines) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(tbb::filter::serial_in_order, MyInputFunc(graph))  & // 1st filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 2nd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 3rd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 4th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 5th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 6th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 7th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 8th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 9th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 10th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 11th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 12th filter
    tbb::make_filter<int, void>(tbb::filter::serial_in_order, MyOutputFunc(graph))   // 13th filter
  );
}

// graph_pipeline_tbb_14_pipes
void graph_pipeline_tbb_14_pipes(LevelGraph& graph, unsigned num_lines) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(tbb::filter::serial_in_order, MyInputFunc(graph))  & // 1st filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 2nd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 3rd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 4th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 5th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 6th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 7th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 8th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 9th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 10th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 11th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 12th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 13th filter
    tbb::make_filter<int, void>(tbb::filter::serial_in_order, MyOutputFunc(graph))   // 14th filter
  );
}

// graph_pipeline_tbb_15_pipes
void graph_pipeline_tbb_15_pipes(LevelGraph& graph, unsigned num_lines) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(tbb::filter::serial_in_order, MyInputFunc(graph))  & // 1st filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 2nd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 3rd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 4th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 5th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 6th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 7th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 8th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 9th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 10th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 11th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 12th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 13th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 14th filter
    tbb::make_filter<int, void>(tbb::filter::serial_in_order, MyOutputFunc(graph))   // 15th filter
  );
}

// graph_pipeline_tbb_16_pipes
void graph_pipeline_tbb_16_pipes(LevelGraph& graph, unsigned num_lines) {
  tbb::parallel_pipeline(
    num_lines,
    tbb::make_filter<void, int>(tbb::filter::serial_in_order, MyInputFunc(graph))  & // 1st filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 2nd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 3rd filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 4th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 5th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 6th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 7th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 8th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 9th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 10th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 11th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 12th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 13th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 14th filter
    tbb::make_filter<int, int>(tbb::filter::serial_in_order,  MyTransformFunc())   & // 15th filter
    tbb::make_filter<int, void>(tbb::filter::serial_in_order, MyOutputFunc(graph))   // 16th filter
  );
}

std::chrono::microseconds measure_time_tbb(
  LevelGraph& graph, size_t pipes, unsigned num_lines, unsigned num_threads) {

  //utility::thread_number_range threads( utility::get_default_num_threads, 0);
  tbb::global_control c(tbb::global_control::max_allowed_parallelism, num_threads);

  auto beg = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();

  length = graph.length();
  level = graph.level();
  len = 0;
  lev = 0;

  switch(pipes) {
    case 1:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_tbb_1_pipe(graph, num_lines);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 2:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_tbb_2_pipes(graph, num_lines);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 3:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_tbb_3_pipes(graph, num_lines);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 4:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_tbb_4_pipes(graph, num_lines);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 5:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_tbb_5_pipes(graph, num_lines);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 6:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_tbb_6_pipes(graph, num_lines);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 7:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_tbb_7_pipes(graph, num_lines);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 8:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_tbb_8_pipes(graph, num_lines);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 9:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_tbb_9_pipes(graph, num_lines);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 10:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_tbb_10_pipes(graph, num_lines);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 11:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_tbb_11_pipes(graph, num_lines);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 12:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_tbb_12_pipes(graph, num_lines);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 13:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_tbb_13_pipes(graph, num_lines);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 14:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_tbb_14_pipes(graph, num_lines);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 15:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_tbb_15_pipes(graph, num_lines);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 16:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_tbb_16_pipes(graph, num_lines);
      end = std::chrono::high_resolution_clock::now();
    break;

    default:
      throw std::runtime_error("can support only up to 16 pipes");
    break;
  }

  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
