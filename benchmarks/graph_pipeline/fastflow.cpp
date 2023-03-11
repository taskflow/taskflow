#include <ff/ff.hpp>
#include <chrono>
#include <iostream>
#include "levelgraph.hpp"


using namespace ff;

static size_t f_length = 0;
static size_t f_level = 0;
static size_t f_lev = 0;
static size_t f_len = 0;

struct mydata {
  LevelGraph* graph;
  int retval;
  mydata(LevelGraph* ptr, int val): graph(ptr), retval(val) {}
};



static inline mydata* STAGE1(mydata* task, ff_node* const) {
  //std::cout << "stage 1 " << task->retval << '\n';
  int uid = task->graph->node_at(f_lev, f_len).uid();
  int retval = work(uid);

  task->graph->node_at(uid/f_level, uid%f_length).set_value(retval);

  if (f_len == f_length-1) {
    ++f_lev;
    f_len = 0;
  }
  else {
    ++f_len;
  }
      
  task->retval = uid;

  //std::cout << "stage 1 " << task->retval << '\n';
  return task;
}
static inline mydata* STAGE2(mydata* task, ff_node* const) {
  //std::cout << "stage 2 " << task->retval << '\n';
  int val = work(task->retval);
  task->graph->node_at(task->retval/f_level, task->retval%f_length).set_value(val);
  //std::cout << "stage 2 " << task->retval << '\n';
  return task;
}

static inline mydata* STAGE3(mydata* task, ff_node* const) {
  //std::cout << "stage 3 " << task->retval << '\n';
  
  int retval = work(task->retval);
  
  int local_lev = task->retval/f_level;
  int local_len = task->retval%f_length;
 
  if (local_lev != 0) {
    int value_prev_level = 0;

    for (auto& in_edge : task->graph->node_at(local_lev, local_len)._in_edges) {
      value_prev_level += task->graph->node_at(local_lev-1, in_edge.first).get_value();
    }
    task->graph->node_at(local_lev, local_len).set_value(retval + value_prev_level);
  }
  else {
    task->graph->node_at(local_lev, local_len).set_value(retval);
  }

  task->retval = retval;
  //std::cout << "stage 3 " << task->retval << '\n';
  return task;
}

// graph_pipeline_fastflow_1_pipe
std::chrono::microseconds graph_pipeline_fastflow_1_pipe(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./ff_1_pipes.txt", std::ofstream::app);
  f_length = graph.length();
  f_level = graph.level();
  f_len = 0;
  f_lev = 0;

  int nodes = 0;
  auto ptr = &graph;
  auto beg = std::chrono::high_resolution_clock::now();
  
  ff_node_F<mydata> stage1(STAGE1);
  ff_Pipe<mydata> pipe(true, stage1);
  pipe.run_then_freeze();
  int node_counts = static_cast<int>(graph.get_node_count());
  while (nodes++ < node_counts) {
    //std::cout << nodes << '\n';
    pipe.offload(new mydata(ptr, 0));
  }
  pipe.offload(pipe.EOS);
  pipe.wait();

  //for (size_t i = 0; i < graph.level(); ++i) {
  //  for (size_t j = 0; j < graph.length(); ++j) {
  //    outputfile << graph.node_at(i, j).get_value() << '\n';
  //  }
  //}
  //outputfile.close();
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end-beg);
}

// graph_pipeline_fastflow_2_pipes
std::chrono::microseconds graph_pipeline_fastflow_2_pipes(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./ff_2_pipes.txt", std::ofstream::app);
  f_length = graph.length();
  f_level = graph.level();
  f_len = 0;
  f_lev = 0;

  int nodes = 0;
  auto ptr = &graph;
  auto beg = std::chrono::high_resolution_clock::now();
  
  ff_node_F<mydata> stage1(STAGE1), stage3(STAGE3);
  ff_Pipe<mydata, mydata> pipe(true, stage1, stage3);
  pipe.run_then_freeze();
  int node_counts = static_cast<int>(graph.get_node_count());
  while (nodes++ < node_counts) {
    //std::cout << nodes << '\n';
    pipe.offload(new mydata(ptr, 0));
  }
  pipe.offload(pipe.EOS);
  pipe.wait();

  //for (size_t i = 0; i < graph.level(); ++i) {
  //  for (size_t j = 0; j < graph.length(); ++j) {
  //    outputfile << graph.node_at(i, j).get_value() << '\n';
  //  }
  //}
  //outputfile.close();

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end-beg);
}


// graph_pipeline_fastflow_3_pipes
std::chrono::microseconds graph_pipeline_fastflow_3_pipes(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./ff_3_pipes.txt", std::ofstream::app);
  f_length = graph.length();
  f_level = graph.level();
  f_len = 0;
  f_lev = 0;

  int nodes = 0;
  auto ptr = &graph;
  auto beg = std::chrono::high_resolution_clock::now();
  
  ff_node_F<mydata> stage1(STAGE1);
  ff_node_F<mydata> stage2(STAGE2);
  ff_node_F<mydata> stage3(STAGE3);

  ff_Pipe<mydata> pipe(true, stage1, stage2, stage3);
  
  pipe.run_then_freeze();
  int node_counts = static_cast<int>(graph.get_node_count());
  while (nodes++ < node_counts) {
    //std::cout << nodes << '\n';
    pipe.offload(new mydata(ptr, 0));
  }
  pipe.offload(pipe.EOS);
  pipe.wait();

  //for (size_t i = 0; i < graph.level(); ++i) {
  //  for (size_t j = 0; j < graph.length(); ++j) {
  //    outputfile << graph.node_at(i, j).get_value() << '\n';
  //  }
  //}
  //outputfile.close();

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end-beg);
}

// graph_pipeline_fastflow_4_pipes
std::chrono::microseconds graph_pipeline_fastflow_4_pipes(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./ff_4_pipes.txt", std::ofstream::app);
  f_length = graph.length();
  f_level = graph.level();
  f_len = 0;
  f_lev = 0;

  int nodes = 0;
  auto ptr = &graph;
  auto beg = std::chrono::high_resolution_clock::now();
  
  ff_node_F<mydata> stage1(STAGE1);
  ff_node_F<mydata> stage2(STAGE2);
  ff_node_F<mydata> stage3(STAGE2);
  ff_node_F<mydata> stage4(STAGE3);

  ff_Pipe<mydata> pipe(true, stage1, stage2, stage3, stage4);
  
  pipe.run_then_freeze();
  int node_counts = static_cast<int>(graph.get_node_count());
  while (nodes++ < node_counts) {
    //std::cout << nodes << '\n';
    pipe.offload(new mydata(ptr, 0));
  }
  pipe.offload(pipe.EOS);
  pipe.wait();

  //for (size_t i = 0; i < graph.level(); ++i) {
  //  for (size_t j = 0; j < graph.length(); ++j) {
  //    outputfile << graph.node_at(i, j).get_value() << '\n';
  //  }
  //}
  //outputfile.close();

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end-beg);
}

// graph_pipeline_fastflow_5_pipes
std::chrono::microseconds graph_pipeline_fastflow_5_pipes(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./ff_5_pipes.txt", std::ofstream::app);
  f_length = graph.length();
  f_level = graph.level();
  f_len = 0;
  f_lev = 0;

  int nodes = 0;
  auto ptr = &graph;
  auto beg = std::chrono::high_resolution_clock::now();
  
  ff_node_F<mydata> stage1(STAGE1);
  ff_node_F<mydata> stage2(STAGE2);
  ff_node_F<mydata> stage3(STAGE2);
  ff_node_F<mydata> stage4(STAGE2);
  ff_node_F<mydata> stage5(STAGE3);

  ff_Pipe<mydata> pipe(true, stage1, stage2, stage3, stage4, stage5);
  
  pipe.run_then_freeze();
  int node_counts = static_cast<int>(graph.get_node_count());
  while (nodes++ < node_counts) {
    //std::cout << nodes << '\n';
    pipe.offload(new mydata(ptr, 0));
  }
  pipe.offload(pipe.EOS);
  pipe.wait();

  //for (size_t i = 0; i < graph.level(); ++i) {
  //  for (size_t j = 0; j < graph.length(); ++j) {
  //    outputfile << graph.node_at(i, j).get_value() << '\n';
  //  }
  //}
  //outputfile.close();

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end-beg);
}

// graph_pipeline_fastflow_6_pipes
std::chrono::microseconds graph_pipeline_fastflow_6_pipes(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./ff_6_pipes.txt", std::ofstream::app);
  f_length = graph.length();
  f_level = graph.level();
  f_len = 0;
  f_lev = 0;

  int nodes = 0;
  auto ptr = &graph;
  auto beg = std::chrono::high_resolution_clock::now();
  
  ff_node_F<mydata> stage1(STAGE1);
  ff_node_F<mydata> stage2(STAGE2);
  ff_node_F<mydata> stage3(STAGE2);
  ff_node_F<mydata> stage4(STAGE2);
  ff_node_F<mydata> stage5(STAGE2);
  ff_node_F<mydata> stage6(STAGE3);

  ff_Pipe<mydata> pipe(true, stage1, stage2, stage3, stage4, stage5, stage6);
  
  pipe.run_then_freeze();
  int node_counts = static_cast<int>(graph.get_node_count());
  while (nodes++ < node_counts) {
    //std::cout << nodes << '\n';
    pipe.offload(new mydata(ptr, 0));
  }
  pipe.offload(pipe.EOS);
  pipe.wait();

  //for (size_t i = 0; i < graph.level(); ++i) {
  //  for (size_t j = 0; j < graph.length(); ++j) {
  //    outputfile << graph.node_at(i, j).get_value() << '\n';
  //  }
  //}
  //outputfile.close();

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end-beg);
}

// graph_pipeline_fastflow_7_pipes
std::chrono::microseconds graph_pipeline_fastflow_7_pipes(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./ff_7_pipes.txt", std::ofstream::app);
  f_length = graph.length();
  f_level = graph.level();
  f_len = 0;
  f_lev = 0;

  int nodes = 0;
  auto ptr = &graph;
  auto beg = std::chrono::high_resolution_clock::now();
  
  ff_node_F<mydata> stage1(STAGE1);
  ff_node_F<mydata> stage2(STAGE2);
  ff_node_F<mydata> stage3(STAGE2);
  ff_node_F<mydata> stage4(STAGE2);
  ff_node_F<mydata> stage5(STAGE2);
  ff_node_F<mydata> stage6(STAGE2);
  ff_node_F<mydata> stage7(STAGE3);

  ff_Pipe<mydata> pipe(true, stage1, stage2, stage3, stage4, stage5, stage6, stage7);
  
  pipe.run_then_freeze();
  int node_counts = static_cast<int>(graph.get_node_count());
  while (nodes++ < node_counts) {
    //std::cout << nodes << '\n';
    pipe.offload(new mydata(ptr, 0));
  }
  pipe.offload(pipe.EOS);
  pipe.wait();

  //for (size_t i = 0; i < graph.level(); ++i) {
  //  for (size_t j = 0; j < graph.length(); ++j) {
  //    outputfile << graph.node_at(i, j).get_value() << '\n';
  //  }
  //}
  //outputfile.close();

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end-beg);
}

// graph_pipeline_fastflow_8_pipes
std::chrono::microseconds graph_pipeline_fastflow_8_pipes(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./ff_8_pipes.txt", std::ofstream::app);
  f_length = graph.length();
  f_level = graph.level();
  f_len = 0;
  f_lev = 0;

  int nodes = 0;
  auto ptr = &graph;
  auto beg = std::chrono::high_resolution_clock::now();
  
  ff_node_F<mydata> stage1(STAGE1);
  ff_node_F<mydata> stage2(STAGE2);
  ff_node_F<mydata> stage3(STAGE2);
  ff_node_F<mydata> stage4(STAGE2);
  ff_node_F<mydata> stage5(STAGE2);
  ff_node_F<mydata> stage6(STAGE2);
  ff_node_F<mydata> stage7(STAGE2);
  ff_node_F<mydata> stage8(STAGE3);

  ff_Pipe<mydata> pipe(true, stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8);
  
  pipe.run_then_freeze();
  int node_counts = static_cast<int>(graph.get_node_count());
  while (nodes++ < node_counts) {
    //std::cout << nodes << '\n';
    pipe.offload(new mydata(ptr, 0));
  }
  pipe.offload(pipe.EOS);
  pipe.wait();

  //for (size_t i = 0; i < graph.level(); ++i) {
  //  for (size_t j = 0; j < graph.length(); ++j) {
  //    outputfile << graph.node_at(i, j).get_value() << '\n';
  //  }
  //}
  //outputfile.close();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end-beg);
}

// graph_pipeline_fastflow_9_pipes
std::chrono::microseconds graph_pipeline_fastflow_9_pipes(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./ff_9_pipes.txt", std::ofstream::app);
  f_length = graph.length();
  f_level = graph.level();
  f_len = 0;
  f_lev = 0;

  int nodes = 0;
  auto ptr = &graph;
  auto beg = std::chrono::high_resolution_clock::now();
  
  ff_node_F<mydata> stage1(STAGE1);
  ff_node_F<mydata> stage2(STAGE2);
  ff_node_F<mydata> stage3(STAGE2);
  ff_node_F<mydata> stage4(STAGE2);
  ff_node_F<mydata> stage5(STAGE2);
  ff_node_F<mydata> stage6(STAGE2);
  ff_node_F<mydata> stage7(STAGE2);
  ff_node_F<mydata> stage8(STAGE2);
  ff_node_F<mydata> stage9(STAGE3);

  ff_Pipe<mydata> pipe(true, stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8, stage9);
  
  pipe.run_then_freeze();
  int node_counts = static_cast<int>(graph.get_node_count());
  while (nodes++ < node_counts) {
    //std::cout << nodes << '\n';
    pipe.offload(new mydata(ptr, 0));
  }
  pipe.offload(pipe.EOS);
  pipe.wait();

  //for (size_t i = 0; i < graph.level(); ++i) {
  //  for (size_t j = 0; j < graph.length(); ++j) {
  //    outputfile << graph.node_at(i, j).get_value() << '\n';
  //  }
  //}
  //outputfile.close();

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end-beg);
}

// graph_pipeline_fastflow_10_pipes
std::chrono::microseconds graph_pipeline_fastflow_10_pipes(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./ff_10_pipes.txt", std::ofstream::app);
  f_length = graph.length();
  f_level = graph.level();
  f_len = 0;
  f_lev = 0;

  int nodes = 0;
  auto ptr = &graph;
  auto beg = std::chrono::high_resolution_clock::now();
  
  ff_node_F<mydata> stage1(STAGE1);
  ff_node_F<mydata> stage2(STAGE2);
  ff_node_F<mydata> stage3(STAGE2);
  ff_node_F<mydata> stage4(STAGE2);
  ff_node_F<mydata> stage5(STAGE2);
  ff_node_F<mydata> stage6(STAGE2);
  ff_node_F<mydata> stage7(STAGE2);
  ff_node_F<mydata> stage8(STAGE2);
  ff_node_F<mydata> stage9(STAGE2);
  ff_node_F<mydata> stage10(STAGE3);

  ff_Pipe<mydata> pipe(true, stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8, stage9, stage10);
  
  pipe.run_then_freeze();
  int node_counts = static_cast<int>(graph.get_node_count());
  while (nodes++ < node_counts) {
    //std::cout << nodes << '\n';
    pipe.offload(new mydata(ptr, 0));
  }
  pipe.offload(pipe.EOS);
  pipe.wait();

  //for (size_t i = 0; i < graph.level(); ++i) {
  //  for (size_t j = 0; j < graph.length(); ++j) {
  //    outputfile << graph.node_at(i, j).get_value() << '\n';
  //  }
  //}
  //outputfile.close();

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end-beg);
}

// graph_pipeline_fastflow_11_pipes
std::chrono::microseconds graph_pipeline_fastflow_11_pipes(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./ff_11_pipes.txt", std::ofstream::app);
  f_length = graph.length();
  f_level = graph.level();
  f_len = 0;
  f_lev = 0;

  int nodes = 0;
  auto ptr = &graph;
  auto beg = std::chrono::high_resolution_clock::now();
  
  ff_node_F<mydata> stage1(STAGE1);
  ff_node_F<mydata> stage2(STAGE2);
  ff_node_F<mydata> stage3(STAGE2);
  ff_node_F<mydata> stage4(STAGE2);
  ff_node_F<mydata> stage5(STAGE2);
  ff_node_F<mydata> stage6(STAGE2);
  ff_node_F<mydata> stage7(STAGE2);
  ff_node_F<mydata> stage8(STAGE2);
  ff_node_F<mydata> stage9(STAGE2);
  ff_node_F<mydata> stage10(STAGE2);
  ff_node_F<mydata> stage11(STAGE3);

  ff_Pipe<mydata> pipe(true, stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8, stage9, stage10, stage11);
  
  pipe.run_then_freeze();
  int node_counts = static_cast<int>(graph.get_node_count());
  while (nodes++ < node_counts) {
    //std::cout << nodes << '\n';
    pipe.offload(new mydata(ptr, 0));
  }
  pipe.offload(pipe.EOS);
  pipe.wait();

  //for (size_t i = 0; i < graph.level(); ++i) {
  //  for (size_t j = 0; j < graph.length(); ++j) {
  //    outputfile << graph.node_at(i, j).get_value() << '\n';
  //  }
  //}
  //outputfile.close();

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end-beg);
}

// graph_pipeline_fastflow_12_pipes
std::chrono::microseconds graph_pipeline_fastflow_12_pipes(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./ff_12_pipes.txt", std::ofstream::app);
  f_length = graph.length();
  f_level = graph.level();
  f_len = 0;
  f_lev = 0;

  int nodes = 0;
  auto ptr = &graph;
  auto beg = std::chrono::high_resolution_clock::now();
  
  ff_node_F<mydata> stage1(STAGE1);
  ff_node_F<mydata> stage2(STAGE2);
  ff_node_F<mydata> stage3(STAGE2);
  ff_node_F<mydata> stage4(STAGE2);
  ff_node_F<mydata> stage5(STAGE2);
  ff_node_F<mydata> stage6(STAGE2);
  ff_node_F<mydata> stage7(STAGE2);
  ff_node_F<mydata> stage8(STAGE2);
  ff_node_F<mydata> stage9(STAGE2);
  ff_node_F<mydata> stage10(STAGE2);
  ff_node_F<mydata> stage11(STAGE2);
  ff_node_F<mydata> stage12(STAGE3);

  ff_Pipe<mydata> pipe(true, stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8, stage9, stage10, stage11, stage12);
  
  pipe.run_then_freeze();
  int node_counts = static_cast<int>(graph.get_node_count());
  while (nodes++ < node_counts) {
    //std::cout << nodes << '\n';
    pipe.offload(new mydata(ptr, 0));
  }
  pipe.offload(pipe.EOS);
  pipe.wait();

  //for (size_t i = 0; i < graph.level(); ++i) {
  //  for (size_t j = 0; j < graph.length(); ++j) {
  //    outputfile << graph.node_at(i, j).get_value() << '\n';
  //  }
  //}
  //outputfile.close();

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end-beg);
}

// graph_pipeline_fastflow_13_pipes
std::chrono::microseconds graph_pipeline_fastflow_13_pipes(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./ff_13_pipes.txt", std::ofstream::app);
  f_length = graph.length();
  f_level = graph.level();
  f_len = 0;
  f_lev = 0;

  int nodes = 0;
  auto ptr = &graph;
  auto beg = std::chrono::high_resolution_clock::now();
  
  ff_node_F<mydata> stage1(STAGE1);
  ff_node_F<mydata> stage2(STAGE2);
  ff_node_F<mydata> stage3(STAGE2);
  ff_node_F<mydata> stage4(STAGE2);
  ff_node_F<mydata> stage5(STAGE2);
  ff_node_F<mydata> stage6(STAGE2);
  ff_node_F<mydata> stage7(STAGE2);
  ff_node_F<mydata> stage8(STAGE2);
  ff_node_F<mydata> stage9(STAGE2);
  ff_node_F<mydata> stage10(STAGE2);
  ff_node_F<mydata> stage11(STAGE2);
  ff_node_F<mydata> stage12(STAGE2);
  ff_node_F<mydata> stage13(STAGE3);

  ff_Pipe<mydata> pipe(true, stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8, stage9, stage10, stage11, stage12, stage13);
  
  pipe.run_then_freeze();
  int node_counts = static_cast<int>(graph.get_node_count());
  while (nodes++ < node_counts) {
    //std::cout << nodes << '\n';
    pipe.offload(new mydata(ptr, 0));
  }
  pipe.offload(pipe.EOS);
  pipe.wait();

  //for (size_t i = 0; i < graph.level(); ++i) {
  //  for (size_t j = 0; j < graph.length(); ++j) {
  //    outputfile << graph.node_at(i, j).get_value() << '\n';
  //  }
  //}
  //outputfile.close();

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end-beg);
}

// graph_pipeline_fastflow_14_pipes
std::chrono::microseconds graph_pipeline_fastflow_14_pipes(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./ff_14_pipes.txt", std::ofstream::app);
  f_length = graph.length();
  f_level = graph.level();
  f_len = 0;
  f_lev = 0;

  int nodes = 0;
  auto ptr = &graph;
  auto beg = std::chrono::high_resolution_clock::now();
  
  ff_node_F<mydata> stage1(STAGE1);
  ff_node_F<mydata> stage2(STAGE2);
  ff_node_F<mydata> stage3(STAGE2);
  ff_node_F<mydata> stage4(STAGE2);
  ff_node_F<mydata> stage5(STAGE2);
  ff_node_F<mydata> stage6(STAGE2);
  ff_node_F<mydata> stage7(STAGE2);
  ff_node_F<mydata> stage8(STAGE2);
  ff_node_F<mydata> stage9(STAGE2);
  ff_node_F<mydata> stage10(STAGE2);
  ff_node_F<mydata> stage11(STAGE2);
  ff_node_F<mydata> stage12(STAGE2);
  ff_node_F<mydata> stage13(STAGE2);
  ff_node_F<mydata> stage14(STAGE3);

  ff_Pipe<mydata> pipe(true, stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8, stage9, stage10, stage11, stage12, stage13, stage14);
  
  pipe.run_then_freeze();
  int node_counts = static_cast<int>(graph.get_node_count());
  while (nodes++ < node_counts) {
    //std::cout << nodes << '\n';
    pipe.offload(new mydata(ptr, 0));
  }
  pipe.offload(pipe.EOS);
  pipe.wait();

  //for (size_t i = 0; i < graph.level(); ++i) {
  //  for (size_t j = 0; j < graph.length(); ++j) {
  //    outputfile << graph.node_at(i, j).get_value() << '\n';
  //  }
  //}
  //outputfile.close();

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end-beg);
}

// graph_pipeline_fastflow_15_pipes
std::chrono::microseconds graph_pipeline_fastflow_15_pipes(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./ff_15_pipes.txt", std::ofstream::app);
  f_length = graph.length();
  f_level = graph.level();
  f_len = 0;
  f_lev = 0;

  int nodes = 0;
  auto ptr = &graph;
  auto beg = std::chrono::high_resolution_clock::now();
  
  ff_node_F<mydata> stage1(STAGE1);
  ff_node_F<mydata> stage2(STAGE2);
  ff_node_F<mydata> stage3(STAGE2);
  ff_node_F<mydata> stage4(STAGE2);
  ff_node_F<mydata> stage5(STAGE2);
  ff_node_F<mydata> stage6(STAGE2);
  ff_node_F<mydata> stage7(STAGE2);
  ff_node_F<mydata> stage8(STAGE2);
  ff_node_F<mydata> stage9(STAGE2);
  ff_node_F<mydata> stage10(STAGE2);
  ff_node_F<mydata> stage11(STAGE2);
  ff_node_F<mydata> stage12(STAGE2);
  ff_node_F<mydata> stage13(STAGE2);
  ff_node_F<mydata> stage14(STAGE2);
  ff_node_F<mydata> stage15(STAGE3);

  ff_Pipe<mydata> pipe(true, stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8, stage9, stage10, stage11, stage12, stage13, stage14, stage15);
  
  pipe.run_then_freeze();
  int node_counts = static_cast<int>(graph.get_node_count());
  while (nodes++ < node_counts) {
    //std::cout << nodes << '\n';
    pipe.offload(new mydata(ptr, 0));
  }
  pipe.offload(pipe.EOS);
  pipe.wait();

  //for (size_t i = 0; i < graph.level(); ++i) {
  //  for (size_t j = 0; j < graph.length(); ++j) {
  //    outputfile << graph.node_at(i, j).get_value() << '\n';
  //  }
  //}
  //outputfile.close();

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end-beg);
}

// graph_pipeline_fastflow_16_pipes
std::chrono::microseconds graph_pipeline_fastflow_16_pipes(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./ff_16_pipes.txt", std::ofstream::app);
  f_length = graph.length();
  f_level = graph.level();
  f_len = 0;
  f_lev = 0;

  int nodes = 0;
  auto ptr = &graph;
  auto beg = std::chrono::high_resolution_clock::now();
  
  ff_node_F<mydata> stage1(STAGE1);
  ff_node_F<mydata> stage2(STAGE2);
  ff_node_F<mydata> stage3(STAGE2);
  ff_node_F<mydata> stage4(STAGE2);
  ff_node_F<mydata> stage5(STAGE2);
  ff_node_F<mydata> stage6(STAGE2);
  ff_node_F<mydata> stage7(STAGE2);
  ff_node_F<mydata> stage8(STAGE2);
  ff_node_F<mydata> stage9(STAGE2);
  ff_node_F<mydata> stage10(STAGE2);
  ff_node_F<mydata> stage11(STAGE2);
  ff_node_F<mydata> stage12(STAGE2);
  ff_node_F<mydata> stage13(STAGE2);
  ff_node_F<mydata> stage14(STAGE2);
  ff_node_F<mydata> stage15(STAGE2);
  ff_node_F<mydata> stage16(STAGE3);

  ff_Pipe<mydata> pipe(true, stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8, stage9, stage10, stage11, stage12, stage13, stage14, stage15, stage16);
  
  pipe.run_then_freeze();
  int node_counts = static_cast<int>(graph.get_node_count());
  while (nodes++ < node_counts) {
    //std::cout << nodes << '\n';
    pipe.offload(new mydata(ptr, 0));
  }
  pipe.offload(pipe.EOS);
  pipe.wait();

  //for (size_t i = 0; i < graph.level(); ++i) {
  //  for (size_t j = 0; j < graph.length(); ++j) {
  //    outputfile << graph.node_at(i, j).get_value() << '\n';
  //  }
  //}
  //outputfile.close();

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end-beg);
}


std::chrono::microseconds measure_time_fastflow(
  LevelGraph& graph, size_t pipes) {

  std::chrono::microseconds elapsed;
  

  switch(pipes) {
    case 1:
      elapsed = graph_pipeline_fastflow_1_pipe(graph);
    break;
    
    case 2:
      elapsed = graph_pipeline_fastflow_2_pipes(graph);
    break;
        
    case 3:
      elapsed = graph_pipeline_fastflow_3_pipes(graph);
    break;
     
    case 4:
      elapsed = graph_pipeline_fastflow_4_pipes(graph);
    break;
     
    case 5:
      elapsed = graph_pipeline_fastflow_5_pipes(graph);
    break;
    
    case 6:
      elapsed = graph_pipeline_fastflow_6_pipes(graph);
    break;
     
    case 7:
      elapsed = graph_pipeline_fastflow_7_pipes(graph);
    break;
    
    case 8:
      elapsed = graph_pipeline_fastflow_8_pipes(graph);
    break;
    
    case 9:
      elapsed = graph_pipeline_fastflow_9_pipes(graph);
    break;
    
    case 10:
      elapsed = graph_pipeline_fastflow_10_pipes(graph);
    break;
    
    case 11:
      elapsed = graph_pipeline_fastflow_11_pipes(graph);
    break;
    
    case 12:
      elapsed = graph_pipeline_fastflow_12_pipes(graph);
    break;
    
    case 13:
      elapsed = graph_pipeline_fastflow_13_pipes(graph);
    break;
    
    case 14:
      elapsed = graph_pipeline_fastflow_14_pipes(graph);
    break;
    
    case 15:
      elapsed = graph_pipeline_fastflow_15_pipes(graph);
    break;
    
    case 16:
      elapsed = graph_pipeline_fastflow_16_pipes(graph);
    break;
    
  }
 
  return elapsed; 
}



