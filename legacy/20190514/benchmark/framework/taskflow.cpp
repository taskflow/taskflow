#include <iostream>
#include <chrono>

#include <taskflow/taskflow.hpp>
#include "levelgraph.hpp"

void build_framework(LevelGraph& graph, tf::Framework& framework) {
  std::vector<std::vector<tf::Task>> tasks;

  tasks.resize(graph.level()); 
  for(size_t i=0; i<tasks.size(); ++i) {
    tasks[i].resize(graph.length());
  }

  for(size_t i=0; i<graph.length(); i++){
    Node& n = graph.node_at(graph.level()-1, i); 
    tasks[graph.level()-1][i] = framework.emplace([&](){ n.mark(); });
  }

  for(int l=graph.level()-2; l>=0 ; l--){
    for(size_t i=0; i<graph.length(); i++){
      Node& n = graph.node_at(l, i);
      tasks[l][i] = framework.emplace([&](){ n.mark();});
      for(size_t k=0; k<n._out_edges.size(); k++){
        tasks[l][i].precede(tasks[l+1][n._out_edges[k]]);
      } 
    }
  }

  // Add a reset task
  auto clear_task = framework.emplace([&](){
    graph.clear_graph();
  });

  for(size_t i=0; i<graph.length(); i++){
    tasks[graph.level()-1][i].precede(clear_task); 
  }
}


void traverse_level_graph_taskflow(LevelGraph& graph, unsigned num_threads, int repeat){
  tf::Executor tf(num_threads);
  tf::Framework framework;
  build_framework(graph, framework);
  tf.run_n(framework, repeat);
  tf.wait_for_all();
}

std::chrono::microseconds measure_time_taskflow(LevelGraph& graph, unsigned num_threads, int repeat){
  auto beg = std::chrono::high_resolution_clock::now();
  traverse_level_graph_taskflow(graph, num_threads, repeat);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

