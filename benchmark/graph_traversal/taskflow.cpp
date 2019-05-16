#include <iostream>
#include <chrono>

#include <taskflow/taskflow.hpp>
#include "levelgraph.hpp"

struct TF {
  
  TF(LevelGraph& graph, unsigned num_threads) : executor(num_threads) {

    tasks.resize(graph.level()); 
    for(size_t i=0; i<tasks.size(); ++i) {
      tasks[i].resize(graph.length());
    }
  
    for(size_t i=0; i<graph.length(); i++){
      Node& n = graph.node_at(graph.level()-1, i); 
      tasks[graph.level()-1][i] = taskflow.emplace([&](){ n.mark(); });
    }
  
    for(int l=graph.level()-2; l>=0 ; l--){
      for(size_t i=0; i<graph.length(); i++){
        Node& n = graph.node_at(l, i);
        tasks[l][i] = taskflow.emplace([&](){ n.mark();});
        for(size_t k=0; k<n._out_edges.size(); k++){
          tasks[l][i].precede(tasks[l+1][n._out_edges[k]]);
        } 
      }
    }
  }

  void run() {
    executor.run(taskflow).get(); 
  }

  tf::Executor executor;
  tf::Taskflow taskflow;
  std::vector<std::vector<tf::Task>> tasks;

};

void traverse_level_graph_taskflow(LevelGraph& graph, unsigned num_threads){
  TF tf(graph, num_threads);
  tf.run();
}

std::chrono::microseconds measure_time_taskflow(LevelGraph& graph, unsigned num_threads){
  auto beg = std::chrono::high_resolution_clock::now();
  traverse_level_graph_taskflow(graph, num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

