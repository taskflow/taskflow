#include <iostream>
#include <chrono>
#include <omp.h>

#include "levelgraph.hpp"
#include <tbb/task_scheduler_init.h>
#include <tbb/flow_graph.h>
  
using namespace tbb;
using namespace tbb::flow;

struct TBB {
  
  TBB(LevelGraph& graph, unsigned num_threads) {

    tbb::task_scheduler_init init(num_threads);

    tasks.resize(graph.level()); 
    for(size_t i=0; i<tasks.size(); ++i) {
      tasks[i].resize(graph.length());
    }

    for(size_t i=0; i<graph.length(); i++){
      Node& n = graph.node_at(graph.level()-1, i);
      tasks[graph.level()-1][i] = std::make_unique<continue_node<continue_msg>>(G, 
        [&](const continue_msg&){ n.mark(); }
      );
    }

    for(int l=graph.level()-2; l>=0 ; l--){
      for(size_t i=0; i<graph.length(); i++){
        Node& n = graph.node_at(l, i);
        tasks[l][i] = std::make_unique<continue_node<continue_msg>>(G, 
          [&](const continue_msg&){ n.mark(); }
        );
        for(size_t k=0; k<n._out_edges.size(); k++){
          make_edge(*tasks[l][i], *tasks[l+1][n._out_edges[k]]);
        }
      }
    }

    source = std::make_unique<continue_node<continue_msg>>(G, 
      [](const continue_msg&){}
    );

    for(int l=0; l>=0 ; l--) {
      for(size_t i=0; i<graph.length(); i++){
        make_edge(*source, *tasks[l][i]);
      }
    }
  }

  void run() {
    source->try_put(continue_msg());
    G.wait_for_all();
  }
  
  tbb::flow::graph G;
  std::vector<std::vector<std::unique_ptr<continue_node<continue_msg>>>> tasks;
  std::unique_ptr<continue_node<continue_msg>> source;
};

void traverse_regular_graph_tbb(LevelGraph& graph, unsigned num_threads, int repeat){
  TBB tbb(graph, num_threads);

  for(auto i=0; i<repeat; i++) {
    tbb.run();
    graph.clear_graph();
  }
}

std::chrono::microseconds measure_time_tbb(LevelGraph& graph, unsigned num_threads, int repeat){
  auto beg = std::chrono::high_resolution_clock::now();
  traverse_regular_graph_tbb(graph, num_threads, repeat);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

