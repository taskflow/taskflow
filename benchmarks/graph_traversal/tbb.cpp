#include <iostream>
#include <chrono>
#include <omp.h>

#include "levelgraph.hpp"
#include <tbb/global_control.h>
#include <tbb/flow_graph.h>
  
using namespace tbb;
using namespace tbb::flow;

struct TBB {
  
  TBB(LevelGraph& graph, unsigned num_threads) {

    tbb::global_control c(
      tbb::global_control::max_allowed_parallelism, num_threads
    );

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

void traverse_regular_graph_tbb(LevelGraph& graph, unsigned num_threads){
  TBB tbb(graph, num_threads);
  tbb.run();
}

std::chrono::microseconds measure_time_tbb(LevelGraph& graph, unsigned num_threads){
  auto beg = std::chrono::high_resolution_clock::now();
  traverse_regular_graph_tbb(graph, num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

//int main(int argc, char* argv[]){
//
//  for(int i=1; i<=200; i++){
//    LevelGraph graph(i, i);
//    auto tbb = measure_time_tbb(graph);
//    std::cout << "Level graph:\t" << i << "\tby\t" << i << std::endl;
//    std::cout << "Elasped time tbb:\t" << tbb << std::endl;
//    std::cout << "Graph is fully traversed:\t" << graph.validate_result() << std::endl;
//    graph.clear_graph();
//    std::cout << std::endl;
//  }
//
//}
