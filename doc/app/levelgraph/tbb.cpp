#include <iostream>
#include <chrono>
#include <omp.h>

#include "levelgraph.hpp"
#include "tbb/task_scheduler_init.h"
#include "tbb/flow_graph.h"


void traverse_regular_graph_tbb(RegularGraph& graph){

  std::cout << "traverse\n";

  using namespace tbb;
  using namespace tbb::flow;
  tbb::task_scheduler_init init(std::thread::hardware_concurrency());

  tbb::flow::graph G;

  for(size_t i=0; i<graph.length(); i++){
    Node& n = graph.node_at(graph.level()-1, i);
    //n._task = tf.silent_emplace([&](){ n.mark(); });
    n.tbb_node = std::make_unique<continue_node<continue_msg>>(G, [&](const continue_msg&){ n.mark(); });
  }

  for(int l=graph.level()-2; l>=0 ; l--){
    for(size_t i=0; i<graph.length(); i++){
      Node& n = graph.node_at(l, i);
      n.tbb_node = std::make_unique<continue_node<continue_msg>>(G, [&](const continue_msg&){ n.mark(); });
      //n._task = tf.silent_emplace([&](){ n.mark();});
      for(size_t k=0; k<n._out_edges.size(); k++){
        //n._task.precede(graph.node_at(l+1, n._out_edges[k])._task);
        make_edge(*n.tbb_node, *(graph.node_at(l+1, n._out_edges[k]).tbb_node));
      }
    }
  }

  auto source = std::make_unique<continue_node<continue_msg>>(G, [](const continue_msg&){});
  for(int l=0; l>=0 ; l--){
    for(size_t i=0; i<graph.length(); i++){
      Node& n = graph.node_at(l, i);
      make_edge(*source, *(n.tbb_node));
    }
  }

  source->try_put(continue_msg());
  G.wait_for_all();
  graph.reset_tbb_node();
}

auto measure_time_tbb(RegularGraph& graph){
  std::cout << "measure\n";
  auto beg = std::chrono::high_resolution_clock::now();
  traverse_regular_graph_tbb(graph);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count();
}

int main(int argc, char* argv[]){

  for(int i=1; i<=200; i++){
    std::cout << "i=\t" << i << std::endl;
    RegularGraph graph(i, i);
    auto tbb = measure_time_tbb(graph);
    std::cout << "Level graph:\t" << i << "\tby\t" << i << std::endl;
    std::cout << "Elasped time tbb:\t" << tbb << std::endl;
    std::cout << "Graph is fully traversed:\t" << graph.validate_result() << std::endl;
    graph.clear_graph();
    std::cout << std::endl;

  }

}
