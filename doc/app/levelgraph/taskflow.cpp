#include <iostream>
#include <chrono>

#include "taskflow.hpp"
#include "levelgraph.hpp"

void traverse_regular_graph_taskflow(RegularGraph& graph){
  
  tf::Taskflow tf(4);

  for(size_t i=0; i<graph.length(); i++){
    Node& n = graph.node_at(graph.level()-1, i); 
    n._task = tf.silent_emplace([&](){ n.mark(); });
  }
  
  for(int l=graph.level()-2; l>=0 ; l--){
    for(size_t i=0; i<graph.length(); i++){
      Node& n = graph.node_at(l, i);
      n._task = tf.silent_emplace([&](){ n.mark();});
      for(size_t k=0; k<n._out_edges.size(); k++){
        n._task.precede(graph.node_at(l+1, n._out_edges[k])._task);
      } 
    }
  }
  tf.wait_for_all();
}

auto measure_time_taskflow(RegularGraph& graph){
  auto beg = std::chrono::high_resolution_clock::now();
  traverse_regular_graph_taskflow(graph);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count();
}

int main(int argc, char* argv[]){

  for(int i=1; i<=200; i++){

    RegularGraph graph(i, i);
    auto taskflow = measure_time_taskflow(graph);
    std::cout << "Level graph:\t" << i << "\tby\t" << i << std::endl;
    std::cout << "Elasped time taskflow:\t" << taskflow << std::endl;
    std::cout << "Graph is fully traversed:\t" << graph.validate_result() << std::endl;  
    graph.clear_graph();
    std::cout << std::endl;
  }

}
