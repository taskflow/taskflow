#include <iostream>
#include <chrono>

#include "levelgraph.hpp"

auto measure_time_seq(RegularGraph& graph){
  auto beg = std::chrono::high_resolution_clock::now();
  for(size_t l=0; l<graph.level(); l++){ 
    for(int i=0; i<graph.length(); i++){
      Node& n = graph.node_at(l, i);
      n.mark();
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count();
}

int main(int argc, char* argv[]){

  for(int i=1; i<=200; i++){

    RegularGraph graph(i, i);
    auto seq = measure_time_seq(graph);
    std::cout << "Level graph:\t" << i << "\tby\t" << i << std::endl;
    std::cout << "Elasped time sequential: \t" << seq << std::endl;
    std::cout << "Graph is fully traversed:\t" << graph.validate_result() << std::endl;  
    graph.clear_graph();
    std::cout << std::endl;

  }

}
