#include <iostream>
#include <chrono>

#include "levelgraph.hpp"

std::chrono::microseconds measure_time_seq(LevelGraph& graph){
  auto beg = std::chrono::high_resolution_clock::now();
  for(size_t l=0; l<graph.level(); l++){ 
    for(size_t i=0; i<graph.length(); i++){
      Node& n = graph.node_at(l, i);
      n.mark();
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
