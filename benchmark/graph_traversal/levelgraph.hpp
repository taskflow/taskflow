#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <tuple>
#include <chrono>
#include <cmath>
#include <memory>

#include <taskflow/taskflow.hpp>
#include <tbb/task_scheduler_init.h>
#include <tbb/flow_graph.h>

class Node{
  
  public:

    Node(size_t level, int index, std::vector<int>& next_level_nodes){
      _level = level;
      _idx = index;
      _out_edges = std::move(next_level_nodes);
    }
   
    void mark(){
      _visited = true;
    }

    void unmark(){
      _visited = false;
    }
 
    bool check_status() { 
      return _visited; 
    }

    void print_node(){
  
      std::cout << "Node: " << _idx << " out_edges: ";
      for(const auto& out_edge: _out_edges){
        std::cout << out_edge << "\t";
      }

      std::cout << "\n" << "in_edges";  
  
      for(const auto& in_edge: _in_edges){
        std::cout << "(" << in_edge.first << "," << in_edge.second << ")\t";
      }

      std::cout << "Status:" << _visited << std::endl;

      std::cout << std::endl;

    }

    int index() const { return _idx; }

    int* edge_ptr(int edge_idx) { return &_out_edges[edge_idx]; }

    std::vector<std::pair<int, int>> _in_edges;
    std::vector<int> _out_edges;

  private:

    size_t _level;
    int _idx;

    bool _visited {false};
};

class LevelGraph {

  public:

    LevelGraph(size_t length, size_t level){

      _level_num = level;
      _length_num = length;

      std::mt19937 g(0);  // fixed the seed

      for(size_t l=0; l<level; ++l){
        
        std::vector<Node> cur_nodes;
        std::vector<int> next_level_nodes;

        for(size_t i=0; i<length; i++){
          next_level_nodes.push_back(i);
        }
     
        //shuffle nodes in the next level
        std::shuffle(next_level_nodes.begin(), next_level_nodes.end(), g);

        size_t edge_num = 1;
        size_t start = 0, end = 0;
        bool re_shuffle = false;  
    
        for(size_t i=0; i<length; i++){
          edge_num = 1 + (std::rand() % _edge_max);
          
  
          if(start + edge_num >= length){
            end = length;
            re_shuffle = true;
          }
          else{
            end = start + edge_num;
          }
          
          //std::cout << "Level\t" << l << "\tidx\t" << i << "\tedge_num\t" << edge_num << "\tstart\t" << start << "\tend\t" << end << std::endl;
          
          std::vector<int> edges(next_level_nodes.begin()+start, next_level_nodes.begin()+end);          
          
          cur_nodes.emplace_back(l, i, edges);

          if(re_shuffle){
            std::shuffle(next_level_nodes.begin(), next_level_nodes.end(), g);
            start = 0;
            re_shuffle = false;
          }
          else{
            start = end;
          }

        } 

        _graph.push_back(std::move(cur_nodes));

      }

      for(size_t l=0; level > 0 && l<level-1; ++l){
        for(size_t i=0; i<length; ++i){
          for(size_t j=0; j<_graph[l][i]._out_edges.size(); ++j){
            int src_idx = _graph[l][i].index();
            int dest_idx = _graph[l][i]._out_edges[j];
            _graph[l+1][dest_idx]._in_edges.push_back(std::make_pair(src_idx, j));
          }
        } 
      }
      

    }

    void print_graph(){
  
      for(size_t l=0; l<_graph.size(); l++){
        std::cout << "-----------Level " << l <<"----------" << std::endl;
        for(size_t i=0; i<_graph[l].size(); i++){
          _graph[l][i].print_node();
        }
      }

    }

    bool validate_result(){
    
      for(size_t l=0; l<_level_num; l++){
        for(size_t i=0; i<_length_num; i++){
          if(_graph[l][i].check_status() == false){
            std::cout << "Level:\t" << l << "\tidx\t" << i << "\tnot visited" << std::endl;
            return false;
          }
        }
      }

      return true;
    }

    void clear_graph(){
      for(size_t l=0; l<_level_num; l++){
        for(size_t i=0; i<_length_num; i++){
          _graph[l][i].unmark();
        }
      }
    }


    Node& node_at(size_t level, size_t index){ return _graph[level][index]; }

    const size_t level(){ return _level_num; }
    const size_t length() { return _length_num;  }

    size_t graph_size() const {
      size_t size = 0;
      for(const auto& nodes : _graph) {
        for(const auto& node : nodes) {
          size++;
          size+=node._out_edges.size();
        }
      }
      return size;
    }

  private:
    
    const size_t _edge_max = 4;
    size_t _level_num;
    size_t _length_num;

    std::vector<std::vector<Node>> _graph;

};

std::chrono::microseconds measure_time_taskflow(LevelGraph&, unsigned);
std::chrono::microseconds measure_time_omp(LevelGraph&, unsigned);
std::chrono::microseconds measure_time_tbb(LevelGraph&, unsigned);






