#include <iostream>
#include <chrono>
#include <omp.h>

#include "levelgraph.hpp"

void traverse_regular_graph_omp(LevelGraph& graph){

  omp_set_num_threads(std::thread::hardware_concurrency());
  
  #pragma omp parallel
  {
    #pragma omp single
    {
      for(size_t l=0; l<graph.level(); l++){ 
        for(int i=0; i<graph.length(); i++){
          Node& n = graph.node_at(l, i);
          size_t out_edge_num = n._out_edges.size();
          size_t in_edge_num = n._in_edges.size();

          switch(in_edge_num){
            
            case(0):{
              
              switch(out_edge_num){

                case(1):{
                  int* out0 = n.edge_ptr(0);
                  #pragma omp task depend(out: out0[0]) shared(n)
                  { n.mark(); }
                  break;
                }

                case(2):{
                  int* out0 = n.edge_ptr(0);
                  int* out1 = n.edge_ptr(1);
                  #pragma omp task depend(out: out0[0], out1[0]) shared(n)
                  { n.mark(); }
                  break;
                }

                case(3):{
                  int* out0 = n.edge_ptr(0);
                  int* out1 = n.edge_ptr(1);
                  int* out2 = n.edge_ptr(2);
                  #pragma omp task depend(out: out0[0], out1[0], out2[0]) shared(n)
                  { n.mark(); }
                  break;
                }
  
                case(4):{
                  int* out0 = n.edge_ptr(0);
                  int* out1 = n.edge_ptr(1);
                  int* out2 = n.edge_ptr(2);
                  int* out3 = n.edge_ptr(3);
                  #pragma omp task depend(out: out0[0], out1[0], out2[0], out3[0]) shared(n)
                  { n.mark(); }
                  break;
                }
              
              }
              break;
            }


            case(1):{
              int* in0 = graph.node_at(l-1, n._in_edges[0].first).edge_ptr(n._in_edges[0].second);

              switch(out_edge_num){

                case(1):{
                  int* out0 = n.edge_ptr(0);
                  #pragma omp task depend(in: in0[0]) depend(out: out0[0]) shared(n)
                  { n.mark(); }
                  break;
                }

                case(2):{
                  int* out0 = n.edge_ptr(0);
                  int* out1 = n.edge_ptr(1);
                  #pragma omp task depend(in: in0[0]) depend(out: out0[0], out1[0]) shared(n)
                  { n.mark(); }
                  break;
                }

                case(3):{

                  int* out0 = n.edge_ptr(0);
                  int* out1 = n.edge_ptr(1);
                  int* out2 = n.edge_ptr(2);
                  #pragma omp task depend(in: in0[0]) depend(out: out0[0], out1[0], out2[0]) shared(n)
                  { n.mark(); }
                  break;
                }
  
                case(4):{

                  int* out0 = n.edge_ptr(0);
                  int* out1 = n.edge_ptr(1);
                  int* out2 = n.edge_ptr(2);
                  int* out3 = n.edge_ptr(3);
                  #pragma omp task depend(in: in0[0]) depend(out: out0[0], out1[0], out2[0], out3[0]) shared(n)
                  { n.mark(); }
                  break;
                }
              }
                break;
            }


            case(2):{

              int* in0 = graph.node_at(l-1, n._in_edges[0].first).edge_ptr(n._in_edges[0].second);
              int* in1 = graph.node_at(l-1, n._in_edges[1].first).edge_ptr(n._in_edges[1].second);

              switch(out_edge_num){

                case(1):{
                  int* out0 = n.edge_ptr(0);
                  #pragma omp task depend(in: in0[0], in1[0]) depend(out: out0[0]) shared(n)
                  { n.mark(); }
                  break;
                }

                case(2):{
                  int* out0 = n.edge_ptr(0);
                  int* out1 = n.edge_ptr(1);
                  #pragma omp task depend(in: in0[0], in1[0]) depend(out: out0[0], out1[0]) shared(n)
                  { n.mark(); }
                  break;
                }

                case(3):{

                  int* out0 = n.edge_ptr(0);
                  int* out1 = n.edge_ptr(1);
                  int* out2 = n.edge_ptr(2);
                  #pragma omp task depend(in: in0[0], in1[0]) depend(out: out0[0], out1[0], out2[0]) shared(n)
                  { n.mark(); }
                  break;

                }
  
                case(4):{

                  int* out0 = n.edge_ptr(0);
                  int* out1 = n.edge_ptr(1);
                  int* out2 = n.edge_ptr(2);
                  int* out3 = n.edge_ptr(3);
                  #pragma omp task depend(in: in0[0], in1[0]) depend(out: out0[0], out1[0], out2[0], out3[0]) shared(n)
                  { n.mark(); }
                  break;
                }
              }
            break;
            }


            case(3):{

              int* in0 = graph.node_at(l-1, n._in_edges[0].first).edge_ptr(n._in_edges[0].second);
              int* in1 = graph.node_at(l-1, n._in_edges[1].first).edge_ptr(n._in_edges[1].second);
              int* in2 = graph.node_at(l-1, n._in_edges[2].first).edge_ptr(n._in_edges[2].second);

              switch(out_edge_num){

                case(1):{
                  int* out0 = n.edge_ptr(0);
                  #pragma omp task depend(in: in0[0], in1[0], in2[0]) depend(out: out0[0]) shared(n)
                  { n.mark(); }
                  break;
                }

                case(2):{
                  int* out0 = n.edge_ptr(0);
                  int* out1 = n.edge_ptr(1);
                  #pragma omp task depend(in: in0[0], in1[0], in2[0]) depend(out: out0[0], out1[0]) shared(n)
                  { n.mark(); }
                  break;

                }

                case(3):{

                  int* out0 = n.edge_ptr(0);
                  int* out1 = n.edge_ptr(1);
                  int* out2 = n.edge_ptr(2);
                  #pragma omp task depend(in: in0[0], in1[0], in2[0]) depend(out: out0[0], out1[0], out2[0]) shared(n)
                  { n.mark(); }
                  break;
                }
  
                case(4):{

                  int* out0 = n.edge_ptr(0);
                  int* out1 = n.edge_ptr(1);
                  int* out2 = n.edge_ptr(2);
                  int* out3 = n.edge_ptr(3);
                  #pragma omp task depend(in: in0[0], in1[0], in2[0]) depend(out: out0[0], out1[0], out2[0], out3[0]) shared(n)
                  { n.mark(); }
                  break;
                }
              }
            break;

            }


            case(4):{

              int* in0 = graph.node_at(l-1, n._in_edges[0].first).edge_ptr(n._in_edges[0].second);
              int* in1 = graph.node_at(l-1, n._in_edges[1].first).edge_ptr(n._in_edges[1].second);
              int* in2 = graph.node_at(l-1, n._in_edges[2].first).edge_ptr(n._in_edges[2].second);
              int* in3 = graph.node_at(l-1, n._in_edges[3].first).edge_ptr(n._in_edges[3].second);

              switch(out_edge_num){

                case(1):{
                  int* out0 = n.edge_ptr(0);
                  #pragma omp task depend(in: in0[0], in1[0], in2[0], in3[0]) depend(out: out0[0]) shared(n)
                  { n.mark(); }
                  break;
                }

                case(2):{
                  int* out0 = n.edge_ptr(0);
                  int* out1 = n.edge_ptr(1);
                  #pragma omp task depend(in: in0[0], in1[0], in2[0], in3[0]) depend(out: out0[0], out1[0]) shared(n)
                  { n.mark(); }
                  break;

                }

                case(3):{

                  int* out0 = n.edge_ptr(0);
                  int* out1 = n.edge_ptr(1);
                  int* out2 = n.edge_ptr(2);
                  #pragma omp task depend(in: in0[0], in1[0], in2[0], in3[0]) depend(out: out0[0], out1[0], out2[0]) shared(n)
                  { n.mark(); }
                  break;

                }
  
                case(4):{

                  int* out0 = n.edge_ptr(0);
                  int* out1 = n.edge_ptr(1);
                  int* out2 = n.edge_ptr(2);
                  int* out3 = n.edge_ptr(3);
                  #pragma omp task depend(in: in0[0], in1[0], in2[0], in3[0]) depend(out: out0[0], out1[0], out2[0], out3[0]) shared(n)
                  { n.mark(); }
                  break;
                }
              }
            break;
            }
          }
        }
      }
    }
  }  
}

std::chrono::microseconds measure_time_omp(LevelGraph& graph){
  auto beg = std::chrono::high_resolution_clock::now();
  traverse_regular_graph_omp(graph);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

//int main(int argc, char* argv[]){
//
//  for(int i=1; i<=200; i++){
//    LevelGraph graph(i, i);
//    auto omp = measure_time_OMP(graph);
//    std::cout << "Level graph:\t" << i << "\tby\t" << i << std::endl;
//    std::cout << "Elasped time OMP:\t" << omp << std::endl;
//    std::cout << "Graph is fully traversed:\t" << graph.validate_result() << std::endl;  
//    graph.clear_graph();
//    std::cout << std::endl;
//  }
//
//}

