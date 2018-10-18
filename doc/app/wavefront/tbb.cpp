#include <algorithm> // for std::max
#include <cstdio>
#include <chrono>
#include <iostream>
#include <thread>
#include <random>
#include <cmath>
#include "tbb/task_scheduler_init.h"
#include "tbb/flow_graph.h" 

using namespace tbb;
using namespace tbb::flow;

int M=40000, N=40000;
int B = 160;
int MB = (M/B) + (M%B>0);
int NB = (N/B) + (N%B>0);

double **value;


inline double calc(double v0, double v1) {
  if(v0 == v1)
    return std::pow(v0/v1, 4.0f);
  else
    return std::max(v0,v1);
}

continue_node<continue_msg> ***node;

void BuildGraph(graph &g) {
  value[M-1][N-1] = 0;
  for( int i=MB; --i>=0; )
    for( int j=NB; --j>=0; ) {
      node[i][j] =
        new continue_node<continue_msg>( g,
          [=]( const continue_msg& ) {
            int start_i = i*B;
            int end_i = (i*B+B > M) ? M : i*B+B;
            int start_j = j*B;
            int end_j = (j*B+B > N) ? N : j*B+B;
            for ( int ii = start_i; ii < end_i; ++ii ) {
              for ( int jj = start_j; jj < end_j; ++jj ) {
                double v0 = (ii == 0) ? 0 : value[ii-1][jj];
                double v1 = (jj == 0) ? 0 : value[ii][jj-1];
                value[ii][jj] = (ii==0 && jj==0) ? 1 : calc(v0,v1);
               }
            }
      });
      if(i+1 < MB) make_edge(*node[i][j], *node[i+1][j]);
      if(j+1 < NB) make_edge(*node[i][j], *node[i][j+1]);
    }
}

double EvaluateGraph(graph &g) {
  node[0][0]->try_put(continue_msg());
  g.wait_for_all();
  return value[M-1][N-1];
}

void CleanupGraph() {
  for(int i=0; i<MB; ++i)
    for(int j=0; j<NB; ++j)
     delete node[i][j];
}

void init_data(){
  value = new double *[M];
  for ( int i = 0; i < M; ++i ) value[i] = new double [N];
  for(int i=0; i<M; ++i){
    for(int j=0; j<N ; ++j){
      value[i][j] = i*N + j;
    }   
  }
}

// The code comes from: 
// https://software.intel.com/en-us/blogs/2011/09/09/implementing-a-wave-front-computation-using-the-intel-threading-building-blocks-flow-graph
int main(int argc, char *argv[]) {
  init_data();

  double result;
  tbb::task_scheduler_init init(std::thread::hardware_concurrency());

  auto beg = std::chrono::high_resolution_clock::now();
  {
    node = new continue_node<continue_msg> **[MB];
    for ( int i = 0; i < MB; ++i ) node[i] = new continue_node<continue_msg> *[NB];

    graph g;
    BuildGraph(g);
    result = EvaluateGraph(g);
    CleanupGraph();
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "TBB: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count() << " ms, Result =" 
            << result << "\n";


  for ( int i = 0; i < MB; ++i ) delete [] node[i];
  delete [] node;

  for ( int i = 0; i < M; ++i ) delete [] value[i];
  delete [] value;
  return 0;
}

