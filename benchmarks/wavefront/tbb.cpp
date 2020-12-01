#include "matrix.hpp"

#include <tbb/global_control.h>
#include <tbb/flow_graph.h>

// the wavefront computation
void wavefront_tbb(unsigned num_threads) {

  using namespace tbb;
  using namespace tbb::flow;
  
  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_threads
  );
    
  continue_node<continue_msg> ***node = new continue_node<continue_msg> **[MB];

  for ( int i = 0; i < MB; ++i ) {
    node[i] = new continue_node<continue_msg> *[NB];
  };

  graph g;

  matrix[M-1][N-1] = 0;

  for( int i=MB; --i>=0; ) {
    for( int j=NB; --j>=0; ) {
      node[i][j] = new continue_node<continue_msg>( g,
        [=]( const continue_msg& ) {
          block_computation(i, j);
        }
      );
      if(i+1 < MB) make_edge(*node[i][j], *node[i+1][j]);
      if(j+1 < NB) make_edge(*node[i][j], *node[i][j+1]);
    }
  }
  
  node[0][0]->try_put(continue_msg());
  g.wait_for_all();
  
  for(int i=0; i<MB; ++i) {
    for(int j=0; j<NB; ++j) {
     delete node[i][j];
    }
    delete [] node[i];
  }

  delete [] node;
}

std::chrono::microseconds measure_time_tbb(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  wavefront_tbb(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


