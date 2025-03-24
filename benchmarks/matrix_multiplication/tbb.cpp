#include "matrix_multiplication.hpp"
#include <tbb/global_control.h>
#include <tbb/flow_graph.h>
#include <tbb/parallel_for.h>

// matrix_multiplication_tbb
void matrix_multiplication_tbb(unsigned num_threads) {

  using namespace tbb;
  using namespace tbb::flow;

  graph g;

  tbb::global_control control(
    tbb::global_control::max_allowed_parallelism, num_threads
  );
  
  // create the source node
  continue_node<continue_msg> src(g, [&](const continue_msg&) {
  });

  // Create flow graph nodes
  continue_node<continue_msg> node1(g, [&](const continue_msg&) {
    tbb::parallel_for(0, N, 1, [=](int i){
      for(int j=0; j<N; ++j) {
        a[i][j] = i + j;
      }
    });
  });

  continue_node<continue_msg> node2(g, [&](const continue_msg&) { 
    tbb::parallel_for(0, N, 1, [=](int i){
      for(int j=0; j<N; ++j) {
        b[i][j] = i * j;
      }
    });
  });

  continue_node<continue_msg> node3(g, [&](const continue_msg&) {
    tbb::parallel_for(0, N, 1, [=](int i){
      for(int j=0; j<N; ++j) {
        c[i][j] = 0;
      }
    });
  });

  // Final node to sync all previous tasks
  continue_node<continue_msg> final_node(g, [&](const continue_msg&) {
    tbb::parallel_for(0, N, 1, [=](int i){
      for(int j=0; j<N; ++j) {
        for(int k=0; k<N; k++) {
          c[i][j] += a[i][k] * b[k][j];
        }
      }
    });
  });

  // Connect dependencies (final_node depends on node1, node2, and node3)
  make_edge(src, node1);
  make_edge(src, node2);
  make_edge(src, node3);
  make_edge(node1, final_node);
  make_edge(node2, final_node);
  make_edge(node3, final_node);

  // Trigger the first three tasks in parallel
  src.try_put(continue_msg());

  // Wait for everything to finish
  g.wait_for_all();
}

std::chrono::microseconds measure_time_tbb(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  matrix_multiplication_tbb(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
