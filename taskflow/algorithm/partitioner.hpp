// reference:
// - gomp: https://github.com/gcc-mirror/gcc/blob/master/libgomp/iter.c
// - komp: https://github.com/llvm-mirror/openmp/blob/master/runtime/src/kmp_dispatch.cpp

#pragma once

#include "../core/executor.hpp"

namespace tf::detail {

/**
@private
@brief guided partitioning algorithm

[prev_b prev_e) [curr_b, curr_e) ...
                (current partition)

all values are indices
*/
template <typename F>
void loop_guided(
  size_t N,
  size_t W,
  size_t chunk_size,
  size_t prev_e,
  std::atomic<size_t>& next, 
  F&& func
) {
      
  size_t p1 = 2 * W * (chunk_size + 1);
  double p2 = 0.5 / static_cast<double>(W);
  size_t curr_b = next.load(std::memory_order_relaxed);

  while(curr_b < N) {

    size_t r = N - curr_b;

    // fine-grained
    if(r < p1) {
      while(1) {
        curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);
        if(curr_b >= N) {
          return;
        }
        size_t curr_e = (chunk_size <= (N - curr_b)) ? curr_b + chunk_size : N;
        func(prev_e, curr_b, curr_e);
        prev_e = curr_e;
      }
      break;
    }
    // coarse-grained
    else {
      size_t q = static_cast<size_t>(p2 * r);
      if(q < chunk_size) {
        q = chunk_size;
      }
      size_t curr_e = (q <= r) ? curr_b + q : N;
      if(next.compare_exchange_strong(curr_b, curr_e, std::memory_order_relaxed,
                                              std::memory_order_relaxed)) {
        func(prev_e, curr_b, curr_e);
        prev_e = curr_e;
        curr_b = next.load(std::memory_order_relaxed);
      }
    }
  }
}

}  // end of namespace tf::detail ---------------------------------------------



