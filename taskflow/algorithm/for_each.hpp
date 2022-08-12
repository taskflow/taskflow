// reference:
// - gomp: https://github.com/gcc-mirror/gcc/blob/master/libgomp/iter.c
// - komp: https://github.com/llvm-mirror/openmp/blob/master/runtime/src/kmp_dispatch.cpp

#pragma once

#include "../core/executor.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// default parallel for
// ----------------------------------------------------------------------------

// Function: for_each
template <typename B, typename E, typename C>
Task FlowBuilder::for_each(B beg, E end, C c) {

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using namespace std::string_literals;

  Task task = emplace([b=beg, e=end, c] (Subflow& sf) mutable {

    // fetch the stateful values
    B_t beg = b;
    E_t end = e;

    if(beg == end) {
      return;
    }

    size_t chunk_size = 1;
    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= chunk_size) {
      std::for_each(beg, end, c);
      return;
    }

    if(N < W) {
      W = N;
    }

    std::atomic<size_t> next(0);
      
    auto loop = [=, &next] () mutable {

      size_t z = 0;
      size_t p1 = 2 * W * (chunk_size + 1);
      double p2 = 0.5 / static_cast<double>(W);
      size_t s0 = next.load(std::memory_order_relaxed);

      while(s0 < N) {

        size_t r = N - s0;

        // fine-grained
        if(r < p1) {
          while(1) {
            s0 = next.fetch_add(chunk_size, std::memory_order_relaxed);
            if(s0 >= N) {
              return;
            }
            size_t e0 = (chunk_size <= (N - s0)) ? s0 + chunk_size : N;
            std::advance(beg, s0-z);
            for(size_t x=s0; x<e0; x++) {
              c(*beg++);
            }
            z = e0;
          }
          break;
        }
        // coarse-grained
        else {
          size_t q = static_cast<size_t>(p2 * r);
          if(q < chunk_size) {
            q = chunk_size;
          }
          size_t e0 = (q <= r) ? s0 + q : N;
          if(next.compare_exchange_strong(s0, e0, std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            std::advance(beg, s0-z);
            for(size_t x = s0; x< e0; x++) {
              c(*beg++);
            }
            z = e0;
            s0 = next.load(std::memory_order_relaxed);
          }
        }
      }
    };

    for(size_t w=0; w<W; w++) {
      auto r = N - next.load(std::memory_order_relaxed);
      // no more loop work to do - finished by previous async tasks
      if(!r) {
        break;
      }
      // tail optimization
      if(r <= chunk_size || w == W-1) {
        loop(); 
        break;
      }
      else {
        sf._named_silent_async(sf._worker, "loop-"s + std::to_string(w), loop);
      }
    }

    sf.join();
  });

  return task;
}

// Function: for_each_index
template <typename B, typename E, typename S, typename C>
Task FlowBuilder::for_each_index(B beg, E end, S inc, C c){

  using namespace std::string_literals;

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using S_t = std::decay_t<unwrap_ref_decay_t<S>>;

  Task task = emplace([b=beg, e=end, a=inc, c] (Subflow& sf) mutable {

    // fetch the iterator values
    B_t beg = b;
    E_t end = e;
    S_t inc = a;

    if(is_range_invalid(beg, end, inc)) {
      TF_THROW("invalid range [", beg, ", ", end, ") with step size ", inc);
    }

    size_t chunk_size = 1;
    size_t W = sf._executor.num_workers();
    size_t N = distance(beg, end, inc);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= chunk_size) {
      for(size_t x=0; x<N; x++, beg+=inc) {
        c(beg);
      }
      return;
    }

    if(N < W) {
      W = N;
    }

    std::atomic<size_t> next(0);
      
    auto loop = [=, &next] () mutable {

      size_t p1 = 2 * W * (chunk_size + 1);
      double p2 = 0.5 / static_cast<double>(W);
      size_t s0 = next.load(std::memory_order_relaxed);

      while(s0 < N) {

        size_t r = N - s0;

        // fine-grained
        if(r < p1) {
          while(1) {
            s0 = next.fetch_add(chunk_size, std::memory_order_relaxed);
            if(s0 >= N) {
              return;
            }
            size_t e0 = (chunk_size <= (N - s0)) ? s0 + chunk_size : N;
            auto s = static_cast<B_t>(s0) * inc + beg;
            for(size_t x=s0; x<e0; x++, s+=inc) {
              c(s);
            }
          }
          break;
        }
        // coarse-grained
        else {
          size_t q = static_cast<size_t>(p2 * r);
          if(q < chunk_size) {
            q = chunk_size;
          }
          size_t e0 = (q <= r) ? s0 + q : N;
          if(next.compare_exchange_strong(s0, e0, std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            auto s = static_cast<B_t>(s0) * inc + beg;
            for(size_t x=s0; x<e0; x++, s+= inc) {
              c(s);
            }
            s0 = next.load(std::memory_order_relaxed);
          }
        }
      }
    };

    for(size_t w=0; w<W; w++) {
      auto r = N - next.load(std::memory_order_relaxed);
      // no more loop work to do - finished by previous async tasks
      if(!r) {
        break;
      }
      // tail optimization
      if(r <= chunk_size || w == W-1) {
        loop(); 
        break;
      }
      else {
        sf._named_silent_async(sf._worker, "loop-"s + std::to_string(w), loop);
      }
    }
      
    sf.join();
  });

  return task;
}

}  // end of namespace tf -----------------------------------------------------



