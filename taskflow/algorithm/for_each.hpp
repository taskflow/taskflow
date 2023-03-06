#pragma once

#include "partitioner.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// default parallel for
// ----------------------------------------------------------------------------

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
      detail::loop_guided(N, W, chunk_size, 0, next, 
        [&](size_t prev_e, size_t curr_b, size_t curr_e) {
          std::advance(beg, curr_b - prev_e);
          for(size_t x = curr_b; x<curr_e; x++) {
            c(*beg++);
          }
        }
      ); 
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
        sf._named_silent_async(
          sf._worker, "loop-"s + std::to_string(w), loop
        );
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
      detail::loop_guided(N, W, chunk_size, 0, next, 
        [&](size_t, size_t curr_b, size_t curr_e) {
          auto idx = static_cast<B_t>(curr_b) * inc + beg;
          for(size_t x=curr_b; x<curr_e; x++, idx += inc) {
            c(idx);
          }
        }
      ); 
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



