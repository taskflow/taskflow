#pragma once

#include "../core/executor.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// default transform
// ----------------------------------------------------------------------------

// Function: transform
template <typename B, typename E, typename O, typename C>
Task FlowBuilder::transform(B first1, E last1, O d_first, C c) {

  using namespace std::string_literals;

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using O_t = std::decay_t<unwrap_ref_decay_t<O>>;

  Task task = emplace(
  [first1, last1, d_first, c] (Subflow& sf) mutable {

    // fetch the stateful values
    B_t beg   = first1;
    E_t end   = last1;
    O_t d_beg = d_first;

    if(beg == end) {
      return;
    }

    size_t chunk_size = 1;
    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= chunk_size) {
      std::transform(beg, end, d_beg, c);
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
            std::advance(d_beg, s0-z);
            for(size_t x=s0; x<e0; x++) {
              *d_beg++ = c(*beg++);
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
            std::advance(d_beg, s0-z);
            for(size_t x = s0; x< e0; x++) {
              *d_beg++ = c(*beg++);
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

// Function: transform
template <typename B1, typename E1, typename B2, typename O, typename C>
Task FlowBuilder::transform(B1 first1, E1 last1, B2 first2, O d_first, C c) {

  using namespace std::string_literals;

  using B1_t = std::decay_t<unwrap_ref_decay_t<B1>>;
  using E1_t = std::decay_t<unwrap_ref_decay_t<E1>>;
  using B2_t = std::decay_t<unwrap_ref_decay_t<B2>>;
  using O_t = std::decay_t<unwrap_ref_decay_t<O>>;

  Task task = emplace(
  [first1, last1, first2, d_first, c] (Subflow& sf) mutable {

    // fetch the stateful values
    B1_t beg1 = first1;
    E1_t end1 = last1;
    B2_t beg2 = first2;
    O_t d_beg = d_first;

    if(beg1 == end1) {
      return;
    }

    size_t chunk_size = 1;
    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg1, end1);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= chunk_size) {
      std::transform(beg1, end1, beg2, d_beg, c);
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
            std::advance(beg1, s0-z);
            std::advance(beg2, s0-z);
            std::advance(d_beg, s0-z);
            for(size_t x=s0; x<e0; x++) {
              *d_beg++ = c(*beg1++, *beg2++);
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
            std::advance(beg1, s0-z);
            std::advance(beg2, s0-z);
            std::advance(d_beg, s0-z);
            for(size_t x = s0; x< e0; x++) {
              *d_beg++ = c(*beg1++, *beg2++);
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


}  // end of namespace tf -----------------------------------------------------



