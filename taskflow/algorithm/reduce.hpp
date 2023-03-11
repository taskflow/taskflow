#pragma once

#include "execution_policy.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// default reduction
// ----------------------------------------------------------------------------

// Function: reduce
template <typename B, typename E, typename T, typename O>
Task FlowBuilder::reduce(B beg, E end, T& init, O bop) {
  return reduce(DefaultExecutionPolicy{}, beg, end, init, bop);
}

// Function: reduce
template <typename P, typename B, typename E, typename T, typename O>
Task FlowBuilder::reduce(P&& policy, B beg, E end, T& init, O bop) {

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using namespace std::string_literals;

  Task task = emplace([b=beg, e=end, &r=init, bop, policy] (Subflow& sf) mutable {

    // fetch the iterator values
    B_t beg = b;
    E_t end = e;

    if(beg == end) {
      return;
    }

    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= policy.chunk_size()) {
      for(; beg!=end; r = bop(r, *beg++));
      return;
    }

    if(N < W) {
      W = N;
    }

    std::mutex mutex;
    std::atomic<size_t> next(0);

    auto loop = [=, &mutex, &next, &r, &policy] () mutable {
      
      // pre-reduce
      size_t s0 = next.fetch_add(2, std::memory_order_relaxed);

      if(s0 >= N) {
        return;
      }

      std::advance(beg, s0);

      if(N - s0 == 1) {
        std::lock_guard<std::mutex> lock(mutex);
        r = bop(r, *beg);
        return;
      }

      auto beg1 = beg++;
      auto beg2 = beg++;

      T sum = bop(*beg1, *beg2);
      
      // loop reduce
      policy(N, W, next, 
        [&, prev_e=size_t{s0+2}](size_t curr_b, size_t curr_e) mutable {
          std::advance(beg, curr_b - prev_e);
          for(size_t x=curr_b; x<curr_e; x++, beg++) {
            sum = bop(sum, *beg);
          }
          prev_e = curr_e;
        }
      ); 
      
      // final reduce
      std::lock_guard<std::mutex> lock(mutex);
      r = bop(r, sum);
    };

    for(size_t w=0; w<W; w++) {
      auto r = N - next.load(std::memory_order_relaxed);
      // no more loop work to do - finished by previous async tasks
      if(!r) {
        break;
      }
      // tail optimization
      if(r <= policy.chunk_size() || w == W-1) {
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

// ----------------------------------------------------------------------------
// default transform and reduction
// ----------------------------------------------------------------------------

// Function: transform_reduce
template <typename B, typename E, typename T, typename BOP, typename UOP>
Task FlowBuilder::transform_reduce(B beg, E end, T& init, BOP bop, UOP uop) {
  return transform_reduce(DefaultExecutionPolicy{}, beg, end, init, bop, uop);
}

// Function: transform_reduce
template <typename P, typename B, typename E, typename T, typename BOP, typename UOP>
Task FlowBuilder::transform_reduce(
  P&& policy, B beg, E end, T& init, BOP bop, UOP uop
) {

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using namespace std::string_literals;

  Task task = emplace([b=beg, e=end, &r=init, bop, uop, policy] (Subflow& sf) mutable {

    // fetch the iterator values
    B_t beg = b;
    E_t end = e;

    if(beg == end) {
      return;
    }

    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= policy.chunk_size()) {
      for(; beg!=end; r = bop(std::move(r), uop(*beg++)));
      return;
    }

    if(N < W) {
      W = N;
    }

    std::mutex mutex;
    std::atomic<size_t> next(0);
      
    auto loop = [=, &mutex, &next, &r, &policy] () mutable {

      // pre-reduce
      size_t s0 = next.fetch_add(2, std::memory_order_relaxed);

      if(s0 >= N) {
        return;
      }

      std::advance(beg, s0);

      if(N - s0 == 1) {
        std::lock_guard<std::mutex> lock(mutex);
        r = bop(std::move(r), uop(*beg));
        return;
      }

      auto beg1 = beg++;
      auto beg2 = beg++;

      T sum = bop(uop(*beg1), uop(*beg2));
      
      // loop reduce
      policy(N, W, next, 
        [&, prev_e=size_t{s0+2}](size_t curr_b, size_t curr_e) mutable {
          std::advance(beg, curr_b - prev_e);
          for(size_t x=curr_b; x<curr_e; x++, beg++) {
            sum = bop(std::move(sum), uop(*beg));
          }
          prev_e = curr_e;
        }
      ); 
      
      // final reduce
      std::lock_guard<std::mutex> lock(mutex);
      r = bop(std::move(r), std::move(sum));
    };

    for(size_t w=0; w<W; w++) {
      auto r = N - next.load(std::memory_order_relaxed);
      // no more loop work to do - finished by previous async tasks
      if(!r) {
        break;
      }
      // tail optimization
      if(r <= policy.chunk_size() || w == W-1) {
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




