#pragma once

#include "../core/executor.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// transform
// ----------------------------------------------------------------------------

// Function: transform
template <typename B, typename E, typename O, typename C>
Task FlowBuilder::transform(B first1, E last1, O d_first, C c) {
  return transform(DefaultExecutionPolicy{}, first1, last1, d_first, c);
}

// Function: transform
template <
  typename P, typename B, typename E, typename O, typename C,
  std::enable_if_t<is_execution_policy_v<P>, void> *
>
Task FlowBuilder::transform(P&& policy, B first1, E last1, O d_first, C c) {

  using namespace std::string_literals;

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using O_t = std::decay_t<unwrap_ref_decay_t<O>>;

  Task task = emplace(
  [first1, last1, d_first, c, policy] (Subflow& sf) mutable {

    // fetch the stateful values
    B_t beg   = first1;
    E_t end   = last1;
    O_t d_beg = d_first;

    if(beg == end) {
      return;
    }

    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= policy.chunk_size()) {
      std::transform(beg, end, d_beg, c);
      return;
    }

    if(N < W) {
      W = N;
    }

    std::atomic<size_t> next(0);
      
    auto loop = [=, &next, &policy] () mutable {
      policy(N, W, next, 
        [&, prev_e=size_t{0}](size_t curr_b, size_t curr_e) mutable {
          std::advance(beg, curr_b - prev_e);
          std::advance(d_beg, curr_b - prev_e);
          for(size_t x = curr_b; x<curr_e; x++) {
            *d_beg++ = c(*beg++);
          }
          prev_e = curr_e;
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
// transform2
// ----------------------------------------------------------------------------

// Function: transform
template <
  typename B1, typename E1, typename B2, typename O, typename C,
  std::enable_if_t<!is_execution_policy_v<B1>, void> *
>
Task FlowBuilder::transform(B1 first1, E1 last1, B2 first2, O d_first, C c) {
  return transform(DefaultExecutionPolicy{}, first1, last1, first2, d_first, c);
}

// Function: transform
template <typename P, typename B1, typename E1, typename B2, typename O, typename C>
Task FlowBuilder::transform(
  P&& policy, B1 first1, E1 last1, B2 first2, O d_first, C c
) {

  using namespace std::string_literals;

  using B1_t = std::decay_t<unwrap_ref_decay_t<B1>>;
  using E1_t = std::decay_t<unwrap_ref_decay_t<E1>>;
  using B2_t = std::decay_t<unwrap_ref_decay_t<B2>>;
  using O_t = std::decay_t<unwrap_ref_decay_t<O>>;

  Task task = emplace(
  [first1, last1, first2, d_first, c, policy] (Subflow& sf) mutable {

    // fetch the stateful values
    B1_t beg1 = first1;
    E1_t end1 = last1;
    B2_t beg2 = first2;
    O_t d_beg = d_first;

    if(beg1 == end1) {
      return;
    }

    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg1, end1);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= policy.chunk_size()) {
      std::transform(beg1, end1, beg2, d_beg, c);
      return;
    }

    if(N < W) {
      W = N;
    }

    std::atomic<size_t> next(0);
    
    auto loop = [=, &next, &policy] () mutable {
      policy(N, W, next, 
        [&, prev_e=size_t{0}](size_t curr_b, size_t curr_e) mutable {
          std::advance(beg1, curr_b - prev_e);
          std::advance(beg2, curr_b - prev_e);
          std::advance(d_beg, curr_b - prev_e);
          for(size_t x = curr_b; x<curr_e; x++) {
            *d_beg++ = c(*beg1++, *beg2++);
          }
          prev_e = curr_e;
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



