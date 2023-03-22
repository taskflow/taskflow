#pragma once

#include "../core/executor.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// for_each
// ----------------------------------------------------------------------------

// Function: for_each
template <typename B, typename E, typename C>
Task FlowBuilder::for_each(B beg, E end, C c) {
  return for_each(DefaultExecutionPolicy{}, beg, end, c);
}

// Function: for_each
template <typename P, typename B, typename E, typename C>
Task FlowBuilder::for_each(P&& policy, B beg, E end, C c) {

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using namespace std::string_literals;

  Task task = emplace([b=beg, e=end, c, policy] (Runtime& sf) mutable {

    // fetch the stateful values
    B_t beg = b;
    E_t end = e;

    if(beg == end) {
      return;
    }

    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= policy.chunk_size()) {
      std::for_each(beg, end, c);
      return;
    }

    if(N < W) {
      W = N;
    }
    
    // static partitioner
    if constexpr(std::decay_t<P>::is_static_partitioner) {

      size_t curr_b = 0;
      size_t chunk_size;

      for(size_t w=0; w<W && curr_b < N; ++w, curr_b += chunk_size) {
      
        chunk_size = policy.chunk_size() == 0 ? 
                     N/W + (w < N%W) : policy.chunk_size();

        auto loop = [=, &policy] () mutable {
          policy(N, W, curr_b, chunk_size,
            [&, prev_e=size_t{0}](size_t curr_b, size_t curr_e) mutable {
              std::advance(beg, curr_b - prev_e);
              for(size_t x = curr_b; x<curr_e; x++) {
                c(*beg++);
              }
              prev_e = curr_e;
            }
          ); 
        };

        if(w == W-1) {
          loop();
        }
        else {
          sf._silent_async(sf._worker, "loop-"s + std::to_string(w), loop);
        }
      }

      sf.join();
    }
    // dynamic partitioner
    else {
      std::atomic<size_t> next(0);

      auto loop = [=, &next, &policy] () mutable {
        policy(N, W, next, 
          [&, prev_e=size_t{0}](size_t curr_b, size_t curr_e) mutable {
            std::advance(beg, curr_b - prev_e);
            for(size_t x = curr_b; x<curr_e; x++) {
              c(*beg++);
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
          sf._silent_async(sf._worker, "loop-"s + std::to_string(w), loop);
        }
      }
      // need to join here in case next goes out of scope
      sf.join();
    }
  });

  return task;
}

// ----------------------------------------------------------------------------
// for_each_index
// ----------------------------------------------------------------------------

// Function: for_each_index
template <typename B, typename E, typename S, typename C>
Task FlowBuilder::for_each_index(B beg, E end, S inc, C c){
  return for_each_index(DefaultExecutionPolicy{}, beg, end, inc, c);
}

// Function: for_each_index
template <typename P, typename B, typename E, typename S, typename C>
Task FlowBuilder::for_each_index(P&& policy, B beg, E end, S inc, C c){

  using namespace std::string_literals;

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using S_t = std::decay_t<unwrap_ref_decay_t<S>>;

  Task task = emplace([b=beg, e=end, a=inc, c, policy] (Runtime& sf) mutable {

    // fetch the iterator values
    B_t beg = b;
    E_t end = e;
    S_t inc = a;

    size_t W = sf._executor.num_workers();
    size_t N = distance(beg, end, inc);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= policy.chunk_size()) {
      for(size_t x=0; x<N; x++, beg+=inc) {
        c(beg);
      }
      return;
    }

    if(N < W) {
      W = N;
    }
    
    // static partitioner
    if constexpr(std::decay_t<P>::is_static_partitioner) {

      size_t curr_b = 0;
      size_t chunk_size;

      for(size_t w=0; w<W && curr_b < N; ++w, curr_b += chunk_size) {
      
        chunk_size = policy.chunk_size() == 0 ? 
                     N/W + (w < N%W) : policy.chunk_size();

        auto loop = [=, &policy] () mutable {
          policy(N, W, curr_b, chunk_size,
            [&](size_t curr_b, size_t curr_e) {
              auto idx = static_cast<B_t>(curr_b) * inc + beg;
              for(size_t x=curr_b; x<curr_e; x++, idx += inc) {
                c(idx);
              }
            }
          ); 
        };

        if(w == W-1) {
          loop();
        }
        else {
          sf._silent_async(sf._worker, "loop-"s + std::to_string(w), loop);
        }
      }

      sf.join();
    }
    // dynamic partitioner
    else {
      std::atomic<size_t> next(0);
      
      auto loop = [=, &next, &policy] () mutable {
        policy(N, W, next, 
          [&](size_t curr_b, size_t curr_e) {
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
        if(r <= policy.chunk_size() || w == W-1) {
          loop(); 
          break;
        }
        else {
          sf._silent_async(sf._worker, "loop-"s + std::to_string(w), loop);
        }
      }

      // need to join here in case next goes out of scope
      sf.join();
    }
  });

  return task;
}

}  // end of namespace tf -----------------------------------------------------

