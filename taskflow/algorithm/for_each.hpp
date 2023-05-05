#pragma once

#include "launch.hpp"

namespace tf {

namespace detail {

// Function: make_for_each_task
template <typename B, typename E, typename C, typename P>
TF_FORCE_INLINE auto make_for_each_task(B beg, E end, C c, P&& part) {
  
  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using namespace std::string_literals;

  return [b=beg, e=end, c, part=std::forward<P>(part)] (Runtime& rt) mutable {

    // fetch the stateful values
    B_t beg = b;
    E_t end = e;

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= part.chunk_size()) {
      std::for_each(beg, end, c);
      return;
    }

    if(N < W) {
      W = N;
    }
    
    // static partitioner
    if constexpr(std::is_same_v<std::decay_t<P>, StaticPartitioner>) {
      size_t chunk_size;
      for(size_t w=0, curr_b=0; w<W && curr_b < N; ++w, curr_b += chunk_size) {
        chunk_size = part.adjusted_chunk_size(N, W, w);
        launch_loop(W, w, rt, [=, &c, &part] () mutable {
          part.loop(N, W, curr_b, chunk_size,
            [&, prev_e=size_t{0}](size_t curr_b, size_t curr_e) mutable {
              std::advance(beg, curr_b - prev_e);
              for(size_t x = curr_b; x<curr_e; x++) {
                c(*beg++);
              }
              prev_e = curr_e;
            }
          ); 
        });
      }

      rt.join();
    }
    // dynamic partitioner
    else {
      std::atomic<size_t> next(0);
      launch_loop(N, W, rt, next, part, [=, &c, &next, &part] () mutable {
        part.loop(N, W, next, 
          [&, prev_e=size_t{0}](size_t curr_b, size_t curr_e) mutable {
            std::advance(beg, curr_b - prev_e);
            for(size_t x = curr_b; x<curr_e; x++) {
              c(*beg++);
            }
            prev_e = curr_e;
          }
        ); 
      });
    }
  };
}

// Function: make_for_each_index_task
template <typename B, typename E, typename S, typename C, typename P>
TF_FORCE_INLINE auto make_for_each_index_task(B beg, E end, S inc, C c, P&& part){

  using namespace std::string_literals;

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using S_t = std::decay_t<unwrap_ref_decay_t<S>>;

  return [b=beg, e=end, a=inc, c, part=std::forward<P>(part)] 
  (Runtime& rt) mutable {

    // fetch the iterator values
    B_t beg = b;
    E_t end = e;
    S_t inc = a;

    size_t W = rt.executor().num_workers();
    size_t N = distance(beg, end, inc);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= part.chunk_size()) {
      for(size_t x=0; x<N; x++, beg+=inc) {
        c(beg);
      }
      return;
    }

    if(N < W) {
      W = N;
    }
    
    // static partitioner
    if constexpr(std::is_same_v<std::decay_t<P>, StaticPartitioner>) {

      size_t chunk_size;

      for(size_t w=0, curr_b=0; w<W && curr_b < N; ++w, curr_b += chunk_size) {
        chunk_size = part.adjusted_chunk_size(N, W, w);
        launch_loop(W, w, rt, [=, &c, &part] () mutable {
          part.loop(N, W, curr_b, chunk_size,
            [&](size_t curr_b, size_t curr_e) {
              auto idx = static_cast<B_t>(curr_b) * inc + beg;
              for(size_t x=curr_b; x<curr_e; x++, idx += inc) {
                c(idx);
              }
            }
          ); 
        });
      }

      rt.join();
    }
    // dynamic partitioner
    else {
      std::atomic<size_t> next(0);
      launch_loop(N, W, rt, next, part, [=, &c, &next, &part] () mutable {
        part.loop(N, W, next, 
          [&](size_t curr_b, size_t curr_e) {
            auto idx = static_cast<B_t>(curr_b) * inc + beg;
            for(size_t x=curr_b; x<curr_e; x++, idx += inc) {
              c(idx);
            }
          }
        ); 
      });
    }
  };
}

}  // end of namespace detail -------------------------------------------------

// ----------------------------------------------------------------------------
// for_each
// ----------------------------------------------------------------------------

// Function: for_each
template <typename B, typename E, typename C, typename P>
Task FlowBuilder::for_each(B beg, E end, C c, P&& part) {
  return emplace(
    detail::make_for_each_task(beg, end, c, std::forward<P>(part))
  );
}

// ----------------------------------------------------------------------------
// for_each_index
// ----------------------------------------------------------------------------

// Function: for_each_index
template <typename B, typename E, typename S, typename C, typename P>
Task FlowBuilder::for_each_index(B beg, E end, S inc, C c, P&& part){
  return emplace(
    detail::make_for_each_index_task(beg, end, inc, c, std::forward<P>(part))
  );
}


}  // end of namespace tf -----------------------------------------------------

