#pragma once

#include "launch.hpp"

namespace tf {

// Function: make_for_each_task
template <typename B, typename E, typename C, typename P = DefaultPartitioner>
TF_FORCE_INLINE auto make_for_each_task(B b, E e, C c, P part = P()) {

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using namespace std::string_literals;

  return [=] (Runtime& rt) mutable {

    // fetch the stateful values
    B_t beg = b;
    E_t end = e;

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= part.chunk_size()) {
      TF_MAKE_LOOP_TASK(
        std::for_each(beg, end, c);
      );
      return;
    }

    if(N < W) {
      W = N;
    }
    
    // static partitioner
    if constexpr(part.type() == PartitionerType::STATIC) {
      size_t chunk_size;
      for(size_t w=0, curr_b=0; w<W && curr_b < N; ++w, curr_b += chunk_size) {
        chunk_size = part.adjusted_chunk_size(N, W, w);
        launch_loop(W, w, rt, [=, &c, &part] () mutable {
          TF_MAKE_LOOP_TASK(
            part.loop(N, W, curr_b, chunk_size,
              [&, prev_e=size_t{0}](size_t part_b, size_t part_e) mutable {
                std::advance(beg, part_b - prev_e);
                for(size_t x = part_b; x<part_e; x++) {
                  c(*beg++);
                }
                prev_e = part_e;
              }
            ); 
          );
        });
      }

      rt.corun_all();
    }
    // dynamic partitioner
    else {
      std::atomic<size_t> next(0);
      launch_loop(N, W, rt, next, part, [=, &c, &next, &part] () mutable {
        TF_MAKE_LOOP_TASK(
          part.loop(N, W, next, 
            [&, prev_e=size_t{0}](size_t part_b, size_t part_e) mutable {
              std::advance(beg, part_b - prev_e);
              for(size_t x = part_b; x<part_e; x++) {
                c(*beg++);
              }
              prev_e = part_e;
            }
          ); 
        );
      });
    }
  };
}

template<typename T, typename C>
using is_index_func = std::is_invocable_r<void, C, T>;

template<typename T, typename C>
using is_range_func = std::is_invocable_r<void, C, T, T>;

// Function: make_for_each_index_task
template <typename T, typename C, typename P = DefaultPartitioner>
TF_FORCE_INLINE auto make_for_each_index_task(T b, T e, C c, P part = P()){
  using T_t = std::decay_t<unwrap_ref_decay_t<T>>;

  static_assert(std::is_integral<T_t>::value, "Begin and end values must be an integral type.");
  static_assert(
        std::disjunction<is_index_func<T_t, C>, is_range_func<T_t, C>>::value,
        "C must be either a void function taking one int or two ints"
  );
  constexpr bool is_index_callable = is_index_func<T, C>::value;

  using namespace std::string_literals;

  return [=] (Runtime& rt) mutable {

    // fetch the iterator values
    T_t beg = b;
    T_t end = e;
    
    // nothing to be done if the range is invalid
    if(is_range_invalid(beg, end, 1)) {
      return;
    }

    size_t W = rt.executor().num_workers();
    T_t n = end-beg;
    size_t N = static_cast<size_t>(n);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= part.chunk_size()) {
      TF_MAKE_LOOP_TASK(
        if constexpr(is_index_callable) {
          for(T_t i=0; i<n; i++) {
            c(i);
          }
        } else {
          c(0, n);
        }
      );
      return;
    }

    if(N < W) {
      W = N;
    }
    
    // static partitioner
    if constexpr(part.type() == PartitionerType::STATIC) {
      size_t chunk_size;
      for(size_t w=0, curr_b=0; w<W && curr_b < N; ++w, curr_b += chunk_size) {
        chunk_size = part.adjusted_chunk_size(N, W, w);
        launch_loop(W, w, rt, [=, &c, &part] () mutable {
          TF_MAKE_LOOP_TASK(
            part.loop(N, W, curr_b, chunk_size,
              [&](size_t part_b, size_t part_e) {
                if constexpr(is_index_callable) {
                  for(size_t i=part_b; i<part_e; i++) {
                    c(static_cast<T_t>(i));
                  }
                } else {
                  c(static_cast<T_t>(part_b), static_cast<T_t>(part_e));
                }
              }
            ); 
          );
        });
      }

      rt.corun_all();
    }
    // dynamic partitioner
    else {
      std::atomic<size_t> next(0);
      launch_loop(N, W, rt, next, part, [=, &c, &next, &part] () mutable {
        TF_MAKE_LOOP_TASK(
          part.loop(N, W, next, 
            [&](size_t part_b, size_t part_e) {
              if constexpr(is_index_callable) {
                for(size_t i=part_b; i<part_e; i++) {
                  c(static_cast<T_t>(i));
                }
              } else {
                c(static_cast<T_t>(part_b), static_cast<T_t>(part_e));
              }
            }
          ); 
        );
      });
    }
  };
}

// ----------------------------------------------------------------------------
// for_each
// ----------------------------------------------------------------------------

// Function: for_each
template <typename B, typename E, typename C, typename P>
Task FlowBuilder::for_each(B beg, E end, C c, P part) {
  return emplace(
    make_for_each_task(beg, end, c, part)
  );
}

// ----------------------------------------------------------------------------
// for_each_index
// ----------------------------------------------------------------------------

// Function: for_each_index
template <typename T, typename C, typename P>
Task FlowBuilder::for_each_index(T beg, T end, C c, P part){
  return emplace(
    make_for_each_index_task(beg, end, c, part)
  );
}


}  // end of namespace tf -----------------------------------------------------
