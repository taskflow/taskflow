#pragma once

#include "launch.hpp"

namespace tf {

// Function: make_for_each_task
template <typename B, typename E, typename C, typename P = DefaultPartitioner>
auto make_for_each_task(B b, E e, C c, P part = P()) {

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;

  return [=] (Runtime& rt) mutable {

    // fetch the stateful values
    B_t beg = b;
    E_t end = e;

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= part.chunk_size()) {
      launch_loop(part, [&](){
        std::for_each(beg, end, c);
      });
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
        launch_loop(W, w, rt, part, [=, &c, &part] () mutable {
          part.loop(N, W, curr_b, chunk_size,
            [&, prev_e=size_t{0}](size_t part_b, size_t part_e) mutable {
              std::advance(beg, part_b - prev_e);
              for(size_t x = part_b; x<part_e; x++) {
                c(*beg++);
              }
              prev_e = part_e;
            }
          ); 
        });
      }

      rt.corun_all();
    }
    // dynamic partitioner
    else {
      std::atomic<size_t> next(0);
      launch_loop(N, W, rt, next, part, [=, &c, &next, &part] () mutable {
        part.loop(N, W, next, 
          [&, prev_e=size_t{0}](size_t part_b, size_t part_e) mutable {
            std::advance(beg, part_b - prev_e);
            for(size_t x = part_b; x<part_e; x++) {
              c(*beg++);
            }
            prev_e = part_e;
          }
        );
      });
    }

  };
}

// Function: make_for_each_index_task
template <typename B, typename E, typename S, typename C, typename P = DefaultPartitioner>
auto make_for_each_index_task(B b, E e, S s, C c, P part = P()){

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using S_t = std::decay_t<unwrap_ref_decay_t<S>>;

  return [=] (Runtime& rt) mutable {

    // fetch the iterator values
    B_t beg = b;
    E_t end = e;
    S_t inc = s;
    
    // nothing to be done if the range is invalid
    if(is_range_invalid(beg, end, inc)) {
      return;
    }

    size_t W = rt.executor().num_workers();
    size_t N = distance(beg, end, inc);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= part.chunk_size()) {
      launch_loop(part, [&](){
        for(size_t x=0; x<N; x++, beg+=inc) {
          c(beg);
        }
      });
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
        launch_loop(W, w, rt, part, [=, &c, &part] () mutable {
          part.loop(N, W, curr_b, chunk_size,
            [&](size_t part_b, size_t part_e) {
              auto idx = static_cast<B_t>(part_b) * inc + beg;
              for(size_t x=part_b; x<part_e; x++, idx += inc) {
                c(idx);
              }
            }
          );
        });
      }

      rt.corun_all();
    }
    // dynamic partitioner
    else {
      std::atomic<size_t> next(0);
      launch_loop(N, W, rt, next, part, [=, &c, &next, &part] () mutable {
        part.loop(N, W, next, 
          [&](size_t part_b, size_t part_e) {
            auto idx = static_cast<B_t>(part_b) * inc + beg;
            for(size_t x=part_b; x<part_e; x++, idx += inc) {
              c(idx);
            }
          }
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
  //return Task(_graph._emplace_back("", 0, nullptr, nullptr, 0,
  //  std::in_place_type_t<Node::Static>{}, std::forward<C>(c)
  //));
}

// ----------------------------------------------------------------------------
// for_each_index
// ----------------------------------------------------------------------------

// Function: for_each_index
template <typename B, typename E, typename S, typename C, typename P>
Task FlowBuilder::for_each_index(B beg, E end, S inc, C c, P part){
  return emplace(
    make_for_each_index_task(beg, end, inc, c, part)
  );
}


}  // end of namespace tf -----------------------------------------------------

