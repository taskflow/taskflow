#pragma once

#include "../taskflow.hpp"

namespace tf {

// Function: make_transform_task
template <InputIteratorLike B, InputIteratorLike E, typename O, typename C,
          PartitionerLike P = DefaultPartitioner>
requires UnaryTransformLike<
  C, 
  std::decay_t<std::unwrap_ref_decay_t<B>>,
  std::decay_t<std::unwrap_ref_decay_t<O>>
>
auto make_transform_task(B first1, E last1, O d_first, C c, P part = P()) {
  
  using namespace std::string_literals;

  using B_t = std::decay_t<std::unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<std::unwrap_ref_decay_t<E>>;
  using O_t = std::decay_t<std::unwrap_ref_decay_t<O>>;
  
  return [=] (Runtime& rt) mutable {

    // fetch the stateful values
    B_t beg   = first1;
    E_t end   = last1;
    O_t d_beg = d_first;

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(beg, end);
    
    if(N == 0) {
      return;
    }

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= part.chunk_size()) {
      part([=]() mutable { std::transform(beg, end, d_beg, c); })();
      return;
    }

    if(N < W) {
      W = N;
    }

    // static partitioner
    if constexpr(part.type() == PartitionerType::STATIC) {
      for(size_t w=0, curr_b=0; w<W && curr_b < N;) {
        auto chunk_size = part.adjusted_chunk_size(N, W, w);
        auto task = part([=] () mutable {
          part.loop(N, W, curr_b, chunk_size, [=, prev_e=size_t{0}](size_t part_b, size_t part_e) mutable {
            std::advance(beg, part_b - prev_e);
            std::advance(d_beg, part_b - prev_e);
            for(size_t x = part_b; x<part_e; x++) {
              *d_beg++ = c(*beg++);
            }
            prev_e = part_e;
          });
        });
        (++w == W || (curr_b += chunk_size) >= N) ? task() : rt.silent_async(task);
      }
    }
    // dynamic partitioner
    else {
      auto next = std::make_shared<std::atomic<size_t>>(0);
      for(size_t w=0; w<W;) {
        auto task = part([=] () mutable {
          part.loop(N, W, *next, [=, prev_e=size_t{0}](size_t part_b, size_t part_e) mutable {
            std::advance(beg, part_b - prev_e);
            std::advance(d_beg, part_b - prev_e);
            for(size_t x = part_b; x<part_e; x++) {
              *d_beg++ = c(*beg++);
            }
            prev_e = part_e;
          }); 
        });
        (++w == W) ? task() : rt.silent_async(task);
      }
    }
  };
}

// Function: make_transform_task
template <InputIteratorLike B1, InputIteratorLike E1, InputIteratorLike B2, typename O, typename C,
          PartitionerLike P = DefaultPartitioner>
requires BinaryTransformLike<
  C, 
  std::decay_t<std::unwrap_ref_decay_t<B1>>,
  std::decay_t<std::unwrap_ref_decay_t<B2>>,
  std::decay_t<std::unwrap_ref_decay_t<O>>
>
auto make_transform_task(B1 first1, E1 last1, B2 first2, O d_first, C c, P part = P()) {
  
  using namespace std::string_literals;

  using B1_t = std::decay_t<std::unwrap_ref_decay_t<B1>>;
  using E1_t = std::decay_t<std::unwrap_ref_decay_t<E1>>;
  using B2_t = std::decay_t<std::unwrap_ref_decay_t<B2>>;
  using O_t = std::decay_t<std::unwrap_ref_decay_t<O>>;

  return [=] (Runtime& rt) mutable {

    // fetch the stateful values
    B1_t beg1 = first1;
    E1_t end1 = last1;
    B2_t beg2 = first2;
    O_t d_beg = d_first;

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(beg1, end1);
    
    if(N == 0) {
      return;
    }

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= part.chunk_size()) {
      part([=]() mutable { std::transform(beg1, end1, beg2, d_beg, c); })();
      return;
    }
    
    if(N < W) {
      W = N;
    }

    // static partitioner
    if constexpr(part.type() == PartitionerType::STATIC) {
      for(size_t w=0, curr_b=0; w<W && curr_b < N;) {
        auto chunk_size = part.adjusted_chunk_size(N, W, w);
        auto task = part([=] () mutable {
          part.loop(N, W, curr_b, chunk_size, [=, prev_e=size_t{0}](size_t part_b, size_t part_e) mutable {
            std::advance(beg1, part_b - prev_e);
            std::advance(beg2, part_b - prev_e);
            std::advance(d_beg, part_b - prev_e);
            for(size_t x = part_b; x<part_e; x++) {
              *d_beg++ = c(*beg1++, *beg2++);
            }
            prev_e = part_e;
          });
        });
        (++w == W || (curr_b += chunk_size) >= N) ? task() : rt.silent_async(task);
      }
    }
    // dynamic partitioner
    else {
      auto next = std::make_shared<std::atomic<size_t>>(0);
      for(size_t w=0; w<W;) {
        auto task = part([=] () mutable {
          part.loop(N, W, *next, [=, prev_e=size_t{0}](size_t part_b, size_t part_e) mutable {
            std::advance(beg1, part_b - prev_e);
            std::advance(beg2, part_b - prev_e);
            std::advance(d_beg, part_b - prev_e);
            for(size_t x = part_b; x<part_e; x++) {
              *d_beg++ = c(*beg1++, *beg2++);
            }
            prev_e = part_e;
          });
        });
        (++w == W) ? task() : rt.silent_async(task);
      }
    }
  };
}

// ----------------------------------------------------------------------------
// transform
// ----------------------------------------------------------------------------

// Function: transform
template <InputIteratorLike B, InputIteratorLike E, typename O, typename C, PartitionerLike P>
requires UnaryTransformLike<
  C, 
  std::decay_t<std::unwrap_ref_decay_t<B>>,
  std::decay_t<std::unwrap_ref_decay_t<O>>
>
Task FlowBuilder::transform(B first1, E last1, O d_first, C c, P part) {
  return emplace(
    make_transform_task(first1, last1, d_first, c, part)
  );
}

// ----------------------------------------------------------------------------
// transform2
// ----------------------------------------------------------------------------
  
// Function: transform
template <
  InputIteratorLike B1, 
  InputIteratorLike E1, 
  InputIteratorLike B2, 
  typename O, 
  typename C, 
  PartitionerLike P
>
requires BinaryTransformLike<
  C, 
  std::decay_t<std::unwrap_ref_decay_t<B1>>,
  std::decay_t<std::unwrap_ref_decay_t<B2>>,
  std::decay_t<std::unwrap_ref_decay_t<O>>
>
Task FlowBuilder::transform(
  B1 first1, E1 last1, B2 first2, O d_first, C c, P part
) {
  return emplace(make_transform_task(
    first1, last1, first2, d_first, c, part
  ));
}


}  // end of namespace tf -----------------------------------------------------
