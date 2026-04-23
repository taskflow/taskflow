#pragma once

#include "../taskflow.hpp"

#include <iterator>
#include <type_traits>

namespace tf {

template <typename B, typename E, typename G, PartitionerLike P = DefaultPartitioner>
auto make_generate_task(B first, E last, G gen, P part = P()) {
  using B_t = std::decay_t<std::unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<std::unwrap_ref_decay_t<E>>;
  return [=](Runtime &rt) mutable {
    B_t beg = first;
    E_t end = last;

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(beg, end);

    // the workload should be sequential
    if (W <= 1 || N <= part.chunk_size()) {
      part([=]() mutable { std::generate(beg, end, gen); })();
      return;
    }

    // use no more workers than the iteration count
    if (N < W) {
      W = N;
    }

    // static partitioner
    if constexpr (part.type() == PartitionerType::STATIC) {
      for (size_t w = 0, curr_b = 0; w < W && curr_b < N;) {
        auto chunk_size = part.adjusted_chunk_size(N, W, w);
        auto task = part([=]() mutable {
          part.loop(
              N, W, curr_b, chunk_size,
              [=, prev_e = size_t{0}](size_t part_b, size_t part_e) mutable {
                std::advance(beg, part_b - prev_e);
                for (size_t x = part_b; x < part_e; x++) {
                  *beg++ = gen();
                }
                prev_e = part_e;
              });
        });
        (++w == W || (curr_b += chunk_size) >= N) ? task()
                                                  : rt.silent_async(task);
      }
    }
    // dynamic partitioner
    else {
      auto next = std::make_shared<std::atomic<size_t>>(0);
      for (size_t w = 0; w < W;) {
        auto task = part([=]() mutable {
          part.loop(
              N, W, *next,
              [=, prev_e = size_t{0}](size_t part_b, size_t part_e) mutable {
                std::advance(beg, part_b - prev_e);
                for (size_t x = part_b; x < part_e; x++) {
                  *beg++ = gen();
                }
                prev_e = part_e;
              });
        });
        (++w == W) ? task() : rt.silent_async(task);
      }
    }
  };
}

template <typename B, std::integral C, typename G, PartitionerLike P = DefaultPartitioner>
auto make_generate_n_task(B first, C count, G gen, P part = P()) {
  using B_t = std::decay_t<std::unwrap_ref_decay_t<B>>;

  return [=](Runtime &rt) mutable {
    B_t beg = first;

    size_t W = rt.executor().num_workers();
    size_t N = count;

    // the workload should be sequential
    if (W <= 1 || N <= part.chunk_size()) {
      part([=]() mutable { std::generate_n(beg, count, gen); })();
      return;
    }

    // use no more workers than the iteration count
    if (N < W) {
      W = N;
    }

    // static partitioner
    if constexpr (part.type() == PartitionerType::STATIC) {
      for (size_t w = 0, curr_b = 0; w < W && curr_b < N;) {
        auto chunk_size = part.adjusted_chunk_size(N, W, w);
        auto task = part([=]() mutable {
          part.loop(
              N, W, curr_b, chunk_size,
              [=, prev_e = size_t{0}](size_t part_b, size_t part_e) mutable {
                std::advance(beg, part_b - prev_e);
                for (size_t x = part_b; x < part_e; x++) {
                  *beg++ = gen();
                }
                prev_e = part_e;
              });
        });
        (++w == W || (curr_b += chunk_size) >= N) ? task()
                                                  : rt.silent_async(task);
      }
    }
    // dynamic partitioner
    else {
      auto next = std::make_shared<std::atomic<size_t>>(0);
      for (size_t w = 0; w < W;) {
        auto task = part([=]() mutable {
          part.loop(
              N, W, *next,
              [=, prev_e = size_t{0}](size_t part_b, size_t part_e) mutable {
                std::advance(beg, part_b - prev_e);
                for (size_t x = part_b; x < part_e; x++) {
                  *beg++ = gen();
                }
                prev_e = part_e;
              });
        });
        (++w == W) ? task() : rt.silent_async(task);
      }
    }
  };
}

template <typename B, typename E, typename G, PartitionerLike P>
Task FlowBuilder::generate(B first, E last, G gen, P part) {
  return emplace(make_generate_task(first, last, gen, part));
}

template <typename B, std::integral C, typename G, PartitionerLike P>
Task FlowBuilder::generate_n(B first, C count, G gen, P part) {
  return emplace(make_generate_n_task(first, count, gen, part));
}


} // namespace tf ---------------------------------------------------------------------------------
