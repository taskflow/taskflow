#pragma once

#include "../taskflow.hpp"

#include <iterator>
#include <type_traits>

namespace tf {
template <typename F, typename G, typename P = DefaultPartitioner>
auto make_generate_task(F first, F last, G gen, P part = P()) {
  using F_t = std::decay_t<std::unwrap_ref_decay_t<F>>;
  return [=](Runtime &rt) mutable {
    F_t beg = first;
    F_t end = last;

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

template <typename F, typename C, typename G, typename P = DefaultPartitioner>
auto make_generate_n_task(F first, C count, G gen, P part = P()) {
  F last = first;
  std::advance(last, count);
  return make_generate_task(first, last, gen, part);
}

template <typename F, typename G, typename P>
  requires Partitioner<std::decay_t<P>> && std::forward_iterator<F>
Task FlowBuilder::generate(F first, F last, G gen, P part) {
  return emplace(make_generate_task(first, last, gen, part));
}

template <typename F, typename C, typename G, typename P>
  requires(Partitioner<std::decay_t<P>> && std::forward_iterator<F> &&
           std::integral<C>)
Task FlowBuilder::generate_n(F first, C count, G gen, P part) {
  return emplace(make_generate_n_task(first, count, gen, part));
}
} // namespace tf