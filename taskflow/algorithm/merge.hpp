#pragma once

#include "../taskflow.hpp"
#include <cstddef>

namespace tf {
template <typename B1, typename E1, typename B2, typename E2, typename O,
          typename C, typename P = DefaultPartitioner>
auto make_merge_task(B1 first1, E1 last1, B2 first2, E2 last2, C cmp, O d_first,
                     P part = P()) {
  using B1_t = std::decay_t<unwrap_ref_decay_t<B1>>;
  using E1_t = std::decay_t<unwrap_ref_decay_t<E1>>;
  using B2_t = std::decay_t<unwrap_ref_decay_t<B2>>;
  using E2_t = std::decay_t<unwrap_ref_decay_t<E2>>;
  using O_t = std::decay_t<unwrap_ref_decay_t<O>>;
  return [=](Runtime &rt) mutable {
    B1_t beg1 = first1;
    E1_t end1 = last1;
    B2_t beg2 = first2;
    E2_t end2 = last2;
    O_t d_beg = d_first;

    size_t n = static_cast<size_t>(std::distance(beg1, end1));
    size_t m = static_cast<size_t>(std::distance(beg2, end2));

    size_t W = rt.executor().num_workers();
    size_t N = n + m;

    // only myself - no need to spawn another graph
    if (W <= 1 || N <= part.chunk_size()) {
      part([=]() mutable { std::merge(beg1, end1, beg2, end2, d_beg, cmp); })();
      return;
    }

    if (N < W) {
      W = N;
    }

    auto co_rank = [cmp, n, m](B1_t f1, B2_t f2, size_t rank) {
      // Prevents underflow if rank < m
      size_t low = (rank > m) ? rank - m : 0;
      size_t high = std::min(n, rank);

      while (low < high) {
        size_t i = low + (high - low) / 2;
        size_t j = rank - i;

        auto pt1 = f1;
        std::advance(pt1, i);
        auto pt2 = f2;
        std::advance(pt2, j - 1);
        // *pt1 <= *pt2
        if (!cmp(*pt2, *pt1)) {
          low = i + 1;
        } else {
          high = i;
        }
      }
      return std::pair(low, rank - low);
    };

    // static partitioner
    if constexpr (part.type() == PartitionerType::STATIC) {
      for (size_t w = 0, curr_b = 0; w < W && curr_b < N;) {
        auto chunk_size = part.adjusted_chunk_size(N, W, w);
        auto task = part([=]() mutable {
          part.loop(
              N, W, curr_b, chunk_size,
              [=, prev_e = size_t{0}](size_t part_b, size_t part_e) mutable {
                auto [b1_ind, b2_ind] = co_rank(beg1, beg2, part_b);
                auto [e1_ind, e2_ind] = co_rank(beg1, beg2, part_e);

                auto b1 = beg1;
                auto b2 = beg2;
                auto e1 = beg1;
                auto e2 = beg2;

                std::advance(b1, b1_ind);
                std::advance(b2, b2_ind);
                std::advance(e1, e1_ind);
                std::advance(e2, e2_ind);

                std::advance(d_beg, part_b - prev_e);

                d_beg = std::merge(b1, e1, b2, e2, d_beg, cmp);
                prev_e = part_e;
              });
        });
        (++w == W || (curr_b += chunk_size) >= N) ? task()
                                                  : rt.silent_async(task);
      }
    } else {
      auto next = std::make_shared<std::atomic<size_t>>(0);
      for (size_t w = 0; w < W;) {
        auto task = part([=]() mutable {
          part.loop(
              N, W, *next,
              [=, prev_e = size_t{0}](size_t part_b, size_t part_e) mutable {
                auto [b1_ind, b2_ind] = co_rank(beg1, beg2, part_b);
                auto [e1_ind, e2_ind] = co_rank(beg1, beg2, part_e);

                auto b1 = beg1;
                auto b2 = beg2;
                auto e1 = beg1;
                auto e2 = beg2;

                std::advance(b1, b1_ind);
                std::advance(b2, b2_ind);
                std::advance(e1, e1_ind);
                std::advance(e2, e2_ind);

                std::advance(d_beg, part_b - prev_e);

                d_beg = std::merge(b1, e1, b2, e2, d_beg, cmp);
                prev_e = part_e;
              });
        });
        (++w == W) ? task() : rt.silent_async(task);
      }
    }
  };
}

// ----------------------------------------------------------------------------
// tf::Taskflow::merge
// ----------------------------------------------------------------------------

// Function: merge
template <typename B1, typename E1, typename B2, typename E2, typename O,
          typename P,
          std::enable_if_t<is_partitioner_v<std::decay_t<P>>, void> *>
Task FlowBuilder::merge(B1 first1, E1 last1, B2 first2, E2 last2, O d_first,
                        P part) {
  return emplace(make_merge_task(first1, last1, first2, last2, std::less<>{},
                                 d_first, part));
}

// Function: merge
template <
    typename B1, typename E1, typename B2, typename E2, typename O, typename C,
    typename P,
    std::enable_if_t<!is_partitioner_v<std::decay_t<C>>, void> *>
Task FlowBuilder::merge(B1 first1, E1 last1, B2 first2, E2 last2, O d_first,
                        C cmp, P part) {
  return emplace(
      make_merge_task(first1, last1, first2, last2, cmp, d_first, part));
}
} // namespace tf
