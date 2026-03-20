#pragma once

#include "../taskflow.hpp"

namespace tf {
template <typename B, typename E, typename O, typename P = DefaultPartitioner>
auto make_merge_task(B first1, E last1, B first2, E last2, O d_first,
                     P part = P()) {
  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using O_t = std::decay_t<unwrap_ref_decay_t<O>>;
  return [=](Runtime& rt) mutable {
    B_t beg1 = first1;
    E_t end1 = last1;
    B_t beg2 = first2;
    E_t end2 = last2;
    O_t d_beg = d_first;

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(beg1, end1);

    // only myself - no need to spawn another graph
    if (W <= 1 || N <= part.chunk_size()) {
      part([=]() mutable { std::merge(beg1, end1, beg2, end2, d_beg); });
      return;
    }

    if (N < W) {
      W = N;
    }

    auto merge = [](B_t first1, E_t last1, B_t first2, E_t last2, O& dest) {
      while (first1 != last1 && first2 != last2) {
        if (*first1 <= *first2) {
          *(dest++) = *(first1++);
        } else {
          *(dest++) = *(first2++);
        }
      }
      if (first1 == last1 && first2 != last2) {
        while (first2 != last2) {
          *(dest++) = *(first2++);
        }
      } else if (first2 == last2 && first1 != last1) {
        while (first1 != last1) {
          *(dest++) = *(first1++);
        }
      }
    };

    // static partitioner
    if constexpr (part.type() == PartitionerType::STATIC) {
      for (size_t w = 0, curr_b = 0; w < W && curr_b < N;) {
        auto chunk_size = part.adjusted_chunk_size(N, W, w);
        auto task = part([=]() mutable {
          part.loop(
              N, W, curr_b, chunk_size,
              [=, prev_e = size_t{0}, prev_e2_it = beg2](
                  size_t part_b, size_t part_e) mutable {
                end1 = beg1;
                std::advance(beg1, part_b - prev_e);
                std::advance(end1, part_e - prev_e);

                B_t beg2_it =
                    (part_b == 0) ? beg2 : std::lower_bound(beg2, end2, *beg1);
                E_t end2_it =
                    (part_e == N) ? end2 : std::lower_bound(beg2, end2, *end1);

                std::advance(d_beg, part_b - prev_e +
                                        std::distance(prev_e2_it, beg2_it));

                merge(beg1, end1, beg2_it, end2_it, d_beg);
                prev_e = part_e;
                beg1 = end1;
                prev_e2_it = end2_it;
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
              [=, prev_e = size_t{0}, prev_e2_it = beg2](
                  size_t part_b, size_t part_e) mutable {
                end1 = beg1;
                std::advance(beg1, part_b - prev_e);
                std::advance(end1, part_e - prev_e);

                B_t beg2_it =
                    (part_b == 0) ? beg2 : std::lower_bound(beg2, end2, *beg1);
                E_t end2_it =
                    (part_e == N) ? end2 : std::lower_bound(beg2, end2, *end1);

                std::advance(d_beg, part_b - prev_e +
                                        std::distance(prev_e2_it, beg2_it));

                std::merge(beg1, end1, beg2_it, end2_it, d_beg);
                std::advance(d_beg,
                             part_e - part_b + std::distance(beg2_it, end2_it));
                prev_e = part_e;
                beg1 = end1;
                prev_e2_it = end2_it;
              });
        });
        (++w == W) ? task() : rt.silent_async(task);
      }
    }
  };
}
template <typename B1, typename E1, typename B2, typename E2, typename O,
          typename P>
Task FlowBuilder::merge(B1 first1, E1 last1, B2 first2, E2 last2, O d_first,
                        P part) {
  return emplace(make_merge_task(first1, last1, first2, last2, d_first, part));
}
}  // namespace tf
