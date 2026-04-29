#pragma once

#include "../taskflow.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// make_merge_task
// ----------------------------------------------------------------------------

/**
@brief creates a parallel merge task over two sorted ranges

@tparam B1 iterator type for the beginning of the first range
@tparam E1 iterator type for the end of the first range
@tparam B2 iterator type for the beginning of the second range
@tparam E2 iterator type for the end of the second range
@tparam O  output iterator type
@tparam C  comparator type

@param first1  iterator to the beginning of the first sorted range
@param last1   iterator to the end of the first sorted range
@param first2  iterator to the beginning of the second sorted range
@param last2   iterator to the end of the second sorted range
@param cmp     comparator defining the sort order
@param d_first iterator to the beginning of the output range

Returns a callable suitable for @c tf::Executor::async,
@c tf::Executor::silent_async, or @c tf::Taskflow::emplace.

@par Algorithm

The algorithm parallelizes @c std::merge using the @em co-rank technique,
which avoids any inter-worker synchronization during the merge itself.

<b>Step 1 — Flat output partitioning.</b>
The merged output has @c N = n + m elements (where @c n and @c m are the
sizes of the two input ranges).  The output is divided into @c W equal
contiguous chunks, one per worker thread.  Worker @c w is responsible for
writing output positions <tt>[w*K, (w+1)*K)</tt> where @c K = N/W.

<b>Step 2 — Co-rank: finding the input sub-ranges.</b>
For each output chunk boundary @c rank, we need to know: "among the first
@c rank elements of the merged output, how many came from seq1 (call it
@c i) and how many from seq2 (call it @c j = rank - i)?"

Once @c (i, j) is known, the worker's input from seq1 is
<tt>seq1[i_start .. i_end)</tt> and from seq2 is
<tt>seq2[j_start .. j_end)</tt> — both independent slices that can be
merged in isolation.

Co-rank finds @c i by binary search over the range
<tt>[max(0, rank-m), min(n, rank)]</tt>.  For a candidate @c i with
@c j = rank - i, the partition is correct when:

@code
  seq1[i-1] <= seq2[j]    and    seq2[j-1] <= seq1[i]
@endcode

i.e. the last element of each slice does not exceed the first unused element
of the other sequence.  Because both sequences are sorted, only one of these
two conditions needs to be checked — as @c i increases, the correctness
condition transitions monotonically from "too small" to "correct", making
binary search applicable.

The special case @c j==0 (all @c rank elements come from seq1, none from
seq2) skips the @c seq2[j-1] dereference entirely, which would otherwise be
an out-of-bounds access.

<b>Step 3 — Independent parallel merge.</b>
Each worker calls @c std::merge on its identified input sub-slices and writes
directly to its non-overlapping portion of the output.  No mutex, no atomic
cursor, no synchronization.

<b>Why no partitioner?</b>
Unlike @c for_each or @c find, where dynamic/guided scheduling can adapt to
unequal per-element costs, @c std::merge on a chunk of size @c K always costs
exactly O(K) regardless of data values.  There is no load imbalance to
mitigate.  Static equal partitioning (W chunks, one per worker) is always
optimal.  Smaller chunks would only add unnecessary @c co_rank overhead
(O(log N) per boundary) without any benefit.

<b>Complexity.</b>
Each worker performs 2 co-rank calls at O(log N) each, plus one
@c std::merge at O(N/W).  Total: O(N + W log N), which is effectively O(N)
for typical W << N.

@note Both input ranges must be sorted with respect to @c cmp.
      The output range must not overlap either input range.
      Both input iterators must be random-access iterators.
*/
template <typename B1, typename E1, typename B2, typename E2, typename O, typename C>
auto make_merge_task(B1 first1, E1 last1, B2 first2, E2 last2, C cmp, O d_first) {

  using B1_t = std::decay_t<std::unwrap_ref_decay_t<B1>>;
  using E1_t = std::decay_t<std::unwrap_ref_decay_t<E1>>;
  using B2_t = std::decay_t<std::unwrap_ref_decay_t<B2>>;
  using E2_t = std::decay_t<std::unwrap_ref_decay_t<E2>>;
  using O_t  = std::decay_t<std::unwrap_ref_decay_t<O>>;

  return [=](Runtime& rt) mutable {

    B1_t beg1  = first1;
    E1_t end1  = last1;
    B2_t beg2  = first2;
    E2_t end2  = last2;
    O_t  d_beg = d_first;

    size_t n = static_cast<size_t>(std::distance(beg1, end1));
    size_t m = static_cast<size_t>(std::distance(beg2, end2));
    size_t N = n + m;
    size_t W = rt.executor().num_workers();

    if(N == 0) {
      return;
    }

    // sequential fallback for single worker or small N
    if(W <= 1 || N <= W) {
      std::merge(beg1, end1, beg2, end2, d_beg, cmp);
      return;
    }

    // co_rank: binary search for the unique split point i in [low, high)
    // such that taking i elements from seq1 and (rank-i) from seq2 gives
    // exactly the rank smallest merged elements.
    //
    // Invariant: seq2[j-1] <= seq1[i] means i is not too large (high = i),
    //            otherwise i is too small (low = i+1).
    //
    // Guard: when j==0, skip the seq2[j-1] dereference (would be seq2[-1])
    // and always increase low — if seq2 contributes nothing, i must grow.
    auto co_rank = [&](size_t rank) -> std::pair<size_t, size_t> {
      size_t low  = (rank > m) ? rank - m : 0;
      size_t high = (std::min)(n, rank);

      while(low < high) {
        size_t i = low + (high - low) / 2;
        size_t j = rank - i;

        if(j == 0 || !cmp(*std::next(beg2, j - 1), *std::next(beg1, i))) {
          low = i + 1;  // seq1[i] <= seq2[j-1]: i is too small, take more from seq1
        } else {
          high = i;     // seq1[i] >  seq2[j-1]: i is too large
        }
      }
      return {low, rank - low};
    };

    // merge_chunk: given an output range [part_b, part_e), use co_rank to
    // find the corresponding input sub-ranges and merge them independently.
    // d_beg is captured by value (= d_first), so each worker positions its
    // output pointer as d_first + part_b with no shared mutable state.
    auto merge_chunk = [&](size_t part_b, size_t part_e) {
      auto [i_beg, j_beg] = co_rank(part_b);
      auto [i_end, j_end] = co_rank(part_e);
      std::merge(
        std::next(beg1, i_beg), std::next(beg1, i_end),
        std::next(beg2, j_beg), std::next(beg2, j_end),
        std::next(d_beg, part_b),
        cmp
      );
    };

    // static equal partition: W chunks, one per worker.
    // Each chunk boundary costs 2 co_rank calls O(log N) — smaller chunks
    // would add overhead with no load-balancing benefit since std::merge
    // always costs O(K) for a chunk of size K regardless of data.
    size_t chunk_size = (N + W - 1) / W;

    for(size_t w = 0; w < W; w++) {
      size_t curr_b = w * chunk_size;
      if(curr_b >= N) {
        break;
      }
      size_t curr_e = (std::min)(curr_b + chunk_size, N);
      if(w + 1 == W) {
        merge_chunk(curr_b, curr_e);  // last worker runs inline
      } else {
        rt.silent_async([=]() { merge_chunk(curr_b, curr_e); });
      }
    }
  };
}

// ----------------------------------------------------------------------------
// tf::Taskflow::merge
// ----------------------------------------------------------------------------

/**
@private
*/
template <typename B1, typename E1, typename B2, typename E2, typename O>
Task FlowBuilder::merge(B1 first1, E1 last1, B2 first2, E2 last2, O d_first) {
  return emplace(make_merge_task(
    first1, last1, first2, last2, std::less<>{}, d_first
  ));
}

/**
@private
*/
template <typename B1, typename E1, typename B2, typename E2,
          typename O, typename C>
Task FlowBuilder::merge(B1 first1, E1 last1, B2 first2, E2 last2, O d_first, C cmp) {
  return emplace(make_merge_task(
    first1, last1, first2, last2, cmp, d_first
  ));
}

}  // namespace tf
