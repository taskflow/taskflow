#pragma once

#include "../core/async.hpp"

namespace tf::detail {

// threshold whether or not to perform parallel sort
template <typename I>
constexpr size_t parallel_sort_cutoff() {

  //using value_type = std::decay_t<decltype(*std::declval<I>())>;
  using value_type = typename std::iterator_traits<I>::value_type;

  constexpr size_t object_size = sizeof(value_type);

  if constexpr(std::is_same_v<value_type, std::string>) {
    return 65536 / sizeof(std::string);
  }
  else {
    if constexpr(object_size < 16) return 4096;
    else if constexpr(object_size < 32) return 2048;
    else if constexpr(object_size < 64) return 1024;
    else if constexpr(object_size < 128) return 768;
    else if constexpr(object_size < 256) return 512;
    else if constexpr(object_size < 512) return 256;
    else return 128;
  }
}

// ----------------------------------------------------------------------------
// pattern-defeating quick sort (pdqsort)
// https://github.com/orlp/pdqsort/
// ----------------------------------------------------------------------------

template<typename T, size_t cacheline_size=64>
inline T* align_cacheline(T* p) {
#if defined(UINTPTR_MAX) && __cplusplus >= 201103L
  std::uintptr_t ip = reinterpret_cast<std::uintptr_t>(p);
#else
  std::size_t ip = reinterpret_cast<std::size_t>(p);
#endif
  ip = (ip + cacheline_size - 1) & -cacheline_size;
  return reinterpret_cast<T*>(ip);
}

template<typename Iter>
inline void swap_offsets(
  Iter first, Iter last,
  unsigned char* offsets_l, unsigned char* offsets_r,
  size_t num, bool use_swaps
) {
  typedef typename std::iterator_traits<Iter>::value_type T;
  if (use_swaps) {
    // This case is needed for the descending distribution, where we need
    // to have proper swapping for pdqsort to remain O(n).
    for (size_t i = 0; i < num; ++i) {
        std::iter_swap(first + offsets_l[i], last - offsets_r[i]);
    }
  } else if (num > 0) {
    Iter l = first + offsets_l[0]; Iter r = last - offsets_r[0];
    T tmp(std::move(*l)); *l = std::move(*r);
    for (size_t i = 1; i < num; ++i) {
        l = first + offsets_l[i]; *r = std::move(*l);
        r = last - offsets_r[i]; *l = std::move(*r);
    }
    *r = std::move(tmp);
  }
}

// Sorts [begin, end) using insertion sort with the given comparison function.
template<typename RandItr, typename Compare>
void insertion_sort(RandItr begin, RandItr end, Compare comp) {

  using T = typename std::iterator_traits<RandItr>::value_type;

  if (begin == end) {
    return;
  }

  for (RandItr cur = begin + 1; cur != end; ++cur) {

    RandItr shift = cur;
    RandItr shift_1 = cur - 1;

    // Compare first to avoid 2 moves for an element
    // already positioned correctly.
    if (comp(*shift, *shift_1)) {
      T tmp = std::move(*shift);
      do {
        *shift-- = std::move(*shift_1);
      }while (shift != begin && comp(tmp, *--shift_1));
      *shift = std::move(tmp);
    }
  }
}

// Sorts [begin, end) using insertion sort with the given comparison function.
// Assumes *(begin - 1) is an element smaller than or equal to any element
// in [begin, end).
template<typename RandItr, typename Compare>
void unguarded_insertion_sort(RandItr begin, RandItr end, Compare comp) {

  using T = typename std::iterator_traits<RandItr>::value_type;

  if (begin == end) {
    return;
  }

  for (RandItr cur = begin + 1; cur != end; ++cur) {
    RandItr shift = cur;
    RandItr shift_1 = cur - 1;

    // Compare first so we can avoid 2 moves
    // for an element already positioned correctly.
    if (comp(*shift, *shift_1)) {
      T tmp = std::move(*shift);

      do {
        *shift-- = std::move(*shift_1);
      }while (comp(tmp, *--shift_1));

      *shift = std::move(tmp);
    }
  }
}

// Attempts to use insertion sort on [begin, end).
// Will return false if more than
// partial_insertion_sort_limit elements were moved,
// and abort sorting. Otherwise it will successfully sort and return true.
template<typename RandItr, typename Compare>
bool partial_insertion_sort(RandItr begin, RandItr end, Compare comp) {

  using T = typename std::iterator_traits<RandItr>::value_type;
  using D = typename std::iterator_traits<RandItr>::difference_type;

  // When we detect an already sorted partition, attempt an insertion sort
  // that allows this amount of element moves before giving up.
  constexpr auto partial_insertion_sort_limit = D{8};

  if (begin == end) return true;

  auto limit = D{0};

  for (RandItr cur = begin + 1; cur != end; ++cur) {

    if (limit > partial_insertion_sort_limit) {
      return false;
    }

    RandItr shift = cur;
    RandItr shift_1 = cur - 1;

    // Compare first so we can avoid 2 moves
    // for an element already positioned correctly.
    if (comp(*shift, *shift_1)) {
      T tmp = std::move(*shift);

      do {
        *shift-- = std::move(*shift_1);
      }while (shift != begin && comp(tmp, *--shift_1));

      *shift = std::move(tmp);
      limit += cur - shift;
    }
  }

  return true;
}

// Partitions [begin, end) around pivot *begin using comparison function comp. Elements equal
// to the pivot are put in the right-hand partition. Returns the position of the pivot after
// partitioning and whether the passed sequence already was correctly partitioned. Assumes the
// pivot is a median of at least 3 elements and that [begin, end) is at least
// insertion_sort_threshold long. Uses branchless partitioning.
template<typename Iter, typename Compare>
std::pair<Iter, bool> partition_right_branchless(Iter begin, Iter end, Compare comp) {

  typedef typename std::iterator_traits<Iter>::value_type T;

  constexpr size_t block_size = 64;
  constexpr size_t cacheline_size = 64;

  // Move pivot into local for speed.
  T pivot(std::move(*begin));
  Iter first = begin;
  Iter last = end;

  // Find the first element greater than or equal than the pivot (the median of 3 guarantees
  // this exists).
  while (comp(*++first, pivot));

  // Find the first element strictly smaller than the pivot. We have to guard this search if
  // there was no element before *first.
  if (first - 1 == begin) while (first < last && !comp(*--last, pivot));
  else                    while (                !comp(*--last, pivot));

  // If the first pair of elements that should be swapped to partition are the same element,
  // the passed in sequence already was correctly partitioned.
  bool already_partitioned = first >= last;
  if (!already_partitioned) {
    std::iter_swap(first, last);
    ++first;

    // The following branchless partitioning is derived from "BlockQuicksort: How Branch
    // Mispredictions don't affect Quicksort" by Stefan Edelkamp and Armin Weiss, but
    // heavily micro-optimized.
    unsigned char offsets_l_storage[block_size + cacheline_size];
    unsigned char offsets_r_storage[block_size + cacheline_size];
    unsigned char* offsets_l = align_cacheline(offsets_l_storage);
    unsigned char* offsets_r = align_cacheline(offsets_r_storage);

    Iter offsets_l_base = first;
    Iter offsets_r_base = last;
    size_t num_l, num_r, start_l, start_r;
    num_l = num_r = start_l = start_r = 0;

    while (first < last) {
      // Fill up offset blocks with elements that are on the wrong side.
      // First we determine how much elements are considered for each offset block.
      size_t num_unknown = last - first;
      size_t left_split = num_l == 0 ? (num_r == 0 ? num_unknown / 2 : num_unknown) : 0;
      size_t right_split = num_r == 0 ? (num_unknown - left_split) : 0;

      // Fill the offset blocks.
      if (left_split >= block_size) {
        for (size_t i = 0; i < block_size;) {
          offsets_l[num_l] = i++; num_l += !comp(*first, pivot); ++first;
          offsets_l[num_l] = i++; num_l += !comp(*first, pivot); ++first;
          offsets_l[num_l] = i++; num_l += !comp(*first, pivot); ++first;
          offsets_l[num_l] = i++; num_l += !comp(*first, pivot); ++first;
          offsets_l[num_l] = i++; num_l += !comp(*first, pivot); ++first;
          offsets_l[num_l] = i++; num_l += !comp(*first, pivot); ++first;
          offsets_l[num_l] = i++; num_l += !comp(*first, pivot); ++first;
          offsets_l[num_l] = i++; num_l += !comp(*first, pivot); ++first;
        }
      } else {
        for (size_t i = 0; i < left_split;) {
          offsets_l[num_l] = i++; num_l += !comp(*first, pivot); ++first;
        }
      }

      if (right_split >= block_size) {
        for (size_t i = 0; i < block_size;) {
          offsets_r[num_r] = ++i; num_r += comp(*--last, pivot);
          offsets_r[num_r] = ++i; num_r += comp(*--last, pivot);
          offsets_r[num_r] = ++i; num_r += comp(*--last, pivot);
          offsets_r[num_r] = ++i; num_r += comp(*--last, pivot);
          offsets_r[num_r] = ++i; num_r += comp(*--last, pivot);
          offsets_r[num_r] = ++i; num_r += comp(*--last, pivot);
          offsets_r[num_r] = ++i; num_r += comp(*--last, pivot);
          offsets_r[num_r] = ++i; num_r += comp(*--last, pivot);
        }
      } else {
        for (size_t i = 0; i < right_split;) {
          offsets_r[num_r] = ++i; num_r += comp(*--last, pivot);
        }
      }

      // Swap elements and update block sizes and first/last boundaries.
      size_t num = std::min(num_l, num_r);
      swap_offsets(
        offsets_l_base, offsets_r_base, 
        offsets_l + start_l, offsets_r + start_r,
        num, num_l == num_r
      );
      num_l -= num; num_r -= num;
      start_l += num; start_r += num;

      if (num_l == 0) {
        start_l = 0;
        offsets_l_base = first;
      }

      if (num_r == 0) {
        start_r = 0;
        offsets_r_base = last;
      }
    }

    // We have now fully identified [first, last)'s proper position. Swap the last elements.
    if (num_l) {
      offsets_l += start_l;
      while (num_l--) std::iter_swap(offsets_l_base + offsets_l[num_l], --last);
      first = last;
    }
    if (num_r) {
      offsets_r += start_r;
      while (num_r--) std::iter_swap(offsets_r_base - offsets_r[num_r], first), ++first;
      last = first;
    }
  }

  // Put the pivot in the right place.
  Iter pivot_pos = first - 1;
  *begin = std::move(*pivot_pos);
  *pivot_pos = std::move(pivot);

  return std::make_pair(pivot_pos, already_partitioned);
}

// Partitions [begin, end) around pivot *begin using comparison function comp.
// Elements equal to the pivot are put in the right-hand partition.
// Returns the position of the pivot after partitioning and whether the passed
// sequence already was correctly partitioned.
// Assumes the pivot is a median of at least 3 elements and that [begin, end)
// is at least insertion_sort_threshold long.
template<typename Iter, typename Compare>
std::pair<Iter, bool> partition_right(Iter begin, Iter end, Compare comp) {

  using T = typename std::iterator_traits<Iter>::value_type;

  // Move pivot into local for speed.
  T pivot(std::move(*begin));

  Iter first = begin;
  Iter last = end;

  // Find the first element greater than or equal than the pivot
  // (the median of 3 guarantees/ this exists).
  while (comp(*++first, pivot));

  // Find the first element strictly smaller than the pivot.
  // We have to guard this search if there was no element before *first.
  if (first - 1 == begin) while (first < last && !comp(*--last, pivot));
  else while (!comp(*--last, pivot));

  // If the first pair of elements that should be swapped to partition
  // are the same element, the passed in sequence already was correctly
  // partitioned.
  bool already_partitioned = first >= last;

  // Keep swapping pairs of elements that are on the wrong side of the pivot.
  // Previously swapped pairs guard the searches,
  // which is why the first iteration is special-cased above.
  while (first < last) {
    std::iter_swap(first, last);
    while (comp(*++first, pivot));
    while (!comp(*--last, pivot));
  }

  // Put the pivot in the right place.
  Iter pivot_pos = first - 1;
  *begin = std::move(*pivot_pos);
  *pivot_pos = std::move(pivot);

  return std::make_pair(pivot_pos, already_partitioned);
}

// Similar function to the one above, except elements equal to the pivot
// are put to the left of the pivot and it doesn't check or return
// if the passed sequence already was partitioned.
// Since this is rarely used (the many equal case),
// and in that case pdqsort already has O(n) performance,
// no block quicksort is applied here for simplicity.
template<typename RandItr, typename Compare>
RandItr partition_left(RandItr begin, RandItr end, Compare comp) {

  using T = typename std::iterator_traits<RandItr>::value_type;

  T pivot(std::move(*begin));

  RandItr first = begin;
  RandItr last = end;

  while (comp(pivot, *--last));

  if (last + 1 == end) {
    while (first < last && !comp(pivot, *++first));
  }
  else {
    while (!comp(pivot, *++first));
  }

  while (first < last) {
    std::iter_swap(first, last);
    while (comp(pivot, *--last));
    while (!comp(pivot, *++first));
  }

  RandItr pivot_pos = last;
  *begin = std::move(*pivot_pos);
  *pivot_pos = std::move(pivot);

  return pivot_pos;
}

template<typename Iter, typename Compare, bool Branchless>
void parallel_pdqsort(
  tf::Runtime& rt,
  Iter begin, Iter end, Compare comp,
  int bad_allowed, bool leftmost = true
) {

  // Partitions below this size are sorted sequentially
  constexpr auto cutoff = parallel_sort_cutoff<Iter>();

  // Partitions below this size are sorted using insertion sort
  constexpr auto insertion_sort_threshold = 24;

  // Partitions above this size use Tukey's ninther to select the pivot.
  constexpr auto ninther_threshold = 128;

  //using diff_t = typename std::iterator_traits<Iter>::difference_type;

  // Use a while loop for tail recursion elimination.
  while (true) {

    //diff_t size = end - begin;
    size_t size = end - begin;

    // Insertion sort is faster for small arrays.
    if (size < insertion_sort_threshold) {
      if (leftmost) {
        insertion_sort(begin, end, comp);
      }
      else {
        unguarded_insertion_sort(begin, end, comp);
      }
      return;
    }

    if(size <= cutoff) {
      std::sort(begin, end, comp);
      return;
    }

    // Choose pivot as median of 3 or pseudomedian of 9.
    //diff_t s2 = size / 2;
    size_t s2 = size >> 1;
    if (size > ninther_threshold) {
      sort3(begin, begin + s2, end - 1, comp);
      sort3(begin + 1, begin + (s2 - 1), end - 2, comp);
      sort3(begin + 2, begin + (s2 + 1), end - 3, comp);
      sort3(begin + (s2 - 1), begin + s2, begin + (s2 + 1), comp);
      std::iter_swap(begin, begin + s2);
    }
    else {
      sort3(begin + s2, begin, end - 1, comp);
    }

    // If *(begin - 1) is the end of the right partition
    // of a previous partition operation, there is no element in [begin, end)
    // that is smaller than *(begin - 1).
    // Then if our pivot compares equal to *(begin - 1) we change strategy,
    // putting equal elements in the left partition,
    // greater elements in the right partition.
    // We do not have to recurse on the left partition,
    // since it's sorted (all equal).
    if (!leftmost && !comp(*(begin - 1), *begin)) {
      begin = partition_left(begin, end, comp) + 1;
      continue;
    }

    // Partition and get results.
    const auto pair = Branchless ? partition_right_branchless(begin, end, comp) :
                                   partition_right(begin, end, comp);
       
    const auto pivot_pos = pair.first;
    const auto already_partitioned = pair.second;

    // Check for a highly unbalanced partition.
    //diff_t l_size = pivot_pos - begin;
    //diff_t r_size = end - (pivot_pos + 1);
    const size_t l_size = pivot_pos - begin;
    const size_t r_size = end - (pivot_pos + 1);
    const bool highly_unbalanced = l_size < size / 8 || r_size < size / 8;

    // If we got a highly unbalanced partition we shuffle elements
    // to break many patterns.
    if (highly_unbalanced) {
      // If we had too many bad partitions, switch to heapsort
      // to guarantee O(n log n).
      if (--bad_allowed == 0) {
        std::make_heap(begin, end, comp);
        std::sort_heap(begin, end, comp);
        return;
      }

      if (l_size >= insertion_sort_threshold) {
        std::iter_swap(begin, begin + l_size / 4);
        std::iter_swap(pivot_pos - 1, pivot_pos - l_size / 4);
        if (l_size > ninther_threshold) {
          std::iter_swap(begin + 1, begin + (l_size / 4 + 1));
          std::iter_swap(begin + 2, begin + (l_size / 4 + 2));
          std::iter_swap(pivot_pos - 2, pivot_pos - (l_size / 4 + 1));
          std::iter_swap(pivot_pos - 3, pivot_pos - (l_size / 4 + 2));
        }
      }

      if (r_size >= insertion_sort_threshold) {
        std::iter_swap(pivot_pos + 1, pivot_pos + (1 + r_size / 4));
        std::iter_swap(end - 1,                   end - r_size / 4);
        if (r_size > ninther_threshold) {
          std::iter_swap(pivot_pos + 2, pivot_pos + (2 + r_size / 4));
          std::iter_swap(pivot_pos + 3, pivot_pos + (3 + r_size / 4));
          std::iter_swap(end - 2,             end - (1 + r_size / 4));
          std::iter_swap(end - 3,             end - (2 + r_size / 4));
        }
      }
    }
    // decently balanced
    else {
      // sequence try to use insertion sort.
      if (already_partitioned &&
          partial_insertion_sort(begin, pivot_pos, comp) &&
          partial_insertion_sort(pivot_pos + 1, end, comp)
      ) {
        return;
      }
    }

    // Sort the left partition first using recursion and
    // do tail recursion elimination for the right-hand partition.
    rt.silent_async(
      [&rt, begin, pivot_pos, comp, bad_allowed, leftmost] () mutable {
        parallel_pdqsort<Iter, Compare, Branchless>(
          rt, begin, pivot_pos, comp, bad_allowed, leftmost
        );
      }
    );
    begin = pivot_pos + 1;
    leftmost = false;
  }
}

// ----------------------------------------------------------------------------
// 3-way quick sort
// ----------------------------------------------------------------------------

// 3-way quick sort
template <typename RandItr, typename C>
void parallel_3wqsort(tf::Runtime& rt, RandItr first, RandItr last, C compare) {

  using namespace std::string_literals;

  constexpr auto cutoff = parallel_sort_cutoff<RandItr>();

  sort_partition:

  if(static_cast<size_t>(last - first) < cutoff) {
    std::sort(first, last+1, compare);
    return;
  }

  auto m = pseudo_median_of_nine(first, last, compare);

  if(m != first) {
    std::iter_swap(first, m);
  }

  auto l = first;
  auto r = last;
  auto f = std::next(first, 1);
  bool is_swapped_l = false;
  bool is_swapped_r = false;

  while(f <= r) {
    if(compare(*f, *l)) {
      is_swapped_l = true;
      std::iter_swap(l, f);
      l++;
      f++;
    }
    else if(compare(*l, *f)) {
      is_swapped_r = true;
      std::iter_swap(r, f);
      r--;
    }
    else {
      f++;
    }
  }

  if(l - first > 1 && is_swapped_l) {
    //rt.emplace([&](tf::Runtime& rtl) mutable {
    //  parallel_3wqsort(rtl, first, l-1, compare);
    //});
    rt.silent_async([&rt, first, l, &compare] () mutable {
      parallel_3wqsort(rt, first, l-1, compare);
    });
  }

  if(last - r > 1 && is_swapped_r) {
    //rt.emplace([&](tf::Runtime& rtr) mutable {
    //  parallel_3wqsort(rtr, r+1, last, compare);
    //});
    //rt.silent_async([&rt, r, last, &compare] () mutable {
    //  parallel_3wqsort(rt, r+1, last, compare);
    //});
    first = r+1;
    goto sort_partition;
  }

  //rt.join();
}

}  // end of namespace tf::detail ---------------------------------------------

namespace tf { 

// Function: make_sort_task
template <typename B, typename E, typename C>
TF_FORCE_INLINE auto make_sort_task(B b, E e, C cmp) {
  
  return [b, e, cmp] (Runtime& rt) mutable {

    using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
    using E_t = std::decay_t<unwrap_ref_decay_t<E>>;

    // fetch the iterator values
    B_t beg = b;
    E_t end = e;

    if(beg == end) {
      return;
    }

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= detail::parallel_sort_cutoff<B_t>()) {
      std::sort(beg, end, cmp);
      return;
    }

    //parallel_3wqsort(rt, beg, end-1, cmp);
    detail::parallel_pdqsort<B_t, C,
      is_std_compare_v<std::decay_t<C>> &&
      std::is_arithmetic_v<typename std::iterator_traits<B_t>::value_type>
    >(rt, beg, end, cmp, log2(end - beg));

    rt.corun_all();
  };
}
  
template <typename B, typename E>
TF_FORCE_INLINE auto make_sort_task(B beg, E end) {
  using value_type = std::decay_t<decltype(*std::declval<B>())>;
  return make_sort_task(beg, end, std::less<value_type>{});
}

// ----------------------------------------------------------------------------
// tf::Taskflow::sort
// ----------------------------------------------------------------------------

// Function: sort
template <typename B, typename E, typename C>
Task FlowBuilder::sort(B beg, E end, C cmp) {
  return emplace(make_sort_task(beg, end, cmp));
}

// Function: sort
template <typename B, typename E>
Task FlowBuilder::sort(B beg, E end) {
  return emplace(make_sort_task(beg, end));
}

}  // namespace tf ------------------------------------------------------------

