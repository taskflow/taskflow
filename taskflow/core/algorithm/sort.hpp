#pragma once

#include "../executor.hpp"

namespace tf {

// threshold whether or not to perform parallel sort
template <typename I>
constexpr size_t parallel_sort_cutoff() {

  //using value_type = std::decay_t<decltype(*std::declval<I>())>;
  using value_type = typename std::iterator_traits<I>::value_type;

  constexpr size_t object_size = sizeof(value_type);

  if constexpr(std::is_same_v<value_type, std::string>) {
    return 128;
  }
  else {
    if(object_size < 16) return 4096;
    else if(object_size < 32) return 2048;
    else if(object_size < 64) return 1024;
    else if(object_size < 128) return 768;
    else if(object_size < 256) return 512;
    else if(object_size < 512) return 256;
    else return 128;
  }
}

// ----------------------------------------------------------------------------
// pattern-defeating quick sort (pdqsort)
// ----------------------------------------------------------------------------

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

template<typename Iter, typename Compare>
void parallel_pdqsort(
  tf::Subflow& sf,
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
    
    if(size <= cutoff) {
      std::sort(begin, end, comp);
      return;
    }
    //// Insertion sort is faster for small arrays.
    //if (size < insertion_sort_threshold) {
    //  if (leftmost) {
    //    insertion_sort(begin, end, comp);
    //  }
    //  else {
    //    unguarded_insertion_sort(begin, end, comp);
    //  }
    //  return;
    //}

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
    auto pair = partition_right(begin, end, comp);
    auto pivot_pos = pair.first;
    auto already_partitioned = pair.second;

    // Check for a highly unbalanced partition.
    //diff_t l_size = pivot_pos - begin;
    //diff_t r_size = end - (pivot_pos + 1);
    size_t l_size = pivot_pos - begin;
    size_t r_size = end - (pivot_pos + 1);
    bool highly_unbalanced = l_size < size / 8 || r_size < size / 8;

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
    sf.silent_async(
      [&sf, begin, pivot_pos, &comp, bad_allowed, leftmost] () mutable {
        parallel_pdqsort(sf, begin, pivot_pos, comp, bad_allowed, leftmost);
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
void parallel_3wqsort(tf::Subflow& sf, RandItr first, RandItr last, C compare) {
  
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
    //sf.emplace([&](tf::Subflow& sfl) mutable {
    //  parallel_3wqsort(sfl, first, l-1, compare);
    //});
    sf.silent_async([&sf, first, l, &compare] () mutable {
      parallel_3wqsort(sf, first, l-1, compare);
    });
  }

  if(last - r > 1 && is_swapped_r) {
    //sf.emplace([&](tf::Subflow& sfr) mutable {
    //  parallel_3wqsort(sfr, r+1, last, compare);
    //});
    //sf.silent_async([&sf, r, last, &compare] () mutable {
    //  parallel_3wqsort(sf, r+1, last, compare);
    //});
    first = r+1;
    goto sort_partition;
  }

  //sf.join();
}

// ----------------------------------------------------------------------------
// tf::Taskflow::sort
// ----------------------------------------------------------------------------

// Function: sort
template <typename B, typename E, typename C>
Task FlowBuilder::sort(B&& beg, E&& end, C&& cmp) {
  
  using I = stateful_iterator_t<B, E>;

  Task task = emplace(
  [b=std::forward<B>(beg),
   e=std::forward<E>(end), 
   c=std::forward<C>(cmp)
   ] (Subflow& sf) mutable {
    
    // fetch the iterator values
    I beg = b;
    I end = e;
  
    if(beg == end) {
      return;
    }

    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= parallel_sort_cutoff<I>()) {
      std::sort(beg, end, c);
      return;
    }

    //parallel_3wqsort(sf, beg, end-1, c);
    parallel_pdqsort(sf, beg, end, c, log2(end - beg));

    sf.join();
  });  

  return task;
}

// Function: sort
template <typename B, typename E>
Task FlowBuilder::sort(B&& beg, E&& end) {
  
  using I = stateful_iterator_t<B, E>;
  //using value_type = std::decay_t<decltype(*std::declval<I>())>;
  using value_type = typename std::iterator_traits<I>::value_type;

  return sort(
    std::forward<B>(beg), std::forward<E>(end), std::less<value_type>{}
  );
}

}  // namespace tf ------------------------------------------------------------

