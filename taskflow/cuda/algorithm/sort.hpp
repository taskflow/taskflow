#pragma once

#include "merge.hpp"

/**
@file taskflow/cuda/algorithm/sort.hpp
@brief CUDA sort algorithm include file
*/

namespace tf::detail {

// ----------------------------------------------------------------------------
// odd-even sort in register
// ----------------------------------------------------------------------------

/**
@private
@brief counts the number of leading zeros starting from the most significant bit
*/
constexpr int cuda_clz(int x) {
  for(int i = 31; i >= 0; --i) {
    if((1<< i) & x) {
      return 31 - i;
    }
  }
  return 32;
}

/**
@private
@brief finds log2(x) and optionally round up to the next integer logarithm.
*/
constexpr int cuda_find_log2(int x, bool round_up = false) {
  int a = 31 - cuda_clz(x);
  if(round_up) {
    a += !is_pow2(x);
  }
  return a;
}

/** @private */
template<typename T, unsigned vt, typename C>
__device__ auto cuda_odd_even_sort(
  cudaArray<T, vt> x, C comp, int flags = 0
) {
  cuda_iterate<vt>([&](auto I) {
    #pragma unroll
    for(auto i = 1 & I; i < vt - 1; i += 2) {
      if((0 == ((2<< i) & flags)) && comp(x[i + 1], x[i]))
        cuda_swap(x[i], x[i + 1]);
    }
  });
  return x;
}

/** @private */
template<typename K, typename V, unsigned vt, typename C>
__device__ auto cuda_odd_even_sort(
  cudaKVArray<K, V, vt> x, C comp, int flags = 0
) {
  cuda_iterate<vt>([&](auto I) {
    #pragma unroll
    for(auto i = 1 & I; i < vt - 1; i += 2) {
      if((0 == ((2<< i) & flags)) && comp(x.keys[i + 1], x.keys[i])) {
        cuda_swap(x.keys[i], x.keys[i + 1]);
        cuda_swap(x.vals[i], x.vals[i + 1]);
      }
    }
  });
  return x;
}

// ----------------------------------------------------------------------------
// range check
// ----------------------------------------------------------------------------

/** @private */
__device__ inline int cuda_out_of_range_flags(int first, int vt, int count) {
  int out_of_range = min(vt, first + vt - count);
  int head_flags = 0;
  if(out_of_range > 0) {
    const int mask = (1<< vt) - 1;
    head_flags = mask & (~mask>> out_of_range);
  }
  return head_flags;
}

/** @private */
__device__ inline auto cuda_compute_merge_sort_frame(
  unsigned partition, unsigned coop, unsigned spacing
) {

  unsigned size = spacing * (coop / 2);
  unsigned start = ~(coop - 1) & partition;
  unsigned a_begin = spacing * start;
  unsigned b_begin = spacing * start + size;

  return cudaMergeRange {
    a_begin,
    a_begin + size,
    b_begin,
    b_begin + size
  };
}

/** @private */
__device__ inline auto cuda_compute_merge_sort_range(
  unsigned count, unsigned partition, unsigned coop, unsigned spacing
) {

  auto frame = cuda_compute_merge_sort_frame(partition, coop, spacing);

  return cudaMergeRange {
    frame.a_begin,
    min(count, frame.a_end),
    min(count, frame.b_begin),
    min(count, frame.b_end)
  };
}

/** @private */
__device__ inline auto cuda_compute_merge_sort_range(
  unsigned count, unsigned partition, unsigned coop, unsigned spacing,
  unsigned mp0, unsigned mp1
) {

  auto range = cuda_compute_merge_sort_range(count, partition, coop, spacing);

  // Locate the diagonal from the start of the A sublist.
  unsigned diag = spacing * partition - range.a_begin;

  // The end partition of the last cta for each merge operation is computed
  // and stored as the begin partition for the subsequent merge. i.e. it is
  // the same partition but in the wrong coordinate system, so its 0 when it
  // should be listSize. Correct that by checking if this is the last cta
  // in this merge operation.
  if(coop - 1 != ((coop - 1) & partition)) {
    range.a_end = range.a_begin + mp1;
    range.b_end = min(count, range.b_begin + diag + spacing - mp1);
  }

  range.a_begin = range.a_begin + mp0;
  range.b_begin = min(count, range.b_begin + diag - mp0);

  return range;
}

/** @private */
template<unsigned nt, unsigned vt, typename K, typename V>
struct cudaBlockSort {

  static constexpr bool has_values = !std::is_same<V, cudaEmpty>::value;
  static constexpr unsigned num_passes = log2(nt);

  /** @private */
  union Storage {
    K keys[nt * vt + 1];
    V vals[nt * vt];
  };

  static_assert(is_pow2(nt), "cudaBlockSort requires pow2 number of threads");

  template<typename C>
  __device__ auto merge_pass(
    cudaKVArray<K, V, vt> x,
    unsigned tid, unsigned count, unsigned pass,
    C comp, Storage& storage
  ) const {

    // Divide the CTA's keys into lists.
    unsigned coop = 2 << pass;
    auto range = cuda_compute_merge_sort_range(count, tid, coop, vt);
    unsigned diag = vt * tid - range.a_begin;

    // Store the keys into shared memory for searching.
    cuda_reg_to_shared_thread<nt, vt>(x.keys, tid, storage.keys);

    // Search for the merge path for this thread within its list.
    auto mp = cuda_merge_path<cudaMergeBoundType::LOWER>(
      storage.keys, range, diag, comp
    );

    // Run a serial merge and return.
    auto merge = cuda_serial_merge<cudaMergeBoundType::LOWER, vt>(
      storage.keys, range.partition(mp, diag), comp
    );
    x.keys = merge.keys;

    if(has_values) {
      // Reorder values through shared memory.
      cuda_reg_to_shared_thread<nt, vt>(x.vals, tid, storage.vals);
      x.vals = cuda_shared_gather<nt, vt>(storage.vals, merge.indices);
    }

    return x;
  }

  template<typename C>
  __device__ auto block_sort(cudaKVArray<K, V, vt> x,
    unsigned tid, unsigned count, C comp, Storage& storage
  ) const {

    // Sort the inputs within each thread. If any threads have fewer than
    // vt items, use the segmented sort network to prevent out-of-range
    // elements from contaminating the sort.
    if(count < nt * vt) {
      auto head_flags = cuda_out_of_range_flags(vt * tid, vt, count);
      x = cuda_odd_even_sort(x, comp, head_flags);
    } else {
      x = cuda_odd_even_sort(x, comp);
    }

    // Merge threads starting with a pair until all values are merged.
    for(unsigned pass = 0; pass < num_passes; ++pass) {
      x = merge_pass(x, tid, count, pass, comp, storage);
    }

    return x;
  }
};

/** @private */
template<typename P, typename K, typename C>
void cuda_merge_sort_partitions(
  P&& p, K keys, unsigned count,
  unsigned coop, unsigned spacing, C comp, unsigned* buf
) {

  // bufer size is num_partitions + 1
  unsigned num_partitions = (count + spacing - 1) / spacing + 1;

  const unsigned nt = 128;
  const unsigned vt = 1;
  const unsigned nv = nt * vt;

  unsigned B = (num_partitions + nv - 1) / nv;  // nt = 128, vt = 1

  cuda_kernel<<<B, nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
    auto range = cuda_get_tile(bid, nt * vt, num_partitions);
    cuda_strided_iterate<nt, vt>([=](auto, auto j) {
      auto index = j + range.begin;
      auto range = cuda_compute_merge_sort_range(count, index, coop, spacing);
      auto diag = min(spacing * index, count) - range.a_begin;
      buf[index] = cuda_merge_path<cudaMergeBoundType::LOWER>(
        keys + range.a_begin, range.a_count(),
        keys + range.b_begin, range.b_count(),
        diag, comp
      );
    }, tid, range.count());
  });
}

/** @private */
template<typename P, typename K_it, typename V_it, typename C>
void merge_sort_loop(
  P&& p, K_it keys_input, V_it vals_input, unsigned count, C comp, void* buf
) {

  using K = typename std::iterator_traits<K_it>::value_type;
  using V = typename std::iterator_traits<V_it>::value_type;
  using E = std::decay_t<P>;

  const bool has_values = !std::is_same<V, cudaEmpty>::value;

  unsigned B = (count + E::nv - 1) / E::nv;
  unsigned R = cuda_find_log2(B, true);

  K* keys_output    {nullptr};
  V* vals_output    {nullptr};
  unsigned *mp_data {nullptr};

  if(R) {
    keys_output = (K*)(buf);
    if(has_values) {
      vals_output = (V*)(keys_output + count);
      mp_data = (unsigned*)(vals_output + count);
    }
    else {
      mp_data = (unsigned*)(keys_output + count);
    }
  }

  //cudaDeviceVector<K> keys_temp(R ? count : 0);
  //auto keys_output = keys_temp.data();
  ////std::cout << "keys_output = " << keys_temp.size()*sizeof(K) << std::endl;

  //cudaDeviceVector<V> vals_temp((has_values && R) ? count : 0);
  //auto vals_output = vals_temp.data();
  //std::cout << "vals_output = " << vals_temp.size()*sizeof(V) << std::endl;

  auto keys_blocksort = (1 & R) ? keys_output : keys_input;
  auto vals_blocksort = (1 & R) ? vals_output : vals_input;

  //printf("B=%u, R=%u\n", B, R);

  cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {

    using sort_t = cudaBlockSort<E::nt, E::vt, K, V>;

    __shared__ union {
      typename sort_t::Storage sort;
      K keys[E::nv];
      V vals[E::nv];
    } shared;

    auto tile = cuda_get_tile(bid, E::nv, count);

    // Load the keys and values.
    cudaKVArray<K, V, E::vt> unsorted;
    unsorted.keys = cuda_mem_to_reg_thread<E::nt, E::vt>(
      keys_input + tile.begin, tid, tile.count(), shared.keys
    );

    if(has_values) {
      unsorted.vals = cuda_mem_to_reg_thread<E::nt, E::vt>(
        vals_input + tile.begin, tid, tile.count(), shared.vals
      );
    }

    // Blocksort.
    auto sorted = sort_t().block_sort(unsorted, tid, tile.count(), comp, shared.sort);

    // Store the keys and values.
    cuda_reg_to_mem_thread<E::nt, E::vt>(
      sorted.keys, tid, tile.count(), keys_blocksort + tile.begin, shared.keys
    );

    if(has_values) {
      cuda_reg_to_mem_thread<E::nt, E::vt>(
        sorted.vals, tid, tile.count(), vals_blocksort + tile.begin, shared.vals
      );
    }
  });

  // merge passes

  if(1 & R) {
    std::swap(keys_input, keys_output);
    std::swap(vals_input, vals_output);
  }

  // number of partitions
  //unsigned num_partitions = B + 1;
  //cudaDeviceVector<unsigned> mem(num_partitions);
  //auto mp_data = mem.data();
  //std::cout << "num_partitions = " << (B+1)*sizeof(unsigned) << std::endl;

  for(unsigned pass = 0; pass < R; ++pass) {

    unsigned coop = 2 << pass;

    cuda_merge_sort_partitions(
      p, keys_input, count, coop, E::nv, comp, mp_data
    );

    cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=]__device__(auto tid, auto bid) {

      __shared__ union {
        K keys[E::nv + 1];
        unsigned indices[E::nv];
      } shared;

      auto tile = cuda_get_tile(bid, E::nv, count);

      // Load the range for this CTA and merge the values into register.
      auto range = cuda_compute_merge_sort_range(
        count, bid, coop, E::nv, mp_data[bid + 0], mp_data[bid + 1]
      );

      auto merge = block_merge_from_mem<cudaMergeBoundType::LOWER, E::nt, E::vt>(
        keys_input, keys_input, range, tid, comp, shared.keys
      );

      // Store merged values back out.
      cuda_reg_to_mem_thread<E::nt>(
        merge.keys, tid, tile.count(), keys_output + tile.begin, shared.keys
      );

      if(has_values) {
        // Transpose the indices from thread order to strided order.
        auto indices = cuda_reg_thread_to_strided<E::nt>(
          merge.indices, tid, shared.indices
        );

        // Gather the input values and merge into the output values.
        cuda_transfer_two_streams_strided<E::nt>(
          vals_input + range.a_begin, range.a_count(),
          vals_input + range.b_begin, range.b_count(),
          indices, tid, vals_output + tile.begin
        );
      }
    });

    std::swap(keys_input, keys_output);
    std::swap(vals_input, vals_output);
  }
}

}  // end of namespace tf::detail ---------------------------------------------

namespace tf {

/**
@brief queries the buffer size in bytes needed to call sort kernels
       for the given number of elements

@tparam P execution policy type
@tparam K key type
@tparam V value type (default tf::cudaEmpty)

@param count number of keys/values to sort

The function is used to allocate a buffer for calling tf::cuda_sort.

*/
template <typename P, typename K, typename V = cudaEmpty>
unsigned cuda_sort_buffer_size(unsigned count) {

  using E = std::decay_t<P>;

  const bool has_values = !std::is_same<V, cudaEmpty>::value;

  unsigned B = (count + E::nv - 1) / E::nv;
  unsigned R = detail::cuda_find_log2(B, true);

  return R ? (count * sizeof(K) + (has_values ? count*sizeof(V) : 0) +
             (B+1)*sizeof(unsigned)) : 0;
}

// ----------------------------------------------------------------------------
// key-value sort
// ----------------------------------------------------------------------------

/**
@brief performs asynchronous key-value sort on a range of items

@tparam P execution policy type
@tparam K_it key iterator type
@tparam V_it value iterator type
@tparam C comparator type

@param p execution policy
@param k_first iterator to the beginning of the key range
@param k_last iterator to the end of the key range
@param v_first iterator to the beginning of the value range
@param comp binary comparator
@param buf pointer to the temporary buffer

Sorts key-value elements in <tt>[k_first, k_last)</tt> and
<tt>[v_first, v_first + (k_last - k_first))</tt> into ascending key order
using the given comparator @c comp.
If @c i and @c j are any two valid iterators in <tt>[k_first, k_last)</tt>
such that @c i precedes @c j, and @c p and @c q are iterators in
<tt>[v_first, v_first + (k_last - k_first))</tt> corresponding to
@c i and @c j respectively, then <tt>comp(*j, *i)</tt> evaluates to @c false.

For example, assume:
  + @c keys are <tt>{1, 4, 2, 8, 5, 7}</tt>
  + @c values are <tt>{'a', 'b', 'c', 'd', 'e', 'f'}</tt>

After sort:
  + @c keys are <tt>{1, 2, 4, 5, 7, 8}</tt>
  + @c values are <tt>{'a', 'c', 'b', 'e', 'f', 'd'}</tt>

*/
template<typename P, typename K_it, typename V_it, typename C>
void cuda_sort_by_key(
  P&& p, K_it k_first, K_it k_last, V_it v_first, C comp, void* buf
) {

  unsigned N = std::distance(k_first, k_last);

  if(N <= 1) {
    return;
  }

  detail::merge_sort_loop(p, k_first, v_first, N, comp, buf);
}

// ----------------------------------------------------------------------------
// key sort
// ----------------------------------------------------------------------------

/**
@brief performs asynchronous key-only sort on a range of items

@tparam P execution policy type
@tparam K_it key iterator type
@tparam C comparator type

@param p execution policy
@param k_first iterator to the beginning of the key range
@param k_last iterator to the end of the key range
@param comp binary comparator
@param buf pointer to the temporary buffer

This method is equivalent to tf::cuda_sort_by_key without values.

*/
template<typename P, typename K_it, typename C>
void cuda_sort(P&& p, K_it k_first, K_it k_last, C comp, void* buf) {
  cuda_sort_by_key(p, k_first, k_last, (cudaEmpty*)nullptr, comp, buf);
}

}  // end of namespace tf -----------------------------------------------------

