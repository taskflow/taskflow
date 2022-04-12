#pragma once

#include "../cudaflow.hpp"

/**
@file taskflow/cuda/algorithm/merge.hpp
@brief CUDA merge algorithm include file
*/

namespace tf::detail {

/**
@private
@brief merge bound type
*/
enum class cudaMergeBoundType {
  LOWER,
  UPPER
};

/** @private */
template<typename T, unsigned N>
struct cudaMergePair {
  cudaArray<T, N> keys;
  cudaArray<unsigned, N> indices;
};

/** @private */
struct cudaMergeRange {
  unsigned a_begin, a_end, b_begin, b_end;

  __device__ unsigned a_count() const { return a_end - a_begin; }
  __device__ unsigned b_count() const { return b_end - b_begin; }
  __device__ unsigned total() const { return a_count() + b_count(); }

  __device__ cudaRange a_range() const {
    return cudaRange { a_begin, a_end };
  }
  __device__ cudaRange b_range() const {
    return cudaRange { b_begin, b_end };
  }

  __device__ cudaMergeRange to_local() const {
    return cudaMergeRange { 0, a_count(), a_count(), total() };
  }

  // Partition from mp to the end.
  __device__ cudaMergeRange partition(unsigned mp0, unsigned diag) const {
    return cudaMergeRange { a_begin + mp0, a_end, b_begin + diag - mp0, b_end };
  }

  // Partition from mp0 to mp1.
  __device__ cudaMergeRange partition(unsigned mp0, unsigned diag0,
    unsigned mp1, unsigned diag1) const {
    return cudaMergeRange {
      a_begin + mp0,
      a_begin + mp1,
      b_begin + diag0 - mp0,
      b_begin + diag1 - mp1
    };
  }

  __device__ bool a_valid() const {
    return a_begin < a_end;
  }
  __device__ bool b_valid() const {
    return b_begin < b_end;
  }
};

/** @private */
template<
  cudaMergeBoundType bounds = cudaMergeBoundType::LOWER,
  typename a_keys_it, typename b_keys_it, typename comp_t
>
__device__ auto cuda_merge_path(
  a_keys_it a_keys, unsigned a_count,
  b_keys_it b_keys, unsigned b_count,
  unsigned diag, comp_t comp
) {

  unsigned beg = (diag > b_count) ? diag - b_count : 0;
  unsigned end = diag < a_count ? diag : a_count;

  while(beg < end) {
    auto mid = (beg + end) / 2;
    auto a_key = a_keys[mid];
    auto b_key = b_keys[diag - 1 - mid];
    bool pred = (cudaMergeBoundType::UPPER == bounds) ?
      comp(a_key, b_key) :
      !comp(b_key, a_key);

    if(pred) beg = mid + 1;
    else end = mid;
  }
  return beg;
}

/** @private */
template<cudaMergeBoundType bounds, typename keys_it, typename comp_t>
__device__ auto cuda_merge_path(
  keys_it keys, cudaMergeRange range, unsigned diag, comp_t comp
) {

  return cuda_merge_path<bounds>(
    keys + range.a_begin, range.a_count(),
    keys + range.b_begin, range.b_count(),
    diag, comp);
}

/** @private */
template<cudaMergeBoundType bounds, bool range_check, typename T, typename comp_t>
__device__ bool cuda_merge_predicate(
  T a_key, T b_key, cudaMergeRange range, comp_t comp
) {

  bool p;
  if(range_check && !range.a_valid()) {
    p = false;
  }
  else if(range_check && !range.b_valid()) {
    p = true;
  }
  else {
    p = (cudaMergeBoundType::UPPER == bounds) ? comp(a_key, b_key) :
                                               !comp(b_key, a_key);
  }
  return p;
}

/** @private */
inline __device__ auto cuda_compute_merge_range(
  unsigned a_count, unsigned b_count,
  unsigned partition, unsigned spacing,
  unsigned mp0, unsigned mp1
) {

  auto diag0 = spacing * partition;
  auto diag1 = min(a_count + b_count, diag0 + spacing);

  return cudaMergeRange { mp0, mp1, diag0 - mp0, diag1 - mp1 };
}

/**
@private

Specialization that emits just one LD instruction. Can only reliably used
with raw pointer types. Fixed not to use pointer arithmetic so that
we don't get undefined behaviors with unaligned types.
*/
template<unsigned nt, unsigned vt, typename T>
__device__ auto cuda_load_two_streams_reg(
  const T* a, unsigned a_count, const T* b, unsigned b_count, unsigned tid
) {

  b -= a_count;
  cudaArray<T, vt> x;
  cuda_strided_iterate<nt, vt>([&](auto i, auto index) {
    const T* p = (index >= a_count) ? b : a;
    x[i] = p[index];
  }, tid, a_count + b_count);

  return x;
}

/** @private */
template<unsigned nt, unsigned vt, typename T, typename a_it, typename b_it>
__device__
std::enable_if_t<
  !(std::is_pointer<a_it>::value && std::is_pointer<b_it>::value),
  cudaArray<T, vt>
> load_two_streams_reg(a_it a, unsigned a_count, b_it b, unsigned b_count, unsigned tid) {
  b -= a_count;
  cudaArray<T, vt> x;
  cuda_strided_iterate<nt, vt>([&](auto i, auto index) {
    x[i] = (index < a_count) ? a[index] : b[index];
  }, tid, a_count + b_count);
  return x;
}

/** @private */
template<unsigned nt, unsigned vt, typename A, typename B, typename T, unsigned S>
__device__ void cuda_load_two_streams_shared(A a, unsigned a_count,
  B b, unsigned b_count, unsigned tid, T (&shared)[S], bool sync = true
) {
  // Load into register then make an unconditional strided store into memory.
  auto x = cuda_load_two_streams_reg<nt, vt, T>(a, a_count, b, b_count, tid);
  cuda_reg_to_shared_strided<nt>(x, tid, shared, sync);
}

/** @private */
template<unsigned nt, unsigned vt, typename T>
__device__ auto cuda_gather_two_streams_strided(const T* a,
  unsigned a_count, const T* b, unsigned b_count, cudaArray<unsigned, vt> indices,
  unsigned tid) {

  ptrdiff_t b_offset = b - a - a_count;
  auto count = a_count + b_count;

  cudaArray<T, vt> x;
  cuda_strided_iterate<nt, vt>([&](auto i, auto j) {
    ptrdiff_t gather = indices[i];
    if(gather >= a_count) gather += b_offset;
    x[i] = a[gather];
  }, tid, count);

  return x;
}

/** @private */
template<unsigned nt, unsigned vt, typename T, typename a_it, typename b_it>
__device__
std::enable_if_t<
  !(std::is_pointer<a_it>::value && std::is_pointer<b_it>::value),
  cudaArray<T, vt>
> cuda_gather_two_streams_strided(a_it a,
  unsigned a_count, b_it b, unsigned b_count, cudaArray<unsigned, vt> indices, unsigned tid) {

  b -= a_count;
  cudaArray<T, vt> x;
  cuda_strided_iterate<nt, vt>([&](auto i, auto j) {
    x[i] = (indices[i] < a_count) ? a[indices[i]] : b[indices[i]];
  }, tid, a_count + b_count);

  return x;
}

/** @private */
template<unsigned nt, unsigned vt, typename a_it, typename b_it, typename c_it>
__device__ void cuda_transfer_two_streams_strided(
  a_it a, unsigned a_count, b_it b, unsigned b_count,
  cudaArray<unsigned, vt> indices, unsigned tid, c_it c
) {

  using T = typename std::iterator_traits<a_it>::value_type;
  auto x = cuda_gather_two_streams_strided<nt, vt, T>(
    a, a_count, b, b_count, indices, tid
  );

  cuda_reg_to_mem_strided<nt>(x, tid, a_count + b_count, c);
}


/**
@private

This function must be able to dereference keys[a_begin] and keys[b_begin],
no matter the indices for each. The caller should allocate at least
nt * vt + 1 elements for
*/
template<cudaMergeBoundType bounds, unsigned vt, typename T, typename comp_t>
__device__ auto cuda_serial_merge(
  const T* keys_shared, cudaMergeRange range, comp_t comp, bool sync = true
) {

  auto a_key = keys_shared[range.a_begin];
  auto b_key = keys_shared[range.b_begin];

  cudaMergePair<T, vt> merge_pair;
  cuda_iterate<vt>([&](auto i) {
    bool p = cuda_merge_predicate<bounds, true>(a_key, b_key, range, comp);
    auto index = p ? range.a_begin : range.b_begin;

    merge_pair.keys[i] = p ? a_key : b_key;
    merge_pair.indices[i] = index;

    T c_key = keys_shared[++index];
    if(p) a_key = c_key, range.a_begin = index;
    else b_key = c_key, range.b_begin = index;
  });

  if(sync) __syncthreads();
  return merge_pair;
}

/**
@private

Load arrays a and b from global memory and merge unsignedo register.
*/
template<cudaMergeBoundType bounds,
  unsigned nt, unsigned vt,
  typename a_it, typename b_it, typename T, typename comp_t, unsigned S
>
__device__ auto block_merge_from_mem(
  a_it a, b_it b, cudaMergeRange range_mem, unsigned tid, comp_t comp, T (&keys_shared)[S]
) {

  static_assert(S >= nt * vt + 1,
    "block_merge_from_mem requires temporary storage of at "
    "least nt * vt + 1 items");

  // Load the data into shared memory.
  cuda_load_two_streams_shared<nt, vt>(
    a + range_mem.a_begin, range_mem.a_count(),
    b + range_mem.b_begin, range_mem.b_count(),
    tid, keys_shared, true
  );

  // Run a merge path to find the start of the serial merge for each thread.
  auto range_local = range_mem.to_local();
  auto diag = vt * tid;
  auto mp = cuda_merge_path<bounds>(keys_shared, range_local, diag, comp);

  // Compute the ranges of the sources in shared memory. The end iterators
  // of the range are inaccurate, but still facilitate exact merging, because
  // only vt elements will be merged.
  auto merged = cuda_serial_merge<bounds, vt>(
    keys_shared, range_local.partition(mp, diag), comp
  );

  return merged;
};

/** @private */
template<cudaMergeBoundType bounds,
  typename P, typename a_keys_it, typename b_keys_it, typename comp_t
>
void cuda_merge_path_partitions(
  P&& p,
  a_keys_it a, unsigned a_count,
  b_keys_it b, unsigned b_count,
  unsigned spacing,
  comp_t comp,
  unsigned* buf
) {

  //int num_partitions = (int)div_up(a_count + b_count, spacing) + 1;

  unsigned num_partitions = (a_count + b_count + spacing - 1) / spacing + 1;

  const unsigned nt = 128;
  const unsigned vt = 1;
  const unsigned nv = nt * vt;

  unsigned B = (num_partitions + nv - 1) / nv;  // nt = 128, vt = 1

  cuda_kernel<<<B, nt, 0, p.stream()>>>([=]__device__(auto tid, auto bid) {
    auto range = cuda_get_tile(bid, nt * vt, num_partitions);
    cuda_strided_iterate<nt, vt>([=](auto, auto j) {
      auto index = range.begin + j;
      auto diag = min(spacing * index, a_count + b_count);
      buf[index] = cuda_merge_path<bounds>(a, a_count, b, b_count, diag, comp);
    }, tid, range.count());
  });
}

//template<typename segments_it>
//auto load_balance_partitions(int64_t dest_count, segments_it segments,
//  int num_segments, int spacing, context_t& context) ->
//  mem_t<typename std::iterator_traits<segments_it>::value_type> {
//
//  typedef typename std::iterator_traits<segments_it>::value_type int_t;
//  return merge_path_partitions<bounds_upper>(counting_iterator_t<int_t>(0),
//    dest_count, segments, num_segments, spacing, less_t<int_t>(), context);
//}

//template<bounds_t bounds, typename keys_it>
//mem_t<int> binary_search_partitions(keys_it keys, int count, int num_items,
//  int spacing, context_t& context) {
//
//  int num_partitions = div_up(count, spacing) + 1;
//  mem_t<int> mem(num_partitions, context);
//  int* p = mem.data();
//  transform([=]MGPU_DEVICE(int index) {
//    int key = min(spacing * index, count);
//    p[index] = binary_search<bounds>(keys, num_items, key, less_t<int>());
//  }, num_partitions, context);
//  return mem;
//}

/** @private */
template<
  typename P,
  typename a_keys_it, typename a_vals_it,
  typename b_keys_it, typename b_vals_it,
  typename c_keys_it, typename c_vals_it,
  typename comp_t
>
void cuda_merge_loop(
  P&& p,
  a_keys_it a_keys, a_vals_it a_vals, unsigned a_count,
  b_keys_it b_keys, b_vals_it b_vals, unsigned b_count,
  c_keys_it c_keys, c_vals_it c_vals,
  comp_t comp,
  void* ptr
) {

  using E = std::decay_t<P>;
  using T = typename std::iterator_traits<a_keys_it>::value_type;
  using V = typename std::iterator_traits<a_vals_it>::value_type;

  auto buf = static_cast<unsigned*>(ptr);

  auto has_values = !std::is_same<V, cudaEmpty>::value;

  cuda_merge_path_partitions<cudaMergeBoundType::LOWER>(
    p, a_keys, a_count, b_keys, b_count, E::nv, comp, buf
  );

  unsigned B = (a_count + b_count + E::nv - 1)/ E::nv;

  // we use small kernel
  cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {

    __shared__ union {
      T keys[E::nv + 1];
      unsigned indices[E::nv];
    } shared;

    // Load the range for this CTA and merge the values into register.
    auto mp0 = buf[bid + 0];
    auto mp1 = buf[bid + 1];
    auto range = cuda_compute_merge_range(a_count, b_count, bid, E::nv, mp0, mp1);

    auto merge = block_merge_from_mem<cudaMergeBoundType::LOWER, E::nt, E::vt>(
      a_keys, b_keys, range, tid, comp, shared.keys
    );

    auto dest_offset = E::nv * bid;
    cuda_reg_to_mem_thread<E::nt>(
      merge.keys, tid, range.total(), c_keys + dest_offset, shared.keys
    );

    if(has_values) {
      // Transpose the indices from thread order to strided order.
      auto indices = cuda_reg_thread_to_strided<E::nt>(
        merge.indices, tid, shared.indices
      );

      // Gather the input values and merge into the output values.
      cuda_transfer_two_streams_strided<E::nt>(
        a_vals + range.a_begin, range.a_count(),
        b_vals + range.b_begin, range.b_count(), indices, tid,
        c_vals + dest_offset
      );
    }
  });
}

}  // end of namespace tf::detail ---------------------------------------------

namespace tf {

// ----------------------------------------------------------------------------
// standalone merge algorithms
// ----------------------------------------------------------------------------

/**
@brief queries the buffer size in bytes needed to call merge kernels

@tparam P execution polity type
@param a_count number of elements in the first input array
@param b_count number of elements in the second input array

The function is used to allocate a buffer for calling
tf::cuda_merge.
*/
template <typename P>
unsigned cuda_merge_buffer_size(unsigned a_count, unsigned b_count) {
  using E = std::decay_t<P>;
  unsigned sz = (a_count + b_count + E::nv - 1) / E::nv + 1;
  return sz*sizeof(unsigned);
}

// ----------------------------------------------------------------------------
// key-value merge
// ----------------------------------------------------------------------------

//template<
//  typename P,
//  typename a_keys_it, typename a_vals_it,
//  typename b_keys_it, typename b_vals_it,
//  typename c_keys_it, typename c_vals_it,
//  typename C
//>
//void cuda_merge(
//  P&& p,
//  a_keys_it a_keys_first, a_vals_it a_vals_first, a_keys_it a_keys_last,
//  b_keys_it b_keys_first, b_vals_it b_vals_first, b_keys_it b_keys_last,
//  c_keys_it c_keys_first, c_vals_it c_vals_first, C comp
//) {
//
//  unsigned a_count = std::distance(a_keys_first, a_keys_last);
//  unsigned b_count = std::distance(b_keys_first, b_keys_last);
//
//  if(a_count + b_count == 0) {
//    return;
//  }
//
//  // allocate temporary buffer
//  cudaDeviceVector<std::byte> temp(cuda_merge_buffer_size<P>(a_count, b_count));
//
//  detail::cuda_merge_loop(
//    p,
//    a_keys_first, a_vals_first, a_count,
//    b_keys_first, b_vals_first, b_count,
//    c_keys_first, c_vals_first, comp,
//    temp.data()
//  );
//
//  // synchronize the execution
//  p.synchronize();
//}

/**
@brief performs asynchronous key-value merge over a range of keys and values

@tparam P execution policy type
@tparam a_keys_it first key iterator type
@tparam a_vals_it first value iterator type
@tparam b_keys_it second key iterator type
@tparam b_vals_it second value iterator type
@tparam c_keys_it output key iterator type
@tparam c_vals_it output value iterator type
@tparam C comparator type

@param p execution policy
@param a_keys_first iterator to the beginning of the first key range
@param a_keys_last iterator to the end of the first key range
@param a_vals_first iterator to the beginning of the first value range
@param b_keys_first iterator to the beginning of the second key range
@param b_keys_last iterator to the end of the second key range
@param b_vals_first iterator to the beginning of the second value range
@param c_keys_first iterator to the beginning of the output key range
@param c_vals_first iterator to the beginning of the output value range
@param comp comparator
@param buf pointer to the temporary buffer

Performs a key-value merge that copies elements from
<tt>[a_keys_first, a_keys_last)</tt> and <tt>[b_keys_first, b_keys_last)</tt>
into a single range, <tt>[c_keys_first, c_keys_last + (a_keys_last - a_keys_first) + (b_keys_last - b_keys_first))</tt>
such that the resulting range is in ascending key order.

At the same time, the merge copies elements from the two associated ranges
<tt>[a_vals_first + (a_keys_last - a_keys_first))</tt> and
<tt>[b_vals_first + (b_keys_last - b_keys_first))</tt> into a single range,
<tt>[c_vals_first, c_vals_first + (a_keys_last - a_keys_first) + (b_keys_last - b_keys_first))</tt>
such that the resulting range is in ascending order
implied by each input element's associated key.

For example, assume:
  + @c a_keys = {8, 1};
  + @c a_vals = {1, 2};
  + @c b_keys = {3, 7};
  + @c b_vals = {3, 4};

After the merge, we have:
  + @c c_keys = {1, 3, 7, 8}
  + @c c_vals = {2, 3, 4, 1}

*/
template<
  typename P,
  typename a_keys_it, typename a_vals_it,
  typename b_keys_it, typename b_vals_it,
  typename c_keys_it, typename c_vals_it,
  typename C
>
void cuda_merge_by_key(
  P&& p,
  a_keys_it a_keys_first, a_keys_it a_keys_last, a_vals_it a_vals_first,
  b_keys_it b_keys_first, b_keys_it b_keys_last, b_vals_it b_vals_first,
  c_keys_it c_keys_first, c_vals_it c_vals_first, C comp,
  void* buf
) {

  unsigned a_count = std::distance(a_keys_first, a_keys_last);
  unsigned b_count = std::distance(b_keys_first, b_keys_last);

  if(a_count + b_count == 0) {
    return;
  }

  detail::cuda_merge_loop(p,
    a_keys_first, a_vals_first, a_count,
    b_keys_first, b_vals_first, b_count,
    c_keys_first, c_vals_first, comp,
    buf
  );
}

// ----------------------------------------------------------------------------
// key-only merge
// ----------------------------------------------------------------------------

//template<typename P,
//  typename a_keys_it, typename b_keys_it, typename c_keys_it, typename C
//>
//void cuda_merge(
//  P&& p,
//  a_keys_it a_keys_first, a_keys_it a_keys_last,
//  b_keys_it b_keys_first, b_keys_it b_keys_last,
//  c_keys_it c_keys_first,
//  C comp
//) {
//  cuda_merge(
//    p,
//    a_keys_first, (const cudaEmpty*)nullptr, a_keys_last,
//    b_keys_first, (const cudaEmpty*)nullptr, b_keys_last,
//    c_keys_first, (cudaEmpty*)nullptr, comp
//  );
//}

/**
@brief performs asynchronous key-only merge over a range of keys

@tparam P execution policy type
@tparam a_keys_it first key iterator type
@tparam b_keys_it second key iterator type
@tparam c_keys_it output key iterator type
@tparam C comparator type

@param p execution policy
@param a_keys_first iterator to the beginning of the first key range
@param a_keys_last iterator to the end of the first key range
@param b_keys_first iterator to the beginning of the second key range
@param b_keys_last iterator to the end of the second key range
@param c_keys_first iterator to the beginning of the output key range
@param comp comparator
@param buf pointer to the temporary buffer

This function is equivalent to tf::cuda_merge_by_key without values.

*/
template<typename P,
  typename a_keys_it, typename b_keys_it, typename c_keys_it, typename C
>
void cuda_merge(
  P&& p,
  a_keys_it a_keys_first, a_keys_it a_keys_last,
  b_keys_it b_keys_first, b_keys_it b_keys_last,
  c_keys_it c_keys_first,
  C comp,
  void* buf
) {
  cuda_merge_by_key(
    p,
    a_keys_first, a_keys_last, (const cudaEmpty*)nullptr,
    b_keys_first, b_keys_last, (const cudaEmpty*)nullptr,
    c_keys_first, (cudaEmpty*)nullptr, comp,
    buf
  );
}

// ----------------------------------------------------------------------------
// cudaFlow merge algorithms
// ----------------------------------------------------------------------------

// Function: merge
template<typename A, typename B, typename C, typename Comp>
cudaTask cudaFlow::merge(
  A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp
) {
  return capture([=](cudaFlowCapturer& cap){
    cap.make_optimizer<cudaLinearCapturing>();
    cap.merge(a_first, a_last, b_first, b_last, c_first, comp);
  });
}

// Function: merge
template<typename A, typename B, typename C, typename Comp>
void cudaFlow::merge(
  cudaTask task, A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp
) {
  capture(task, [=](cudaFlowCapturer& cap){
    cap.make_optimizer<cudaLinearCapturing>();
    cap.merge(a_first, a_last, b_first, b_last, c_first, comp);
  });
}

// Function: merge_by_key
template<
  typename a_keys_it, typename a_vals_it,
  typename b_keys_it, typename b_vals_it,
  typename c_keys_it, typename c_vals_it,
  typename C
>
cudaTask cudaFlow::merge_by_key(
  a_keys_it a_keys_first, a_keys_it a_keys_last, a_vals_it a_vals_first,
  b_keys_it b_keys_first, b_keys_it b_keys_last, b_vals_it b_vals_first,
  c_keys_it c_keys_first, c_vals_it c_vals_first, C comp
) {
  return capture([=](cudaFlowCapturer& cap){
    cap.make_optimizer<cudaLinearCapturing>();
    cap.merge_by_key(
      a_keys_first, a_keys_last, a_vals_first,
      b_keys_first, b_keys_last, b_vals_first,
      c_keys_first, c_vals_first,
      comp
    );
  });
}

// Function: merge_by_key
template<
  typename a_keys_it, typename a_vals_it,
  typename b_keys_it, typename b_vals_it,
  typename c_keys_it, typename c_vals_it,
  typename C
>
void cudaFlow::merge_by_key(
  cudaTask task,
  a_keys_it a_keys_first, a_keys_it a_keys_last, a_vals_it a_vals_first,
  b_keys_it b_keys_first, b_keys_it b_keys_last, b_vals_it b_vals_first,
  c_keys_it c_keys_first, c_vals_it c_vals_first, C comp
) {
  capture(task, [=](cudaFlowCapturer& cap){
    cap.make_optimizer<cudaLinearCapturing>();
    cap.merge_by_key(
      a_keys_first, a_keys_last, a_vals_first,
      b_keys_first, b_keys_last, b_vals_first,
      c_keys_first, c_vals_first,
      comp
    );
  });
}

// ----------------------------------------------------------------------------
// cudaFlowCapturer merge algorithms
// ----------------------------------------------------------------------------

// Function: merge
template<typename A, typename B, typename C, typename Comp>
cudaTask cudaFlowCapturer::merge(
  A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp
) {
  // TODO
  auto bufsz = cuda_merge_buffer_size<cudaDefaultExecutionPolicy>(
    std::distance(a_first, a_last), std::distance(b_first, b_last)
  );

  return on([=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
  (cudaStream_t stream) mutable {
    cuda_merge(cudaDefaultExecutionPolicy{stream},
      a_first, a_last, b_first, b_last, c_first, comp, buf.get().data()
    );
  });
}

// Procedure: merge (update)
template<typename A, typename B, typename C, typename Comp>
void cudaFlowCapturer::merge(
  cudaTask task, A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp
) {
  // TODO
  auto bufsz = cuda_merge_buffer_size<cudaDefaultExecutionPolicy>(
    std::distance(a_first, a_last), std::distance(b_first, b_last)
  );

  on(task, [=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
  (cudaStream_t stream) mutable {
    cuda_merge(cudaDefaultExecutionPolicy{stream},
      a_first, a_last, b_first, b_last, c_first, comp, buf.get().data()
    );
  });
}

// Function: merge_by_key
template<
  typename a_keys_it, typename a_vals_it,
  typename b_keys_it, typename b_vals_it,
  typename c_keys_it, typename c_vals_it,
  typename C
>
cudaTask cudaFlowCapturer::merge_by_key(
  a_keys_it a_keys_first, a_keys_it a_keys_last, a_vals_it a_vals_first,
  b_keys_it b_keys_first, b_keys_it b_keys_last, b_vals_it b_vals_first,
  c_keys_it c_keys_first, c_vals_it c_vals_first, C comp
) {

  auto bufsz = cuda_merge_buffer_size<cudaDefaultExecutionPolicy>(
    std::distance(a_keys_first, a_keys_last),
    std::distance(b_keys_first, b_keys_last)
  );

  return on([=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
  (cudaStream_t stream) mutable {
    cuda_merge_by_key(cudaDefaultExecutionPolicy{stream},
      a_keys_first, a_keys_last, a_vals_first,
      b_keys_first, b_keys_last, b_vals_first,
      c_keys_first, c_vals_first,
      comp,
      buf.get().data()
    );
  });
}

// Function: merge_by_key
template<
  typename a_keys_it, typename a_vals_it,
  typename b_keys_it, typename b_vals_it,
  typename c_keys_it, typename c_vals_it,
  typename C
>
void cudaFlowCapturer::merge_by_key(
  cudaTask task,
  a_keys_it a_keys_first, a_keys_it a_keys_last, a_vals_it a_vals_first,
  b_keys_it b_keys_first, b_keys_it b_keys_last, b_vals_it b_vals_first,
  c_keys_it c_keys_first, c_vals_it c_vals_first, C comp
) {

  auto bufsz = cuda_merge_buffer_size<cudaDefaultExecutionPolicy>(
    std::distance(a_keys_first, a_keys_last),
    std::distance(b_keys_first, b_keys_last)
  );

  on(task, [=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
  (cudaStream_t stream) mutable {
    cuda_merge_by_key(cudaDefaultExecutionPolicy{stream},
      a_keys_first, a_keys_last, a_vals_first,
      b_keys_first, b_keys_last, b_vals_first,
      c_keys_first, c_vals_first,
      comp,
      buf.get().data()
    );
  });
}



}  // end of namespace tf -----------------------------------------------------
