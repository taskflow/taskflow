#pragma once

#include "../syclflow.hpp"

namespace tf::detail {

// ----------------------------------------------------------------------------
// reduction helper functions
// ----------------------------------------------------------------------------

/** @private */
template<unsigned nt, typename T>
struct syclBlockReduce {

  static const unsigned group_size = std::min(nt, SYCL_WARP_SIZE);
  static const unsigned shm_size   = std::max(nt, 2* group_size);
  static const unsigned num_passes = log2(group_size);
  static const unsigned num_items  = nt / group_size;

  static_assert(
    nt && (0 == nt % SYCL_WARP_SIZE),
    "syclBlockReduce requires num threads to be a multiple of warp_size (32)"
  );

  using shm_t = sycl::accessor<
    T, 1, sycl::access::mode::read_write, sycl::access::target::local
  >;

  template<typename op_t>
  T operator()(
    sycl::nd_item<1>&, T, const shm_t&, unsigned, op_t, bool = true
  ) const;
};

// function: reduce to be called from a block
template<unsigned nt, typename T>
template<typename op_t>
T syclBlockReduce<nt, T>::operator ()(
  sycl::nd_item<1>& item,
  T x,
  const shm_t& shm,
  unsigned count,
  op_t op,
  bool ret
) const {

  auto tid = item.get_local_id(0);

  // Store your data into shared memory.
  shm[tid] = x;
  item.barrier(sycl::access::fence_space::local_space);

  if(tid < group_size) {
    // Each thread scans within its lane.
    sycl_strided_iterate<group_size, num_items>([&](auto i, auto j) {
      if(i > 0) {
        x = op(x, shm[j]);
      }
    }, tid, count);
    shm[tid] = x;
  }
  item.barrier(sycl::access::fence_space::local_space);

  auto count2 = count < group_size ? count : group_size;
  auto first = (1 & num_passes) ? group_size : 0;
  if(tid < group_size) {
    shm[first + tid] = x;
  }
  item.barrier(sycl::access::fence_space::local_space);

  sycl_iterate<num_passes>([&](auto pass) {
    if(tid < group_size) {
      if(auto offset = 1 << pass; tid + offset < count2) {
        x = op(x, shm[first + offset + tid]);
      }
      first = group_size - first;
      shm[first + tid] = x;
    }
    item.barrier(sycl::access::fence_space::local_space);
  });

  if(ret) {
    x = shm[0];
    item.barrier(sycl::access::fence_space::local_space);
  }
  return x;
}

/** @private */
template <typename P, typename I, typename T, typename O>
sycl::event sycl_reduce_loop(
  P&& p,
  I input,
  unsigned count,
  T* res,
  O op,
  bool incl,
  void* ptr,
  std::vector<sycl::event> evs
) {

  using E = std::decay_t<P>;
  using R = syclBlockReduce<E::nt, T>;

  auto buf = static_cast<T*>(ptr);
  auto B   = (count + E::nv - 1) / E::nv;

  auto e = p.queue().submit([=, evs=std::move(evs)](sycl::handler& h) {

    h.depends_on(evs);

    // create a shared memory
    typename R::shm_t shm(sycl::range<1>(R::shm_size), h);

    h.parallel_for(
      sycl::nd_range<1>{sycl::range<1>(B*E::nt), sycl::range<1>(E::nt)},
      [=](sycl::nd_item<1> item) {

        auto tid = item.get_local_id(0);
        auto bid = item.get_group(0);

        // get the tile of this group
        auto tile = sycl_get_tile(bid, E::nv, count);

        // load data from input to register
        auto x = sycl_mem_to_reg_strided<E::nt, E::vt>(
          input + tile.begin, tid, tile.count()
        );
        // reduce multiple values per thread into a scalar.
        T s;
        sycl_strided_iterate<E::nt, E::vt>(
          [&] (auto i, auto) { s = i ? op(s, x[i]) : x[0]; }, tid, tile.count()
        );
        // reduce to a scalar per block.
        s = R()(
          item, s, shm, (tile.count()<E::nt ? tile.count() : E::nt), op, false
        );
        if(!tid) {
          (1 == B) ? *res = (incl ? op(*res, s) : s) : buf[bid] = s;
        }
      }
    );
  });

  if(B > 1) {
    return sycl_reduce_loop(p, buf, B, res, op, incl, buf+B, {e});
  }
  else {
    return e;
  }
}

}  // end of namespace detail -------------------------------------------------

namespace tf {

/**
@brief queries the buffer size in bytes needed to call reduce kernels

@tparam P execution policy type
@tparam T value type

@param count number of elements to reduce

The function is used to allocate a buffer for calling asynchronous reduce.
Please refer to @ref SYCLSTDReduce for details.
*/
template <typename P, typename T>
unsigned sycl_reduce_buffer_size(unsigned count) {
  using E = std::decay_t<P>;
  unsigned B = (count + E::nv - 1) / E::nv;
  unsigned n = 0;
  for(auto b=B; b>1; n += (b=(b+E::nv-1)/E::nv));
  return n*sizeof(T);
}

//// sycl reduction
//template <typename I, typename T, typename C, bool uninitialized>
//auto syclFlow::_reduce_cgh(I first, I last, T* res, C&& op) {
//
//  // TODO: special case N == 0?
//  size_t N = std::distance(first, last);
//  size_t B = _default_group_size(N);
//
//  return [=, op=std::forward<C>(op)](sycl::handler& handler) mutable {
//
//    // create a shared memory
//    sycl::accessor<
//      T, 1, sycl::access::mode::read_write, sycl::access::target::local
//    > shm(sycl::range<1>(B), handler);
//
//    // perform parallel reduction
//    handler.parallel_for(
//      sycl::nd_range<1>{sycl::range<1>(B), sycl::range<1>(B)},
//      [=] (sycl::nd_item<1> item) {
//
//      size_t tid = item.get_global_id(0);
//
//      if(tid >= N) {
//        return;
//      }
//
//      shm[tid] = *(first+tid);
//
//      for(size_t i=tid+B; i<N; i+=B) {
//        shm[tid] = op(shm[tid], *(first+i));
//      }
//
//      item.barrier(sycl::access::fence_space::local_space);
//
//      for(size_t s = B / 2; s > 0; s >>= 1) {
//        if(tid < s && tid + s < N) {
//          shm[tid] = op(shm[tid], shm[tid+s]);
//        }
//        item.barrier(sycl::access::fence_space::local_space);
//      }
//
//      if(tid == 0) {
//        if constexpr (uninitialized) {
//          *res = shm[0];
//        }
//        else {
//          *res = op(*res, shm[0]);
//        }
//      }
//    });
//  };
//}

// ----------------------------------------------------------------------------
// SYCL standard reduce algorithms
// ----------------------------------------------------------------------------

/**
@brief performs parallel reduction over a range of items

@tparam P execution policy type
@tparam I input iterator type
@tparam T value type
@tparam O binary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param res pointer to the result
@param op binary operator to apply to reduce elements

This method is equivalent to the parallel execution of the following loop
on a SYCL device:

@code{.cpp}
while (first != last) {
  *result = op(*result, *first++);
}
@endcode
 */
template<typename P, typename I, typename T, typename O>
void sycl_reduce(P&& p, I first, I last, T* res, O op) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }

  // allocate temporary buffer
  auto tmp = sycl::malloc_device(
    sycl_reduce_buffer_size<P, T>(count), p.queue()
  );

  // reduction loop
  detail::sycl_reduce_loop(p, first, count, res, op, true, tmp, {}).wait();

  // deallocate the temporary buffer
  sycl::free(tmp, p.queue());
}

/**
@brief performs asynchronous parallel reduction over a range of items

@tparam P execution policy type
@tparam I input iterator type
@tparam T value type
@tparam O binary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param res pointer to the result
@param op binary operator to apply to reduce elements
@param buf pointer to the temporary buffer

@return an SYCL event

Please refer to @ref SYCLSTDReduce for details.
 */
template<typename P, typename I, typename T, typename O>
sycl::event sycl_reduce_async(
  P&& p, I first, I last, T* res, O op, void* buf, std::vector<sycl::event> dep
) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return {};
  }

  // reduction loop
  return detail::sycl_reduce_loop(
    p, first, count, res, op, true, buf, std::move(dep)
  );
}

/**
@brief performs parallel reduction over a range of items
       without an initial value

@tparam P execution policy type
@tparam I input iterator type
@tparam T value type
@tparam O binary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param res pointer to the result
@param op binary operator to apply to reduce elements

This method is equivalent to the parallel execution of the following loop
on a SYCL device:

@code{.cpp}
*result = *first++;  // no initial values partitipcate in the loop
while (first != last) {
  *result = op(*result, *first++);
}
@endcode
*/
template<typename P, typename I, typename T, typename O>
void sycl_uninitialized_reduce(P&& p, I first, I last, T* res, O op) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }

  // allocate temporary buffer
  auto tmp = sycl::malloc_device(
    sycl_reduce_buffer_size<P, T>(count), p.queue()
  );

  // reduction loop
  detail::sycl_reduce_loop(p, first, count, res, op, false, tmp, {}).wait();

  // deallocate the temporary buffer
  sycl::free(tmp, p.queue());
}

/**
@brief performs asynchronous parallel reduction over a range of items
       without an initial value

@tparam P execution policy type
@tparam I input iterator type
@tparam T value type
@tparam O binary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param res pointer to the result
@param op binary operator to apply to reduce elements
@param buf pointer to the temporary buffer

@return an SYCL event

Please refer to @ref SYCLSTDReduce for details.
*/
template<typename P, typename I, typename T, typename O>
sycl::event sycl_uninitialized_reduce_async(
  P&& p, I first, I last, T* res, O op, void* buf, std::vector<sycl::event> dep
) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return {};
  }

  // reduction loop
  return detail::sycl_reduce_loop(
    p, first, count, res, op, false, buf, std::move(dep)
  );
}

// ----------------------------------------------------------------------------
// syclFlow reduce
// ----------------------------------------------------------------------------

// Function: reduce
template <typename I, typename T, typename C>
syclTask syclFlow::reduce(I first, I last, T* res, C&& op) {

  //return on(_reduce_cgh<I, T, C, false>(first, last, res, std::forward<C>(op)));

  auto bufsz = sycl_reduce_buffer_size<syclDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  return on([=, buf=MoC{syclScopedDeviceMemory<std::byte>(bufsz, _queue)}]
  (sycl::queue& queue, std::vector<sycl::event> events) mutable {
    syclDefaultExecutionPolicy p(queue);
    return sycl_reduce_async(
      p, first, last, res, op, buf.get().data(), std::move(events)
    );
  });
}

// Function: uninitialized_reduce
template <typename I, typename T, typename C>
syclTask syclFlow::uninitialized_reduce(I first, I last, T* res, C&& op) {
  //return on(_reduce_cgh<I, T, C, true>(first, last, res, std::forward<C>(op)));

  auto bufsz = sycl_reduce_buffer_size<syclDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  return on([=, buf=MoC{syclScopedDeviceMemory<std::byte>(bufsz, _queue)}]
  (sycl::queue& queue, std::vector<sycl::event> events) mutable {
    syclDefaultExecutionPolicy p(queue);
    return sycl_uninitialized_reduce_async(
      p, first, last, res, op, buf.get().data(), std::move(events)
    );
  });

}

// ----------------------------------------------------------------------------
// rebind methods
// ----------------------------------------------------------------------------

//// Function: reduce
//template <typename I, typename T, typename C>
//void syclFlow::reduce(syclTask task, I first, I last, T* res, C&& op) {
//  //on(task, _reduce_cgh<I, T, C, false>(
//  //  first, last, res, std::forward<C>(op)
//  //));
//
//  auto bufsz = sycl_reduce_buffer_size<syclDefaultExecutionPolicy, T>(
//    std::distance(first, last)
//  );
//
//  on(task, [=, buf=MoC{syclScopedDeviceMemory<std::byte>(bufsz, _queue)}]
//  (sycl::queue& queue, std::vector<sycl::event> events) mutable {
//    syclDefaultExecutionPolicy p(queue);
//    return sycl_reduce_async(
//      p, first, last, res, op, buf.get().data(), std::move(events)
//    );
//  });
//}
//
//// Function: uninitialized_reduce
//template <typename I, typename T, typename C>
//void syclFlow::uninitialized_reduce(
//  syclTask task, I first, I last, T* res, C&& op
//) {
//  //on(task, _reduce_cgh<I, T, C, true>(
//  //  first, last, res, std::forward<C>(op)
//  //));
//  auto bufsz = sycl_reduce_buffer_size<syclDefaultExecutionPolicy, T>(
//    std::distance(first, last)
//  );
//
//  on(task, [=, buf=MoC{syclScopedDeviceMemory<std::byte>(bufsz, _queue)}]
//  (sycl::queue& queue, std::vector<sycl::event> events) mutable {
//    syclDefaultExecutionPolicy p(queue);
//    return sycl_uninitialized_reduce_async(
//      p, first, last, res, op, buf.get().data(), std::move(events)
//    );
//  });
//}


}  // end of namespace tf -----------------------------------------------------


