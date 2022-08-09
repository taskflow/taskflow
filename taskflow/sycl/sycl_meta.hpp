#pragma once

#include "sycl_execution_policy.hpp"

namespace tf {

// default warp size
inline constexpr unsigned SYCL_WARP_SIZE = 32;

// empty type
struct syclEmpty { };

// ----------------------------------------------------------------------------
// iterator unrolling
// ----------------------------------------------------------------------------

// Template unrolled looping construct.
template<unsigned i, unsigned count, bool valid = (i < count)>
struct syclIterate {
  template<typename F>
  static void eval(F f) {
    f(i);
    syclIterate<i + 1, count>::eval(f);
  }
};

template<unsigned i, unsigned count>
struct syclIterate<i, count, false> {
  template<typename F>
  static void eval(F) { }
};

template<unsigned begin, unsigned end, typename F>
void sycl_iterate(F f) {
  syclIterate<begin, end>::eval(f);
}

template<unsigned count, typename F>
void sycl_iterate(F f) {
  sycl_iterate<0, count>(f);
}

template<unsigned count, typename T>
T reduce(const T(&x)[count]) {
  T y;
  sycl_iterate<count>([&](auto i) { y = i ? x[i] + y : x[i]; });
  return y;
}

template<unsigned count, typename T>
void fill(T(&x)[count], T val) {
  sycl_iterate<count>([&](auto i) { x[i] = val; });
}

// Invoke unconditionally.
template<unsigned nt, unsigned vt, typename F>
void sycl_strided_iterate(F f, unsigned tid) {
  sycl_iterate<vt>([=](auto i) { f(i, nt * i + tid); });
}

// Check range.
template<unsigned nt, unsigned vt, unsigned vt0 = vt, typename F>
void sycl_strided_iterate(F f, unsigned tid, unsigned count) {
  // Unroll the first vt0 elements of each thread.
  if(vt0 > 1 && count >= nt * vt0) {
    sycl_strided_iterate<nt, vt0>(f, tid);    // No checking
  } else {
    sycl_iterate<vt0>([=](auto i) {
      auto j = nt * i + tid;
      if(j < count) f(i, j);
    });
  }

  // TODO: seems dummy when vt0 == vt
  sycl_iterate<vt0, vt>([=](auto i) {
    auto j = nt * i + tid;
    if(j < count) f(i, j);
  });
}

template<unsigned vt, typename F>
void sycl_thread_iterate(F f, unsigned tid) {
  sycl_iterate<vt>([=](auto i) { f(i, vt * tid + i); });
}

// ----------------------------------------------------------------------------
// syclRange
// ----------------------------------------------------------------------------

// syclRange
struct syclRange {
  unsigned begin, end;
  unsigned size() const { return end - begin; }
  unsigned count() const { return size(); }
  bool valid() const { return end > begin; }
};

inline syclRange sycl_get_tile(unsigned b, unsigned nv, unsigned count) {
  return syclRange { nv * b, std::min(count, nv * (b + 1)) };
}


// ----------------------------------------------------------------------------
// syclArray
// ----------------------------------------------------------------------------

template<typename T, unsigned size>
struct syclArray {
  T data[size];

  T operator[](unsigned i) const { return data[i]; }
  T& operator[](unsigned i) { return data[i]; }

  syclArray() = default;
  syclArray(const syclArray&) = default;
  syclArray& operator=(const syclArray&) = default;

  // Fill the array with x.
  syclArray(T x) {
    sycl_iterate<size>([&](unsigned i) { data[i] = x; });
  }
};

template<typename T>
struct syclArray<T, 0> {
  T operator[](unsigned) const { return T(); }
  T& operator[](unsigned) { return *(T*)nullptr; }
};

template<typename T, typename V, unsigned size>
struct syclKVArray {
  syclArray<T, size> keys;
  syclArray<V, size> vals;
};

// ----------------------------------------------------------------------------
// thread reg <-> global mem
// ----------------------------------------------------------------------------

template<unsigned nt, unsigned vt, unsigned vt0 = vt, typename I>
auto sycl_mem_to_reg_strided(I mem, unsigned tid, unsigned count) {
  using T = typename std::iterator_traits<I>::value_type;
  syclArray<T, vt> x;
  sycl_strided_iterate<nt, vt, vt0>(
    [&](auto i, auto j) { x[i] = mem[j]; }, tid, count
  );
  return x;
}

template<unsigned nt, unsigned vt, unsigned vt0 = vt, typename T, typename it_t>
void sycl_reg_to_mem_strided(
  syclArray<T, vt> x, unsigned tid, unsigned count, it_t mem) {

  sycl_strided_iterate<nt, vt, vt0>(
    [=](auto i, auto j) { mem[j] = x[i]; }, tid, count
  );
}

template<unsigned nt, unsigned vt, unsigned vt0 = vt, typename I, typename O>
auto sycl_transform_mem_to_reg_strided(
  I mem, unsigned tid, unsigned count, O op
) {
  using T = std::invoke_result_t<O, typename std::iterator_traits<I>::value_type>;
  syclArray<T, vt> x;
  sycl_strided_iterate<nt, vt, vt0>(
    [&](auto i, auto j) { x[i] = op(mem[j]); }, tid, count
  );
  return x;
}

// ----------------------------------------------------------------------------
// thread reg <-> shared
// ----------------------------------------------------------------------------

//template<unsigned nt, unsigned vt, typename T, unsigned shared_size>
//void sycl_reg_to_shared_thread(
//  syclArray<T, vt> x, unsigned tid, T (&shared)[shared_size], bool sync = true
//) {
//
//  static_assert(shared_size >= nt * vt,
//    "reg_to_shared_thread must have at least nt * vt storage");
//
//  sycl_thread_iterate<vt>([&](auto i, auto j) { shared[j] = x[i]; }, tid);
//
//  if(sync) __syncthreads();
//}
//
//template<unsigned nt, unsigned vt, typename T, unsigned shared_size>
//auto sycl_shared_to_reg_thread(
//  const T (&shared)[shared_size], unsigned tid, bool sync = true
//) {
//
//  static_assert(shared_size >= nt * vt,
//    "reg_to_shared_thread must have at least nt * vt storage");
//
//  syclArray<T, vt> x;
//  sycl_thread_iterate<vt>([&](auto i, auto j) {
//    x[i] = shared[j];
//  }, tid);
//
//  if(sync) __syncthreads();
//
//  return x;
//}
//
//template<unsigned nt, unsigned vt, typename T, unsigned shared_size>
//void sycl_reg_to_shared_strided(
//  syclArray<T, vt> x, unsigned tid, T (&shared)[shared_size], bool sync = true
//) {
//
//  static_assert(shared_size >= nt * vt,
//    "reg_to_shared_strided must have at least nt * vt storage");
//
//  sycl_strided_iterate<nt, vt>(
//    [&](auto i, auto j) { shared[j] = x[i]; }, tid
//  );
//
//  if(sync) __syncthreads();
//}
//
//template<unsigned nt, unsigned vt, typename T, unsigned shared_size>
//auto sycl_shared_to_reg_strided(
//  const T (&shared)[shared_size], unsigned tid, bool sync = true
//) {
//
//  static_assert(shared_size >= nt * vt,
//    "shared_to_reg_strided must have at least nt * vt storage");
//
//  syclArray<T, vt> x;
//  sycl_strided_iterate<nt, vt>([&](auto i, auto j) { x[i] = shared[j]; }, tid);
//  if(sync) __syncthreads();
//
//  return x;
//}
//
//template<
//  unsigned nt, unsigned vt, unsigned vt0 = vt, typename T, typename it_t,
//  unsigned shared_size
//>
//auto sycl_reg_to_mem_thread(
//  syclArray<T, vt> x, unsigned tid,
//  unsigned count, it_t mem, T (&shared)[shared_size]
//) {
//  sycl_reg_to_shared_thread<nt>(x, tid, shared);
//  auto y = sycl_shared_to_reg_strided<nt, vt>(shared, tid);
//  sycl_reg_to_mem_strided<nt, vt, vt0>(y, tid, count, mem);
//}
//
//template<
//  unsigned nt, unsigned vt, unsigned vt0 = vt, typename T, typename it_t,
//  unsigned shared_size
//>
//auto sycl_mem_to_reg_thread(
//  it_t mem, unsigned tid, unsigned count, T (&shared)[shared_size]
//) {
//
//  auto x = sycl_mem_to_reg_strided<nt, vt, vt0>(mem, tid, count);
//  sycl_reg_to_shared_strided<nt, vt>(x, tid, shared);
//  auto y = sycl_shared_to_reg_thread<nt, vt>(shared, tid);
//  return y;
//}
//
//template<unsigned nt, unsigned vt, typename T, unsigned S>
//auto sycl_shared_gather(
//  const T(&data)[S], syclArray<unsigned, vt> indices, bool sync = true
//) {
//
//  static_assert(S >= nt * vt,
//    "shared_gather must have at least nt * vt storage");
//
//  syclArray<T, vt> x;
//  sycl_iterate<vt>([&](auto i) { x[i] = data[indices[i]]; });
//
//  if(sync) __syncthreads();
//
//  return x;
//}
//
//
//
//// ----------------------------------------------------------------------------
//// reg<->reg
//// ----------------------------------------------------------------------------
//
//template<unsigned nt, unsigned vt, typename T, unsigned S>
//auto sycl_reg_thread_to_strided(
//  syclArray<T, vt> x, unsigned tid, T (&shared)[S]
//) {
//  sycl_reg_to_shared_thread<nt>(x, tid, shared);
//  return sycl_shared_to_reg_strided<nt, vt>(shared, tid);
//}
//
//template<unsigned nt, unsigned vt, typename T, unsigned S>
//auto sycl_reg_strided_to_thread(
//  syclArray<T, vt> x, unsigned tid, T (&shared)[S]
//) {
//  sycl_reg_to_shared_strided<nt>(x, tid, shared);
//  return sycl_shared_to_reg_thread<nt, vt>(shared, tid);
//}

// ----------------------------------------------------------------------------
// syclLoadStoreIterator
// ----------------------------------------------------------------------------

template<typename L, typename S, typename T, typename I>
struct syclLoadStoreIterator : std::iterator_traits<const T*> {

  L load;
  S store;
  I base;

  syclLoadStoreIterator(L load_, S store_, I base_) :
    load(load_), store(store_), base(base_) { }

  struct assign_t {
    L load;
    S store;
    I index;

    assign_t& operator=(T rhs) {
      static_assert(!std::is_same<S, syclEmpty>::value,
        "load_iterator is being stored to.");
      store(rhs, index);
      return *this;
    }
    operator T() const {
      static_assert(!std::is_same<L, syclEmpty>::value,
        "store_iterator is being loaded from.");
      return load(index);
    }
  };

  assign_t operator[](I index) const {
    return assign_t { load, store, base + index };
  }
  assign_t operator*() const {
    return assign_t { load, store, base };
  }

  syclLoadStoreIterator operator+(I offset) const {
    syclLoadStoreIterator cp = *this;
    cp += offset;
    return cp;
  }

  syclLoadStoreIterator& operator+=(I offset) {
    base += offset;
    return *this;
  }

  syclLoadStoreIterator operator-(I offset) const {
    syclLoadStoreIterator cp = *this;
    cp -= offset;
    return cp;
  }

  syclLoadStoreIterator& operator-=(I offset) {
    base -= offset;
    return *this;
  }
};

//template<typename T>
//struct trivial_load_functor {
//  template<typename I>
//  T operator()(I index) const {
//    return T();
//  }
//};

//template<typename T>
//struct trivial_store_functor {
//  template<typename I>
//  void operator()(T v, I index) const { }
//};

template <typename T, typename I = int, typename L, typename S>
auto sycl_make_load_store_iterator(L load, S store, I base = 0) {
  return syclLoadStoreIterator<L, S, T, I>(load, store, base);
}

template <typename T, typename I = int, typename L>
auto sycl_make_load_iterator(L load, I base = 0) {
  return sycl_make_load_store_iterator<T>(load, syclEmpty(), base);
}

template <typename T, typename I = int, typename S>
auto sycl_make_store_iterator(S store, I base = 0) {
  return sycl_make_load_store_iterator<T>(syclEmpty(), store, base);
}

// ----------------------------------------------------------------------------
// swap
// ----------------------------------------------------------------------------

template<typename T>
void sycl_swap(T& a, T& b) {
  auto c = a;
  a = b;
  b = c;
}

// ----------------------------------------------------------------------------
// launch kernel
// ----------------------------------------------------------------------------

//template<typename F, typename... args_t>
//__global__ void sycl_kernel(F f, args_t... args) {
//  f(threadIdx.x, blockIdx.x, args...);
//}

// ----------------------------------------------------------------------------
// operators
// ----------------------------------------------------------------------------

template <typename T>
struct sycl_plus : public std::binary_function<T, T, T> {
  T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct sycl_minus : public std::binary_function<T, T, T> {
  T operator()(T a, T b) const { return a - b; }
};

template <typename T>
struct sycl_multiplies : public std::binary_function<T, T, T> {
  T operator()(T a, T b) const { return a * b; }
};

template <typename T>
struct sycl_maximum  : public std::binary_function<T, T, T> {
  T operator()(T a, T b) const { return a > b ? a : b; }
};

template <typename T>
struct sycl_minimum  : public std::binary_function<T, T, T> {
  T operator()(T a, T b) const { return a < b ? a : b; }
};

template <typename T>
struct sycl_less : public std::binary_function<T, T, T> {
  T operator()(T a, T b) const { return a < b; }
};

template <typename T>
struct sycl_greater : public std::binary_function<T, T, T> {
  T operator()(T a, T b) const { return a > b; }
};

// ----------------------------------------------------------------------------
// Memory Object
// ----------------------------------------------------------------------------

/**
@private
*/
template <typename T>
class syclScopedDeviceMemory {

  public:

    syclScopedDeviceMemory() = delete;

    syclScopedDeviceMemory(size_t N, sycl::queue& queue) : 
      _queue {queue},
      _N {N} {
      if(N) {
        _data = sycl::malloc_device<T>(N, _queue);
      }
    }

    syclScopedDeviceMemory(syclScopedDeviceMemory&& rhs) :
      _queue{std::move(rhs._queue)}, _data{rhs._data}, _N {rhs._N} {
      rhs._data = nullptr;
      rhs._N    = 0;
    }

    ~syclScopedDeviceMemory() {
      if(_data) {
        sycl::free(_data, _queue);
      }
    }

    syclScopedDeviceMemory& operator = (syclScopedDeviceMemory&& rhs) {
      if(_data) {
        sycl::free(_data, _queue);
      }
      _queue = std::move(rhs._queue);
      _data  = rhs._data;
      _N     = rhs._N;
      rhs._data = nullptr;
      rhs._N    = 0;
      return *this;
    }

    size_t size() const { return _N; }

    T* data() { return _data; }
    const T* data() const { return _data; }

    syclScopedDeviceMemory(const syclScopedDeviceMemory&) = delete;
    syclScopedDeviceMemory& operator = (const syclScopedDeviceMemory&) = delete;

  private:

    sycl::queue& _queue;

    T* _data  {nullptr};
    size_t _N {0};
};


}  // end of namespace tf -----------------------------------------------------



