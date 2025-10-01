#pragma once

#include "tsq.hpp"

namespace tf {

/**
@private 

returns the floor of `log2(N)` at compile time with the special case of 
returnning 1 when N<4
*/
template <size_t N>
constexpr size_t static_freelist_size() {
  return (N < 4) ? 1 : 1 + static_freelist_size<N / 2>();
}

/**
@private
*/
template <size_t... Is>
constexpr auto make_freelist_size_lut_impl(std::index_sequence<Is...>) {
  return std::array<size_t, sizeof...(Is)>{ static_freelist_size<Is>()... };
}

/**
@private
generates a static look-up table for the floor of log2 up to `N` numbers
*/
template <size_t N>
constexpr auto make_freelist_size_lut() {
  static_assert(N > 0, "N must be greater than 0");
  return make_freelist_size_lut_impl(std::make_index_sequence<N>{});
}

/**
@private
*/
template <typename T>
class Freelist {

  friend class Executor;

  public:

  inline constexpr static auto SIZE_LUT = make_freelist_size_lut<128>();

  static_assert(SIZE_LUT[0] == 1);
  static_assert(SIZE_LUT[1] == 1);
  static_assert(SIZE_LUT[2] == 1);
  static_assert(SIZE_LUT[3] == 1);
  static_assert(SIZE_LUT[4] == 2);
  static_assert(SIZE_LUT[5] == 2);
  static_assert(SIZE_LUT[6] == 2);
  static_assert(SIZE_LUT[7] == 2);
  static_assert(SIZE_LUT[8] == 3);

  struct Bucket {
    std::mutex mutex;
    UnboundedTaskQueue<T> queue;
  };  
  
  // Here, we don't create just N task queues in the freelist as it will cause
  // the work-stealing loop to spand a lot of time on stealing tasks.
  // Experimentally speaking, we found floor_log2(N) is the best.
  TF_FORCE_INLINE Freelist(size_t N) : 
    _buckets(N < SIZE_LUT.size() ? SIZE_LUT[N] : floor_log2(N)) {
  }

  // Pointers are aligned to 8 bytes. We perform a simple hash to avoid contention caused
  // by hashing to the same slot.
  TF_FORCE_INLINE void push(T item) {
    //auto b = reinterpret_cast<uintptr_t>(item) % _buckets.size();
    auto b = (reinterpret_cast<uintptr_t>(item) >> 16) % _buckets.size();
    std::scoped_lock lock(_buckets[b].mutex);
    _buckets[b].queue.push(item);
  }

  TF_FORCE_INLINE T steal(size_t w) {
    return _buckets[w].queue.steal();
  }
  
  TF_FORCE_INLINE T steal_with_hint(size_t w, size_t& num_empty_steals) {
    return _buckets[w].queue.steal_with_hint(num_empty_steals);
  }

  TF_FORCE_INLINE size_t size() const {
    return _buckets.size();
  }

  private:
  
  std::vector<Bucket> _buckets;
};


}  // end of namespace tf -----------------------------------------------------
