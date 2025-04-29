#pragma once

#include "tsq.hpp"

namespace tf {

/**
@private
*/
template <typename T>
class Freelist {

  friend class Executor;

  public:
  struct Bucket {
    std::mutex mutex;
    UnboundedTaskQueue<T> queue;
  };  
  
  // Here, we don't create just N task queues in the freelist as it will cause
  // the work-stealing loop to spand a lot of time on stealing tasks.
  // Experimentally speaking, we found floor_log2(N) is the best.
  TF_FORCE_INLINE Freelist(size_t N) : _buckets(N < 4 ? 1 : floor_log2(N)) {}

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
