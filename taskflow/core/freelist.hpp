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

  TF_FORCE_INLINE Freelist(size_t N) : _buckets(N < 4 ? 1 : floor_log2(N)) {}

  //TF_FORCE_INLINE void push(size_t w, T item) {
  //  std::scoped_lock lock(_buckets[w].mutex);
  //  _buckets[w].queue.push(item);  
  //}

  TF_FORCE_INLINE void push(T item) {
    auto b = reinterpret_cast<uintptr_t>(item) % _buckets.size();
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

  TF_FORCE_INLINE bool empty(size_t& which) const {
    for(which=0; which<_buckets.size(); ++which) {
      if(!_buckets[which].queue.empty()) {
        return false;
      }
    }
    return true;
  }

  private:
  
  std::vector<Bucket> _buckets;
};


}  // end of namespace tf -----------------------------------------------------
