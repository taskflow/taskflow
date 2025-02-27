#pragma once

#include "tsq.hpp"

namespace tf {

/**
@private
*/
template <typename T>
class Freelist {

  friend class Executor;

  struct Bucket {
    std::mutex mutex;
    UnboundedTaskQueue<T> queue;
  };

  public:

  TF_FORCE_INLINE Freelist(size_t N) : _buckets(N) {}

  TF_FORCE_INLINE void push(size_t w, T item) {
    std::scoped_lock lock(_buckets[w].mutex);
    _buckets[w].queue.push(item);  
  }

  TF_FORCE_INLINE void push(T item) {
    push(reinterpret_cast<uintptr_t>(item) % _buckets.size(), item);
  }

  TF_FORCE_INLINE T steal(size_t w) {
    return _buckets[w].queue.steal();
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
