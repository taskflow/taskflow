#pragma once

#include "tsq.hpp"

namespace tf {

/**
@private
*/
template <typename T>
class Freelist {

  struct Head {
    std::mutex mutex;
    UnboundedTaskQueue<T> queue;
  };

  public:

  Freelist(size_t N) : _heads(N) {}

  void push(size_t w, T item) {
    std::scoped_lock lock(_heads[w].mutex);
    _heads[w].queue.push(item);  
  }

  void push(T item) {
    push(reinterpret_cast<uintptr_t>(item) % _heads.size(), item);
  }

  T steal(size_t w) {
    return _heads[w].queue.steal();
  }


  bool empty() const {
    for(const auto& q : _heads) {
      if(!q.queue.empty()) {
        return false;
      }
    }
    return true;
  }

  private:
  
  std::vector<Head> _heads;
};


}  // end of namespace tf -----------------------------------------------------
