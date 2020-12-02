#pragma once

#include <vector>
#include <mutex>

#include "declarations.hpp"

/** 
@file semaphore.hpp
@brief semaphore include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// Semaphore
// ----------------------------------------------------------------------------

/**
@class Semaphore

@brief class to create a semophore object for building a concurrency constraint

A semaphore creates a constraint that limits the maximum concurrency,
i.e., the number of workers, in a set of tasks.
*/
class Semaphore {

  friend class Node;

  public:
    
    /**
    @brief constructs a semaphore with the given counter
    */
    explicit Semaphore(int initial);
    
    /**
    @brief queries the counter value (not thread-safe)
    */
    int count() const;
    
  private:

    std::mutex _mtx;

    int _counter;

    std::vector<Node*> _waiters;
    
    bool _try_acquire();

    std::vector<Node*> _release();

    void _wait(Node* me);
};

inline Semaphore::Semaphore(int initial) : 
  _counter(initial) {
}
    
inline bool Semaphore::_try_acquire() {
  std::lock_guard<std::mutex> lock(_mtx);
  if(_counter > 0) {
    --_counter;
    return true;
  }
  return false;
}

inline std::vector<Node*> Semaphore::_release() {
  std::lock_guard<std::mutex> lock(_mtx);
  ++_counter;
  std::vector<Node*> r{std::move(_waiters)};
  return r;
}

inline int Semaphore::count() const {
  return _counter;
}

inline void Semaphore::_wait(Node* me) {
  std::lock_guard<std::mutex> lock(_mtx);
  _waiters.push_back(me);
}

}  // end of namespace tf. ---------------------------------------------------


