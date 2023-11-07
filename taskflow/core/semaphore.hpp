#pragma once

#include <vector>
#include <mutex>

#include "declarations.hpp"
#include "graph.hpp"

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
You can let a task acquire/release one or multiple semaphores before/after
executing its work.
A task can acquire and release a semaphore,
or just acquire or just release it.
A tf::Semaphore object starts with an initial count.
As long as that count is above 0, tasks can acquire the semaphore and do
their work.
If the count is 0 or less, a task trying to acquire the semaphore will not run
but goes to a waiting list of that semaphore.
When the semaphore is released by another task,
it reschedules all tasks on that waiting list.

@code{.cpp}
tf::Executor executor(8);   // create an executor of 8 workers
tf::Taskflow taskflow;

tf::Semaphore semaphore(1); // create a semaphore with initial count 1

std::vector<tf::Task> tasks {
  taskflow.emplace([](){ std::cout << "A" << std::endl; }),
  taskflow.emplace([](){ std::cout << "B" << std::endl; }),
  taskflow.emplace([](){ std::cout << "C" << std::endl; }),
  taskflow.emplace([](){ std::cout << "D" << std::endl; }),
  taskflow.emplace([](){ std::cout << "E" << std::endl; })
};

for(auto & task : tasks) {  // each task acquires and release the semaphore
  task.acquire(semaphore);
  task.release(semaphore);
}

executor.run(taskflow).wait();
@endcode

The above example creates five tasks with no dependencies between them.
Under normal circumstances, the five tasks would be executed concurrently.
However, this example has a semaphore with initial count 1,
and all tasks need to acquire that semaphore before running and release that
semaphore after they are done.
This arrangement limits the number of concurrently running tasks to only one.

*/
class Semaphore {

  friend class Node;
  friend class Runtime;

  public:

    /**
    @brief constructs a semaphore with the given counter

    A semaphore creates a constraint that limits the maximum concurrency,
    i.e., the number of workers, in a set of tasks.

    @code{.cpp}
    tf::Semaphore semaphore(4);  // concurrency constraint of 4 workers
    @endcode
    */
    explicit Semaphore(size_t max_workers);

    /**
    @brief queries the counter value (not thread-safe during the run)
    */
    size_t count() const;

  private:

    std::mutex _mtx;

    size_t _counter;

    std::vector<Node*> _waiters;

    bool _try_acquire_or_wait(Node*);
    bool _try_acquire_or_wait_pred(Node*);

    bool _try_acquire_or_wait_opt(Node*);
    bool _try_acquire_or_wait_pred_opt(Node*);

    std::vector<Node*> _release();
    void _release(Node*);
};

inline Semaphore::Semaphore(size_t max_workers) :
  _counter(max_workers) {
}

inline bool Semaphore::_try_acquire_or_wait(Node* me) {
  std::lock_guard<std::mutex> lock(_mtx);
  if(_counter > 0) {
    --_counter;
    return true;
  }
  else {
    _waiters.push_back(me);
    return false;
  }
}

inline bool Semaphore::_try_acquire_or_wait_pred(Node* me) {
  std::lock_guard<std::mutex> lock(_mtx);
  if(_counter > 0) {
    --_counter;
    return true;
  } else {
    return false;
  }
}

inline bool Semaphore::_try_acquire_or_wait_opt(Node* me) {
  if (_mtx.try_lock()) {
    if(_counter > 0) {
      --_counter;
      _mtx.unlock();
      return true;
    }
    else {
      _waiters.push_back(me);
      _mtx.unlock();
      return false;
    }
  }
  return false;
}

inline bool Semaphore::_try_acquire_or_wait_pred_opt(Node* me) {
  if (_mtx.try_lock()) {
    if(_counter > 0) {
      --_counter;
      _mtx.unlock();
      return true;
    } else {
      _mtx.unlock();
      return false;
    }
    return false;
  }
  return false;
}

inline std::vector<Node*> Semaphore::_release() {
  std::lock_guard<std::mutex> lock(_mtx);
  ++_counter;
  std::vector<Node*> r{std::move(_waiters)};
  return r;
}

inline void Semaphore::_release(Node* me) {
  std::lock_guard<std::mutex> lock(_mtx);
  for (uint32_t i = 0; i < _waiters.size(); ++i) {
    if (_waiters[i] == me) {
      _waiters.erase(_waiters.begin() + i);
      break;
    }
  }
  ++_counter;
}

inline size_t Semaphore::count() const {
  return _counter;
}

}  // end of namespace tf. ---------------------------------------------------

