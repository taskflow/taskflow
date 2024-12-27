#pragma once

#include <mutex>

#include "declarations.hpp"
#include "../utility/small_vector.hpp"

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

SmallVector<tf::Task> tasks {
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
  friend class Executor;

  public:

    /**
    @brief constructs a semaphore with the given value (i.e., counter)

    A semaphore creates a constraint that limits the maximum concurrency,
    i.e., the number of workers, in a set of tasks.

    @code{.cpp}
    tf::Semaphore semaphore(4);  // concurrency constraint of 4 workers
    @endcode
    */
    explicit Semaphore(size_t max_value);

    /**
    @brief queries the current counter value
    */
    size_t value() const;

    /**
    @brief queries the maximum allowable value of this semaphore
    */
    size_t max_value() const;

    /**
    @brief resets the semaphores to a clean state
    */
    void reset();
    
    /**
    @brief resets the semaphores to a clean state with the given new maximum value
    */
    void reset(size_t new_max_value);

  private:

    mutable std::mutex _mtx;
    
    size_t _max_value;
    size_t _cur_value;

    SmallVector<Node*> _waiters;

    bool _try_acquire_or_wait(Node*);

    SmallVector<Node*> _release();
};

inline Semaphore::Semaphore(size_t max_value) :
  _max_value(max_value),
  _cur_value(max_value) {
}

inline bool Semaphore::_try_acquire_or_wait(Node* me) {
  std::lock_guard<std::mutex> lock(_mtx);
  if(_cur_value > 0) {
    --_cur_value;
    return true;
  }
  else {
    _waiters.push_back(me);
    return false;
  }
}

inline SmallVector<Node*> Semaphore::_release() {
  std::lock_guard<std::mutex> lock(_mtx);
  if(_cur_value >= _max_value) {
    TF_THROW("can't release the semaphore more than its maximum value: ", _max_value);
  }
  ++_cur_value;
  SmallVector<Node*> r{std::move(_waiters)};
  return r;
}

inline size_t Semaphore::max_value() const {
  return _max_value; 
}

inline size_t Semaphore::value() const {
  std::lock_guard<std::mutex> lock(_mtx);
  return _cur_value;
}

inline void Semaphore::reset() {
  std::lock_guard<std::mutex> lock(_mtx);
  _cur_value = _max_value;
  _waiters.clear();
}

inline void Semaphore::reset(size_t new_max_value) {
  std::lock_guard<std::mutex> lock(_mtx);
  _cur_value = (_max_value = new_max_value);
  _waiters.clear();
}

}  // end of namespace tf. ---------------------------------------------------

