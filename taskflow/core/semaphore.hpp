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

  public:

    /**
    @brief constructs a semaphore with the given counter

    A semaphore creates a constraint that limits the maximum concurrency,
    i.e., the number of workers, in a set of tasks.

    @code{.cpp}
    tf::Semaphore semaphore(4);  // a semaphore initialized with 4
    @endcode
    */
    explicit Semaphore(size_t count) : _count(count) {
    }

    /**
    @brief queries the counter value (not thread-safe during the run)

    @param memory_order the memory order of this load (default std::memory_order_relaxed)
    */
    size_t count(std::memory_order memory_order = std::memory_order_relaxed) const {
      return _count.load(memory_order);
    }

    /**
    @brief tries to atomically decrement the internal counter by @c 1 if it is greater than @c 0

    @return @c true if it decremented the internal counter, otherwise @c false

    */
    bool try_acquire() {
      auto old = _count.load(std::memory_order_acquire);
      if(old == 0) {
        return false;
      }
      return _count.compare_exchange_strong(
        old, old - 1, std::memory_order_acquire, std::memory_order_relaxed
      );
    }

    /**
    @brief tries to atomically increment the internal counter by @c n

    @param n the value by which the internal counter will be incremented
    @return @c true if it decremented the internal counter, otherwise @c false

    */
    void release(size_t n = 1) {
      _count.fetch_add(n, std::memory_order_release);
    }

  private:

    std::atomic<size_t> _count;
};

template <typename I,
  std::enable_if_t<std::is_same_v<deref_t<I>, Semaphore>, void> * = nullptr
>
bool try_acquire(I begin, I end) {
  I ptr = begin;
  for(; ptr != end; ptr++) {
    if(ptr->try_acquire() == false) {
      for(I ptr2 = begin; ptr2 != ptr; ptr2++) {
        ptr2->release();
      }
      return false;
    }
  }
  return true;
}

template<typename... S, 
  std::enable_if_t<all_same_v<Semaphore, std::decay_t<S>...>, void>* = nullptr
>
bool try_acquire(S&&... semaphores) {
  constexpr size_t N = sizeof...(S);
  std::array<Semaphore*, N> items { std::addressof(semaphores)... };
  size_t i = 0;
  for(; i < N; i++) {
    if(items[i]->try_acquire() == false) {
      for(size_t j=0; j<i; j++) {
        items[j]->release();
      }
      return false;
    }
  }
  return true;
}

}  // end of namespace tf. ---------------------------------------------------

