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

for(size_t i=0; i<1000; i++) {
  taskflow.emplace([&](tf::Runtime& rt){ 
    rt.acquire(semaphore);
    std::cout << "critical section here (one worker at any time)\n"; 
    critical_section();
    rt.release(semaphore);
  });
}

executor.run(taskflow).wait();
@endcode

The above example creates a taskflow of 1000 independent tasks while
only one worker will run @c critical_section at any time
due to the semaphore constraint.
This arrangement limits the parallelism of @c critical_section to 
just one.

@note %Taskflow use a non-blocking algorithm to implement the acquisition
of semaphores and thus is deadlock-free.

*/
class Semaphore {

  friend class Node;

  public:

    /**
    @brief constructs a default semaphore with count equal to zero

    Application can use tf::Semaphore::reset to reset the counter of 
    the semaphore later.
    */
    Semaphore() : _count(0) { };

    /**
    @brief constructs a semaphore with the given count

    A semaphore creates a constraint that limits the maximum concurrency,
    i.e., the number of workers, in a set of tasks.

    @code{.cpp}
    tf::Semaphore semaphore(4);  // a semaphore initialized with 4
    @endcode
    */
    explicit Semaphore(size_t count) : _count(count) {
    }

    /**
    @brief queries the current value of the associated counter

    @param memory_order the memory order of this load (default std::memory_order_seq_cst)

    Queries the current value of the associated counter.
    */
    size_t count(std::memory_order memory_order = std::memory_order_seq_cst) const {
      return _count.load(memory_order);
    }

    /**
    @brief tries to atomically decrement the internal counter by @c 1 if it is greater than @c 0

    @return @c true if it decremented the internal counter, otherwise @c false

    Tries to atomically decrement the internal counter by @c 1. If the operation succeeds,
    returns @c true, otherwise @c false.
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
    @brief atomically increment the internal counter by @c n

    @param n the value by which the internal counter will be incremented
    
    The release operation always succeeds as it simply increments 
    the counter of this semaphore.
    */
    void release(size_t n = 1) {
      _count.fetch_add(n, std::memory_order_release);
    }
    
    /**
    @brief resets the semaphore to the given count

    @param count the new count value
    @param memory_order memory order to which this operation will be applied
                        (default std::memory_order_seq_cst)
    
    @note
    Calling tf::Semaphore::reset will immediately change the underlying
    counter to the given @c count value, regardless other threads acquiring 
    or releasing the semaphore.
    */
    void reset(size_t count, std::memory_order memory_order = std::memory_order_seq_cst) {
      _count.store(count, memory_order);
    }

  private:

    std::atomic<size_t> _count;
};

/**
@brief tries to acquire all semaphores in the specified range

@tparam I iterator type
@param first iterator to the beginning (inclusive)
@param last iterator to the end (exclusive)

Tries to acquire all semaphores in the specified range.

@return @c true if all semaphores are acquired, otherwise @c false
*/
template <typename I,
  std::enable_if_t<std::is_same_v<deref_t<I>, Semaphore>, void> * = nullptr
>
bool try_acquire(I first, I last) {
  // Ideally, we should use a better deadlock-avoidance algorithm but
  // in practice the number of semaphores is small and
  // tf::Semaphore does not provide blocking require. Hence, we are 
  // mostly safe here. This is similar to the GCC try_lock implementation:
  // https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/std/mutex
  for(I ptr = first; ptr != last; ptr++) {
    if(ptr->try_acquire() == false) {
      for(I ptr2 = first; ptr2 != ptr; ptr2++) {
        ptr2->release();
      }
      return false;
    }
  }
  return true;
}

/**
@brief tries to acquire all semaphores

@param semaphores semaphores to acquire

Tries to acquire all the semaphores.

@return @c true if all semaphores are acquired, otherwise @c false
*/
template<typename... S, 
  std::enable_if_t<all_same_v<Semaphore, std::decay_t<S>...>, void>* = nullptr
>
bool try_acquire(S&&... semaphores) {
  // Ideally, we should use a better deadlock-avoidance algorithm but
  // in practice the number of semaphores is small and
  // tf::Semaphore does not provide blocking require. Hence, we are 
  // mostly safe here. This is similar to the GCC try_lock implementation:
  // https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/std/mutex
  constexpr size_t N = sizeof...(S);
  std::array<Semaphore*, N> items { std::addressof(semaphores)... };
  for(size_t i=0; i<N; i++) {
    if(items[i]->try_acquire() == false) {
      for(size_t j=0; j<i; j++) {
        items[j]->release();
      }
      return false;
    }
  }
  return true;
}

/**
@brief tries to acquire all semaphores in the specified range

@tparam I iterator type
@param first iterator to the beginning (inclusive)
@param last iterator to the end (exclusive)

Releases all the semaphores in the given range.
*/
template <typename I,
  std::enable_if_t<std::is_same_v<deref_t<I>, Semaphore>, void> * = nullptr
>
void release(I first, I last) {
  std::for_each(first, last, [](tf::Semaphore& semaphore){ 
    semaphore.release();
  });
}

/**
@brief tries to acquire all semaphores

@param semaphores semaphores to release

Releases all the semaphores.
*/
template<typename... S, 
  std::enable_if_t<all_same_v<Semaphore, std::decay_t<S>...>, void>* = nullptr
>
void release(S&&... semaphores) {
  (semaphores.release(), ...);
}

}  // end of namespace tf. ---------------------------------------------------

