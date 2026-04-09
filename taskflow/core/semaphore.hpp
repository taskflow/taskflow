#pragma once

#include <atomic>
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

### Implementation notes (hybrid lock-free / mutex design)

The acquire hot path — taking a token when the count is positive — is
lock-free: a CAS loop on the atomic counter suffices and the mutex is never
touched.  This is the common case for lightly-loaded semaphores.

The cold path — when the count hits zero — acquires the mutex only to protect
the waiter list and to perform a safe double-check.  Under the lock we use a
CAS loop (not fetch_sub) to decrement, so that a concurrent fast-path CAS
that already claimed the token is detected and we park instead of underflowing
the counter to SIZE_MAX.

_release performs the increment and the waiter drain both under the mutex.
This is the critical correctness invariant: if the increment were done outside
the lock, a concurrent fast-path acquire could steal the token in the window
between the increment and the lock acquisition, then a released waiter would
also get scheduled — both would see the semaphore as "held", violating the
concurrency limit.  By holding the lock for both the increment and the drain,
we guarantee that exactly one token is either added to the counter (no
waiters) or handed directly to the rescheduled waiter set (waiters present),
with no window for a fast-path steal in between.
*/
class Semaphore {

  friend class Node;
  friend class Executor;

  public:

    /**
    @brief constructs a default semaphore

    A default semaphore has the value of zero. Users can call tf::Semaphore::reset
    to reassign a new value to the semaphore.
    */
    Semaphore() = default;

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
    size_t value() const noexcept;

    /**
    @brief queries the maximum allowable value of this semaphore
    */
    size_t max_value() const noexcept;

    /**
    @brief resets the semaphores to a clean state
    */
    void reset();

    /**
    @brief resets the semaphores to a clean state with the given new maximum value
    */
    void reset(size_t new_max_value);

  private:

    // _max_value is set at construction / reset and never modified concurrently.
    size_t _max_value{0};

    // Hot-path: lock-free token counter.
    // Decremented on acquire via CAS loop (fast path, no lock) or under _mtx
    // (slow path, double-check).  Incremented on release under _mtx.
    alignas(TF_CACHELINE_SIZE) std::atomic<size_t> _cur_value{0};

    // Guards _cur_value for the release increment and the _waiters list.
    // Also taken in the acquire slow path for a double-check.
    mutable std::mutex _mtx;

    // Parked nodes waiting for a token.  Protected by _mtx.
    SmallVector<Node*> _waiters;

    bool _try_acquire_or_wait(Node*);

    void _release(SmallVector<Node*>&);
};

// ---------------------------------------------------------------------------
// Semaphore — inline method definitions
// ---------------------------------------------------------------------------

inline Semaphore::Semaphore(size_t max_value) :
  _max_value(max_value),
  _cur_value(max_value) {
}

inline size_t Semaphore::max_value() const noexcept {
  return _max_value;
}

inline size_t Semaphore::value() const noexcept {
  return _cur_value.load(std::memory_order_relaxed);
}

// Procedure: _try_acquire_or_wait
//
// Lock-free fast path: CAS-decrement the counter if a token is available.
// acq_rel on success: the store (decrement) is release-ordered so other
// threads' acquire-loads see it; the load of the old value is
// acquire-ordered so this thread sees all stores that preceded the token
// being made available.  Returns immediately without touching the mutex in
// the common (uncontended) case.
//
// Slow path (mutex): entered when the fast-path CAS loop exhausts all tokens.
// Under the lock we perform a CAS loop rather than a plain fetch_sub.
// This is critical: a concurrent fast-path CAS may decrement the counter
// between our acquire-load and a hypothetical fetch_sub, causing the count
// to wrap to SIZE_MAX.  The CAS loop detects this — if it fails, we reload
// the updated count and retry; if the refreshed count is 0, we park.
//
// Double-check-after-lock: if _release incremented the counter before we
// took the lock, the acquire-load inside the lock will see the updated count
// and we return true without touching the waiter list.
inline bool Semaphore::_try_acquire_or_wait(Node* me) {

  // ── Lock-free fast path ─────────────────────────────────────────────────
  size_t cur = _cur_value.load(std::memory_order_relaxed);
  while(cur > 0) {
    if(_cur_value.compare_exchange_weak(cur, cur - 1,
          std::memory_order_acq_rel,
          std::memory_order_relaxed)) {
      return true;   // acquired — mutex never touched
    }
    // cur refreshed by compare_exchange_weak on failure; retry
  }

  // ── Slow path: take the lock and double-check ───────────────────────────
  std::lock_guard<std::mutex> lock(_mtx);

  // Acquire-load: see any _release increment that happened before we took
  // the lock (the fetch_add in _release uses memory_order_release).
  cur = _cur_value.load(std::memory_order_acquire);
  while(cur > 0) {
    if(_cur_value.compare_exchange_weak(cur, cur - 1,
          std::memory_order_acq_rel,
          std::memory_order_relaxed)) {
      return true;
    }
    // cur refreshed; keep retrying
  }

  // Count is 0 under the lock: park this node in the waiter list.
  _waiters.push_back(me);
  return false;
}

// Procedure: _release
//
// The counter increment and the waiter drain both happen under the mutex.
//
// Why hold the lock for the increment?
//   If the increment were done outside the lock, a fast-path acquire could
//   steal the newly available token in the window between the increment and
//   the lock acquisition.  When the lock is later taken and all waiters are
//   drained, those waiters get scheduled.  They re-try _try_acquire_or_wait,
//   find the counter at 0 (the fast-path stole it), and park again — but the
//   fast-path acquirer and the re-scheduled waiters are briefly both "in
//   flight", which violates the semaphore's concurrency guarantee.
//
//   Holding the lock for the increment closes this window: the fast path
//   cannot see the incremented count until _release has also finished
//   draining the waiter list, ensuring the token goes either to a waiter or
//   to the counter — never to both simultaneously.
//
// The woken nodes are appended into the caller-supplied SmallVector and
// scheduled by the executor, preserving the original batch-wake semantics.
// The woken nodes re-compete for the token via _try_acquire_or_wait when
// the executor re-invokes them.
inline void Semaphore::_release(SmallVector<Node*>& dst) {

  std::lock_guard<std::mutex> lock(_mtx);

  if(_cur_value.load(std::memory_order_relaxed) >= _max_value) {
    TF_THROW("can't release the semaphore more than its maximum value: ",
             _max_value);
  }

  // Increment under the lock — see design note above.
  _cur_value.fetch_add(1, std::memory_order_release);

  if(dst.empty()) {
    dst.swap(_waiters);
  }
  else {
    dst.reserve(dst.size() + _waiters.size());
    dst.insert(dst.end(), _waiters.begin(), _waiters.end());
    _waiters.clear();
  }
}

inline void Semaphore::reset() {
  std::lock_guard<std::mutex> lock(_mtx);
  _cur_value.store(_max_value, std::memory_order_relaxed);
  _waiters.clear();
}

inline void Semaphore::reset(size_t new_max_value) {
  std::lock_guard<std::mutex> lock(_mtx);
  _cur_value.store((_max_value = new_max_value), std::memory_order_relaxed);
  _waiters.clear();
}

}  // end of namespace tf. ---------------------------------------------------
