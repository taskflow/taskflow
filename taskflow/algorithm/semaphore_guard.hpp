#pragma once

#include "../core/graph.hpp"
#include "../core/semaphore.hpp"

/**
@file semaphore.hpp
@brief semaphore guard include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// SemaphoreGuard
// ----------------------------------------------------------------------------

/**
@class SemaphoreGuard

@brief class to implement a RAII semaphore guard to help developer control
tf::Semaphore ownership within a scope, releasing ownership in the
destructor. It is recommended to use it to make sure lock is release in a
properly order to avoid deadlocking.

The usage of SemaphoreGuard is pretty like std::lock_guard's. One difference is
that user should pass tf::Runtime reference as a parameter to tf::SemaphoreGuard
due to the implementation need.

@code{.cpp}
tf::Semaphore se(1);
tf::Executor executor(2);
tf::Taskflow taskflow;
int32_t count = 0;
auto t1 = taskflow.emplace([&](tf::Runtime& rt) {
                    tf::SemaphoreGuard gd(rt, se);
                    --count;
                  })
auto t2 = taskflow.emplace([&](tf::Runtime& rt) {
                    tf::SemaphoreGuard gd(rt, se);
                    ++count;
                  });
executor.run(taskflow);
executor.wait_for_all();
@endcode

*/
class SemaphoreGuard {
  public:
  explicit SemaphoreGuard(tf::Runtime& rt, tf::Semaphore& se) :
          _rt(rt), _se(se) {
    _rt.acquire(_se);
  }

  ~SemaphoreGuard() {
    _rt.release(_se);
  }

  SemaphoreGuard(const SemaphoreGuard&) = delete;
  SemaphoreGuard& operator=(const SemaphoreGuard&) = delete;

  private:
  tf::Runtime& _rt;
  tf::Semaphore& _se;
};

} // end of namespace tf. ---------------------------------------------------
