#pragma once

#include "task_scheduler.hpp"

/**
@file executor.hpp
@brief executor include file
*/

namespace tf
{

// ----------------------------------------------------------------------------
// Executor Definition
// ----------------------------------------------------------------------------


/** @class Executor

@brief execution interface for running a taskflow graph

An executor object manages a set of worker threads to run taskflow(s)
which are scheduled by the base TaskScheduler class.

*/

class Executor : public TaskScheduler
{
  public:
  
    /**
    @brief constructs the task scheduler with N worker threads
    */
    explicit Executor(size_t N = std::thread::hardware_concurrency());

    ~Executor();

  private:

    std::vector<std::thread> _threads;

    void _spawn_thread_pool(size_t);

};

// Constructor
inline Executor::Executor(size_t N) : TaskScheduler() {
    _spawn_thread_pool(N);
    _configure();
}

inline Executor::~Executor() {
    _shutdown();

    for (auto& t : _threads) {
        t.join();
    }
}

// Procedure: _spawn
inline void Executor::_spawn_thread_pool(size_t N) {

  _threads.resize(N);
  for(size_t id=0; id<N; ++id) {

    register_worker(_threads[id]);
  }
}
}
