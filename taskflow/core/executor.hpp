#pragma once

include "task_scheduler.hpp"

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
};

// Constructor
inline Executor::Executor(size_t N) : TaskScheduler(N) {}

}

