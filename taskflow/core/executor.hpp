#pragma once

include "task_scheduler.hpp"

namespace tf {

class Executor : public TaskScheduler
{
  public:
    explicit Executor(size_t N = std::thread::hardware_concurrency());
};

inline Executor::Executor(size_t N) : TaskScheduler(N) {}

}

