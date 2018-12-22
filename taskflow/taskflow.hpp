#pragma once

#include "threadpool/threadpool.hpp"
#include "graph/basic_taskflow.hpp"

namespace tf {

using Taskflow = BasicTaskflow<WorkStealingThreadpool>;

};  // end of namespace tf. ---------------------------------------------------





