#pragma once

#include "executor/executor.hpp"
#include "graph/basic_taskflow.hpp"

namespace tf {

using Taskflow = BasicTaskflow<WorkStealingExecutor>;

}  // end of namespace tf. ---------------------------------------------------





