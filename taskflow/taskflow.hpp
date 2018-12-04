#pragma once

#include "graph/basic_taskflow.hpp"

namespace tf {

//using Taskflow = BasicTaskflow<SpeculativeThreadpool>;
using Taskflow = BasicTaskflow<PrivatizedThreadpool>;

};  // end of namespace tf. ---------------------------------------------------





