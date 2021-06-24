#pragma once

#include "../core/taskflow.hpp"

namespace tf {

class InfiniteLoopSanitizer {

  struct ilsGraph {

  };

  public:

    InfiniteLoopSanitizer(const Taskflow& taskflow) : _taskflow {taskflow} {
    }

    std::vector<std::vector<Task>> operator ()(std::ostream&) {
      
      // copy _taskflow to _graph
      return {};
    }

  private:

    const Taskflow& _taskflow;

    ilsGraph _graph;
};


} 
