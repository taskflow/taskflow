#pragma once

#include "../core/taskflow.hpp"

namespace tf {

class NonReachableSanitizer {

  struct nrsNode {
    bool reachable {false};
    Node* node {nullptr};
  };
  
  struct nrsGraph {
    std::vector<nrsNode> nodes;
  };

  public:

    NonReachableSanitizer(const Taskflow& taskflow) : _taskflow {taskflow} {
    }

    std::vector<Task> operator ()(std::ostream& ) {
      
      // copy _taskflow to _graph
      return {};
    }

  private:

    const Taskflow& _taskflow;

    nrsGraph _graph;

};

} // end of namespace tf 
