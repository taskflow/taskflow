#pragma once

#include "../taskflow.hpp"

namespace tf {

// ----------------------------------------------------------------------------

template <typename T>
auto Algorithm::make_module_task(T&& target) {
  return [&target=std::forward<T>(target)](tf::Runtime& rt){
    auto& graph = target.graph();
    if(graph.empty()) {
      return;
    }
    PreemptionGuard preemption_guard(rt);
    rt._executor._schedule_graph_with_parent(
      rt._worker, graph.begin(), graph.end(), rt._parent, NSTATE::NONE
    );
  };
}

// ----------------------------------------------------------------------------

template <typename T>
auto make_module_task(T&& target) {
  return Algorithm::make_module_task(std::forward<T>(target));
}

}  // end of namespact tf -----------------------------------------------------
