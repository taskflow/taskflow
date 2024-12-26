#pragma once

#include "../taskflow.hpp"

namespace tf {

// ----------------------------------------------------------------------------

/**
@private
*/
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

/**
 * @brief creates a module task using the given target
 * 
 * @tparam T Type of the target object, which must define the method `tf::Graph& graph()`.
 * @param target The target object used to create the module task.
 * @return module task that can be used by %Taskflow or asynchronous tasking.
 * 
 *
 * This example demonstrates how to create and launch multiple taskflows in parallel 
 * using asynchronous tasking:
 *
 * @code{.cpp}
 * tf::Executor executor;
 *
 * tf::Taskflow A;
 * tf::Taskflow B;
 * tf::Taskflow C;
 * tf::Taskflow D;
 *
 * A.emplace([](){ printf("Taskflow A\n"); }); 
 * B.emplace([](){ printf("Taskflow B\n"); }); 
 * C.emplace([](){ printf("Taskflow C\n"); }); 
 * D.emplace([](){ printf("Taskflow D\n"); }); 
 *
 * // launch the four taskflows using asynchronous tasking
 * executor.async(tf::make_module_task(A));
 * executor.async(tf::make_module_task(B));
 * executor.async(tf::make_module_task(C));
 * executor.async(tf::make_module_task(D));
 * executor.wait_for_all();  
 * @endcode
 *
 * The module task maker, tf::make_module_task, is basically the same as tf::Taskflow::composed_of 
 * but provides a more generic interface that can be used beyond %Taskflow.
 * For instance, the following two approaches achieve the same functionality.
 *
 * @code{.cpp}
 * // approach 1: composition using composed_of
 * tf::Task m1 = taskflow1.composed_of(taskflow2);
 * 
 * // approach 2: composition using make_module_task
 * tf::Task m1 = taskflow1.emplace(tf::make_module_task(taskflow2));
 * @endcode
 * 
 * @attention
 * Users are responsible for ensuring that the given target remains valid throughout its execution. 
 * The executor does not assume ownership of the target object.
 */
template <typename T>
auto make_module_task(T&& target) {
  return Algorithm::make_module_task(std::forward<T>(target));
}

}  // end of namespact tf -----------------------------------------------------
