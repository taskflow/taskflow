#pragma once

#include "cuda_graph.hpp"

/**
@file cuda_task.hpp
@brief cudaTask include file
*/

namespace tf {



// ----------------------------------------------------------------------------
// cudaTask
// ----------------------------------------------------------------------------

/**
@class cudaTask

@brief class to create a task handle over an internal node of a %cudaFlow graph
*/
class cudaTask {

  friend class cudaFlow;
  friend class cudaFlowCapturer;
  friend class cudaFlowCapturerBase;

  friend std::ostream& operator << (std::ostream&, const cudaTask&);

  public:

    /**
    @brief constructs an empty cudaTask
    */
    cudaTask() = default;

    /**
    @brief copy-constructs a cudaTask
    */
    cudaTask(const cudaTask&) = default;

    /**
    @brief copy-assigns a cudaTask
    */
    cudaTask& operator = (const cudaTask&) = default;

    /**
    @brief adds precedence links from this to other tasks

    @tparam Ts parameter pack

    @param tasks one or multiple tasks

    @return @c *this
    */
    template <typename... Ts>
    cudaTask& precede(Ts&&... tasks);

    /**
    @brief adds precedence links from other tasks to this

    @tparam Ts parameter pack

    @param tasks one or multiple tasks

    @return @c *this
    */
    template <typename... Ts>
    cudaTask& succeed(Ts&&... tasks);

    /**
    @brief queries the number of successors
    */
    size_t num_successors() const;

    /**
    @brief queries the number of dependents
    */
    size_t num_predecessors() const;

    /**
    @brief queries the type of this task
    */
    auto type() const;

    /**
    @brief dumps the task through an output stream

    @param os an output stream target
    */
    void dump(std::ostream& os) const;

  private:

    cudaTask(cudaGraph_t, cudaGraphNode_t);
    
    cudaGraph_t _native_graph {nullptr};
    cudaGraphNode_t _native_node {nullptr};
};

// Constructor
inline cudaTask::cudaTask(cudaGraph_t native_graph, cudaGraphNode_t native_node) : 
  _native_graph {native_graph}, _native_node  {native_node} {
}
  
// Function: precede
template <typename... Ts>
cudaTask& cudaTask::precede(Ts&&... tasks) {
  (
    cudaGraphAddDependencies(
      _native_graph, &_native_node, &(tasks._native_node), 1
    ), ...
  );
  return *this;
}

// Function: succeed
template <typename... Ts>
cudaTask& cudaTask::succeed(Ts&&... tasks) {
  (tasks.precede(*this), ...);
  return *this;
}

//// Function: num_successors
//inline size_t cudaTask::num_successors() const {
//  return _node->_successors.size();
//}

// Function: num_predecessors
inline size_t cudaTask::num_predecessors() const {
  size_t num_predecessors {0};
  cudaGraphNodeGetDependencies(_native_node, nullptr, &num_predecessors);
  return num_predecessors;
}

// Function: num_successors
inline size_t cudaTask::num_successors() const {
  size_t num_successors {0};
  cudaGraphNodeGetDependentNodes(_native_node, nullptr, &num_successors);
  return num_successors;
}

// Function: type
inline auto cudaTask::type() const {
  cudaGraphNodeType type;
  cudaGraphNodeGetType(_native_node, &type);
  return type;
}

// Function: dump
inline void cudaTask::dump(std::ostream& os) const {
  os << "cudaTask [type=" << to_string(type()) << ']';
}

//// Function: for_each_successor
//template <typename V>
//void cudaTask::for_each_successor(V&& visitor) const {
//  for(size_t i=0; i<_node->_successors.size(); ++i) {
//    visitor(cudaTask(_node->_successors[i]));
//  }
//}
//
//// Function: for_each_dependent
//template <typename V>
//void cudaTask::for_each_dependent(V&& visitor) const {
//  for(size_t i=0; i<_node->_dependents.size(); ++i) {
//    visitor(cudaTask(_node->_dependents[i]));
//  }
//}

// ----------------------------------------------------------------------------
// global ostream
// ----------------------------------------------------------------------------

/**
@brief overload of ostream inserter operator for cudaTask
*/
inline std::ostream& operator << (std::ostream& os, const cudaTask& ct) {
  ct.dump(os);
  return os;
}

}  // end of namespace tf -----------------------------------------------------



