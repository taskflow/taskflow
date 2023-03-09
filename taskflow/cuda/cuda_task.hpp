#pragma once

#include "cuda_graph.hpp"

/**
@file cuda_task.hpp
@brief cudaTask include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// cudaTask Types
// ----------------------------------------------------------------------------

/**
@enum cudaTaskType

@brief enumeration of all %cudaTask types
*/
enum class cudaTaskType : int {
  /** @brief empty task type */
  EMPTY = 0,
  /** @brief host task type */
  HOST,
  /** @brief memory set task type */
  MEMSET,
  /** @brief memory copy task type */
  MEMCPY,
  /** @brief memory copy task type */
  KERNEL,
  /** @brief subflow (child graph) task type */
  SUBFLOW,
  /** @brief capture task type */
  CAPTURE,
  /** @brief undefined task type */
  UNDEFINED
};

/**
@brief convert a cuda_task type to a human-readable string
*/
constexpr const char* to_string(cudaTaskType type) {
  switch(type) {
    case cudaTaskType::EMPTY:   return "empty";
    case cudaTaskType::HOST:    return "host";
    case cudaTaskType::MEMSET:  return "memset";
    case cudaTaskType::MEMCPY:  return "memcpy";
    case cudaTaskType::KERNEL:  return "kernel";
    case cudaTaskType::SUBFLOW: return "subflow";
    case cudaTaskType::CAPTURE: return "capture";
    default:                    return "undefined";
  }
}

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
    @brief assigns a name to the task

    @param name a @std_string acceptable string

    @return @c *this
    */
    cudaTask& name(const std::string& name);

    /**
    @brief queries the name of the task
    */
    const std::string& name() const;

    /**
    @brief queries the number of successors
    */
    size_t num_successors() const;

    /**
    @brief queries the number of dependents
    */
    size_t num_dependents() const;

    /**
    @brief queries if the task is associated with a cudaFlowNode
    */
    bool empty() const;

    /**
    @brief queries the task type
    */
    cudaTaskType type() const;

    /**
    @brief dumps the task through an output stream

    @tparam T output stream type with insertion operator (<<) defined
    @param ostream an output stream target
    */
    template <typename T>
    void dump(T& ostream) const;

    /**
    @brief applies an visitor callable to each successor of the task
    */
    template <typename V>
    void for_each_successor(V&& visitor) const;

    /**
    @brief applies an visitor callable to each dependents of the task
    */
    template <typename V>
    void for_each_dependent(V&& visitor) const;

  private:

    cudaTask(cudaFlowNode*);

    cudaFlowNode* _node {nullptr};
};

// Constructor
inline cudaTask::cudaTask(cudaFlowNode* node) : _node {node} {
}

// Function: precede
template <typename... Ts>
cudaTask& cudaTask::precede(Ts&&... tasks) {
  (_node->_precede(tasks._node), ...);
  return *this;
}

// Function: succeed
template <typename... Ts>
cudaTask& cudaTask::succeed(Ts&&... tasks) {
  (tasks._node->_precede(_node), ...);
  return *this;
}

// Function: empty
inline bool cudaTask::empty() const {
  return _node == nullptr;
}

// Function: name
inline cudaTask& cudaTask::name(const std::string& name) {
  _node->_name = name;
  return *this;
}

// Function: name
inline const std::string& cudaTask::name() const {
  return _node->_name;
}

// Function: num_successors
inline size_t cudaTask::num_successors() const {
  return _node->_successors.size();
}

// Function: num_dependents
inline size_t cudaTask::num_dependents() const {
  return _node->_dependents.size();
}

// Function: type
inline cudaTaskType cudaTask::type() const {
  switch(_node->_handle.index()) {
    case cudaFlowNode::EMPTY:   return cudaTaskType::EMPTY;
    case cudaFlowNode::HOST:    return cudaTaskType::HOST;
    case cudaFlowNode::MEMSET:  return cudaTaskType::MEMSET;
    case cudaFlowNode::MEMCPY:  return cudaTaskType::MEMCPY;
    case cudaFlowNode::KERNEL:  return cudaTaskType::KERNEL;
    case cudaFlowNode::SUBFLOW: return cudaTaskType::SUBFLOW;
    case cudaFlowNode::CAPTURE: return cudaTaskType::CAPTURE;
    default:                return cudaTaskType::UNDEFINED;
  }
}

// Procedure: dump
template <typename T>
void cudaTask::dump(T& os) const {
  os << "cudaTask ";
  if(_node->_name.empty()) os << _node;
  else os << _node->_name;
  os << " [type=" << to_string(type()) << ']';
}

// Function: for_each_successor
template <typename V>
void cudaTask::for_each_successor(V&& visitor) const {
  for(size_t i=0; i<_node->_successors.size(); ++i) {
    visitor(cudaTask(_node->_successors[i]));
  }
}

// Function: for_each_dependent
template <typename V>
void cudaTask::for_each_dependent(V&& visitor) const {
  for(size_t i=0; i<_node->_dependents.size(); ++i) {
    visitor(cudaTask(_node->_dependents[i]));
  }
}

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



