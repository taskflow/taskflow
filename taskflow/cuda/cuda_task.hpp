#pragma once

#include "cuda_graph.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// cudaTask Types
// ----------------------------------------------------------------------------

/**
@enum cudaTaskType

@brief enumeration of all cudaTask types
*/
enum cudaTaskType {
  CUDA_NOOP_TASK   = cudaNode::CUDA_NOOP_TASK,
  CUDA_MEMSET_TASK = cudaNode::CUDA_MEMSET_TASK,
  CUDA_MEMCPY_TASK = cudaNode::CUDA_MEMCPY_TASK,
  CUDA_KERNEL_TASK = cudaNode::CUDA_KERNEL_TASK,
  CUDA_SUBFLOW_TASK = cudaNode::CUDA_SUBFLOW_TASK,
  CUDA_CAPTURE_TASK = cudaNode::CUDA_CAPTURE_TASK
};

/**
@brief convert a cuda_task type to a human-readable string
*/
inline const char* cuda_task_type_to_string(cudaTaskType type) {

  const char* val;

  switch(type) {
    case CUDA_NOOP_TASK:    val = "cuda_noop";    break;
    case CUDA_MEMSET_TASK:  val = "cuda_memset";  break;
    case CUDA_MEMCPY_TASK:  val = "cuda_memcpy";  break;
    case CUDA_KERNEL_TASK:  val = "cuda_kernel";  break;
    case CUDA_SUBFLOW_TASK: val = "cuda_subflow"; break;
    case CUDA_CAPTURE_TASK: val = "cuda_capture"; break;
    default:                val = "undefined";    break;
  }
  return val;
}

// ----------------------------------------------------------------------------
// cudaTask 
// ----------------------------------------------------------------------------

/**
@class cudaTask

@brief handle to a node in a cudaGraph
*/
class cudaTask {

  friend class cudaFlow;

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

    @tparam Ts... parameter pack

    @param tasks one or multiple tasks

    @return @c *this
    */
    template <typename... Ts>
    cudaTask& precede(Ts&&... tasks);
    
    /**
    @brief adds precedence links from other tasks to this

    @tparam Ts... parameter pack

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
    @brief queries if the task is associated with a cudaNode
    */
    bool empty() const;

    /**
    @brief queries the task type
    */
    cudaTaskType type() const;

    /**
    @brief dumps the task through an output stream
    */
    void dump(std::ostream& os) const;

  private:
    
    cudaTask(cudaNode*);

    cudaNode* _node {nullptr};
};

// Constructor
inline cudaTask::cudaTask(cudaNode* node) : _node {node} {
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

// Function: type
inline cudaTaskType cudaTask::type() const {
  return static_cast<cudaTaskType>(_node->_handle.index());
}

// Procedure: dump
inline void cudaTask::dump(std::ostream& os) const {
  os << "cudaTask ";
  if(_node->_name.empty()) os << _node;
  else os << _node->_name;
  os << " [type=" << cuda_task_type_to_string(type()) << ']';
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



