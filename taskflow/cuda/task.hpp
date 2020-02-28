#pragma once

#include "graph.hpp"

namespace tf {

/**
@class Task

@brief task handle to a node in a cudaGraph
*/
class cudaTask {

  friend class cudaFlow;

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
    @brief assigns a name to the task

    @param name a @std_string acceptable string

    @return @c *this
    */
    cudaTask& name(const std::string& name);
    
    /**
    @brief queries the name of the task
    */
    const std::string& name() const;

  private:
    
    cudaTask(cudaNode*);

    cudaNode* _node {nullptr};

    template <typename T>
    void _precede(T&&);

    template <typename T, typename... Ts>
    void _precede(T&&, Ts&&...);
};

// Constructor
inline cudaTask::cudaTask(cudaNode* node) : _node {node} {
}

// Function: precede
template <typename... Ts>
cudaTask& cudaTask::precede(Ts&&... tasks) {
  _precede(std::forward<Ts>(tasks)...);
  return *this;
}

// Procedure: precede
template <typename T>
void cudaTask::_precede(T&& other) {
  _node->_precede(other._node);
}

// Procedure: _precede
template <typename T, typename... Ts>
void cudaTask::_precede(T&& task, Ts&&... others) {
  _precede(std::forward<T>(task));
  _precede(std::forward<Ts>(others)...);
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

}  // end of namespace tf -----------------------------------------------------
