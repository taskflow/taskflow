#pragma once

#include "graph.hpp"

namespace tf {

/**
@class Task

@brief Handle to modify and access a task.

A Task is a wrapper of a node in a dependency graph. 
It provides a set of methods for users to access and modify the attributes of 
the task node,
preventing direct access to the internal data storage.

*/
class Task {

  friend class FlowBuilder;

  template <template<typename...> typename E> 
  friend class BasicTaskflow;

  public:
    
    /**
    @brief constructs an empty task
    */
    Task() = default;

    /**
    @brief constructs the task with the copy of the other task
    */
    Task(const Task& other);
    
    /**
    @brief constructs the task with the content of the other task using move semantics

    After the move, other is guaranteed to be empty
    */
    Task(Task&& other);
    
    /**
    @brief replaces the contents with a copy of the other task
    */
    Task& operator = (const Task&);
    
    /**
    @brief queries the name of the task
    */
    const std::string& name() const;
    
    /**
    @brief queries the number of successors of the task
    */
    size_t num_successors() const;

    /**
    @brief queries the number of predecessors of the task
    */
    size_t num_dependents() const;
    
    /**
    @brief assigns a name to the task

    @param name a @std_string acceptable string

    @return @c *this
    */
    Task& name(const std::string& name);

    /**
    @brief assigns a new callable object to the task

    @tparam C callable object type

    @param callable a callable object acceptable to @std_function

    @return @c *this
    */
    template <typename C>
    Task& work(C&& callable);
    
    /**
    @brief adds precedence links from this to other tasks

    @tparam Ts... parameter pack

    @param tasks... one or multiple tasks

    @return @c *this
    */
    template <typename... Ts>
    Task& precede(Ts&&... tasks);
    
    /**
    @brief adds precedence links from other tasks to this

    @tparam Ts parameter pack 

    @param tasks... one or multiple tasks

    @return @c *this
    */
    template <typename... Ts>
    Task& gather(Ts&&... tasks);
    
    /**
    @brief adds precedence links from other tasks to this

    @param tasks a vector of tasks

    @return @c *this
    */
    Task& gather(std::vector<Task>& tasks);

    /**
    @brief adds precedence links from other tasks to this

    @param tasks an initializer list of tasks

    @return @c *this
    */
    Task& gather(std::initializer_list<Task> tasks);
    
    template <typename... Bs>
    Task& broadcast(Bs&&...);
    
    Task& broadcast(std::vector<Task>&);
    Task& broadcast(std::initializer_list<Task>);

  private:
    
    Task(Node&);

    Node* _node {nullptr};

    template<typename S>
    void _broadcast(S&);

    template<typename S>
    void _gather(S&);
};

// Constructor
inline Task::Task(Node& t) : _node {&t} {
}

// Constructor
inline Task::Task(const Task& rhs) : _node {rhs._node} {
}

// Function: broadcast
template <typename... Bs>
Task& Task::broadcast(Bs&&... tgts) {
  (_node->precede(*(tgts._node)), ...);
  return *this;
}

// Procedure: _broadcast
template <typename S>
inline void Task::_broadcast(S& tgts) {
  for(auto& to : tgts) {
    _node->precede(*(to._node));
  }
}
      
// Function: broadcast
inline Task& Task::broadcast(std::vector<Task>& tgts) {
  _broadcast(tgts);
  return *this;
}

// Function: broadcast
inline Task& Task::broadcast(std::initializer_list<Task> tgts) {
  _broadcast(tgts);
  return *this;
}

// Function: precede
template <typename... Ts>
Task& Task::precede(Ts&&... tgts) {
  (_node->precede(*(tgts._node)), ...);
  return *this;
}

// Function: gather
template <typename... Bs>
Task& Task::gather(Bs&&... tgts) {
  (tgts._node->precede(*_node), ...);
  return *this;
}

// Procedure: _gather
template <typename S>
void Task::_gather(S& tgts) {
  for(auto& from : tgts) {
    from._node->precede(*_node);
  }
}

// Function: gather
inline Task& Task::gather(std::vector<Task>& tgts) {
  _gather(tgts);
  return *this;
}

// Function: gather
inline Task& Task::gather(std::initializer_list<Task> tgts) {
  _gather(tgts);
  return *this;
}

// Operator =
inline Task& Task::operator = (const Task& rhs) {
  _node = rhs._node;
  return *this;
}

// Constructor
inline Task::Task(Task&& rhs) : _node{rhs._node} { 
  rhs._node = nullptr; 
}

// Function: work
template <typename C>
inline Task& Task::work(C&& c) {
  _node->_work = std::forward<C>(c);
  return *this;
}

// Function: name
inline Task& Task::name(const std::string& name) {
  _node->_name = name;
  return *this;
}

// Function: name
inline const std::string& Task::name() const {
  return _node->_name;
}

// Function: num_dependents
inline size_t Task::num_dependents() const {
  return _node->_dependents.load(std::memory_order_relaxed);
}

// Function: num_successors
inline size_t Task::num_successors() const {
  return _node->_successors.size();
}

};  // end of namespace tf. ---------------------------------------------------
