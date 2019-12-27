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
  friend class Taskflow;
  friend class TaskView;

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
    @brief replaces the contents with a copy of the other task
    */
    Task& operator = (const Task&);
    
    /**
    @brief replaces the contents with a null pointer
    */
    Task& operator = (std::nullptr_t);
    
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

    @param tasks one or multiple tasks

    @return @c *this
    */
    template <typename... Ts>
    Task& precede(Ts&&... tasks);
    
    /**
    @brief adds precedence links from this to others

    @param tasks a vector of tasks to precede

    @return @c *this
    */
    Task& precede(std::vector<Task>& tasks);

    /**
    @brief adds precedence links from this to others

    @param tasks an initializer list of tasks to precede

    @return @c *this
    */
    Task& precede(std::initializer_list<Task> tasks);
    
    /**
    @brief adds precedence links from other tasks to this

    @tparam Ts parameter pack 

    @param tasks one or multiple tasks

    @return @c *this
    */
    template <typename... Ts>
    Task& succeed(Ts&&... tasks);
    
    /**
    @brief adds precedence links from other tasks to this

    @param tasks a vector of tasks

    @return @c *this
    */
    Task& succeed(std::vector<Task>& tasks);

    /**
    @brief adds precedence links from other tasks to this

    @param tasks an initializer list of tasks

    @return @c *this
    */
    Task& succeed(std::initializer_list<Task> tasks);
    
    /**
    @brief adds precedence links from other tasks to this (same as succeed)

    @tparam Ts parameter pack 

    @param tasks one or multiple tasks

    @return @c *this
    */
    template <typename... Ts>
    Task& gather(Ts&&... tasks);
    
    /**
    @brief adds precedence links from other tasks to this (same as succeed)

    @param tasks a vector of tasks

    @return @c *this
    */
    Task& gather(std::vector<Task>& tasks);

    /**
    @brief adds precedence links from other tasks to this (same as succeed)

    @param tasks an initializer list of tasks

    @return @c *this
    */
    Task& gather(std::initializer_list<Task> tasks);

    /**
    @brief resets the task handle to null
    
    @return @c *this
    */
    Task& reset();

    /**
    @brief queries if the task handle points to a task node
    */
    bool empty() const;

    /**
    @brief queries if the task has a work assigned
    */
    bool has_work() const;

  private:
    
    Task(Node&);
    Task(Node*);

    Node* _node {nullptr};

    template <typename S>
    void _gather(S&);

    template <typename S>
    void _precede(S&);
    
    template <typename S>
    void _succeed(S&);
};

// Constructor
inline Task::Task(Node& node) : _node {&node} {
}

// Constructor
inline Task::Task(Node* node) : _node {node} {
}

// Constructor
inline Task::Task(const Task& rhs) : _node {rhs._node} {
}

// Function: precede
template <typename... Ts>
Task& Task::precede(Ts&&... tgts) {
  (_node->precede(*(tgts._node)), ...);
  return *this;
}

// Function: precede
inline Task& Task::precede(std::vector<Task>& tgts) {
  _precede(tgts);
  return *this;
}

// Function: precede
inline Task& Task::precede(std::initializer_list<Task> tgts) {
  _precede(tgts);
  return *this;
}

// Procedure: _precede
template <typename S>
void Task::_precede(S& tgts) {
  for(auto& to : tgts) {
    _node->precede(*(to._node));
  }
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

// Function: succeed
template <typename... Bs>
Task& Task::succeed(Bs&&... tgts) {
  (tgts._node->precede(*_node), ...);
  return *this;
}

// Procedure: _succeed
template <typename S>
void Task::_succeed(S& tgts) {
  for(auto& from : tgts) {
    from._node->precede(*_node);
  }
}

// Function: succeed
inline Task& Task::succeed(std::vector<Task>& tgts) {
  _succeed(tgts);
  return *this;
}

// Function: succeed
inline Task& Task::succeed(std::initializer_list<Task> tgts) {
  _succeed(tgts);
  return *this;
}

// Operator =
inline Task& Task::operator = (const Task& rhs) {
  _node = rhs._node;
  return *this;
}

// Operator =
inline Task& Task::operator = (std::nullptr_t ptr) {
  _node = ptr;
  return *this;
}

// Function: work
template <typename C>
inline Task& Task::work(C&& c) {

  if(_node->_module) {
    TF_THROW(Error::TASKFLOW, "can't assign work to a module task");
  }

  _node->_work = std::forward<C>(c);

  return *this;
}

// Function: name
inline Task& Task::name(const std::string& name) {
  _node->_name = name;
  return *this;
}

// Procedure: reset
inline Task& Task::reset() {
  _node = nullptr;
  return *this;
}

// Function: name
inline const std::string& Task::name() const {
  return _node->_name;
}

// Function: num_dependents
inline size_t Task::num_dependents() const {
  return _node->num_dependents();
}

// Function: num_successors
inline size_t Task::num_successors() const {
  return _node->num_successors();
}

// Function: empty
inline bool Task::empty() const {
  return _node == nullptr;
}

// Function: has_work
inline bool Task::has_work() const {
  return _node ? _node->_work.index() != 0 : false;
}

// ----------------------------------------------------------------------------

/**
@class TaskView

@brief A constant wrapper class to a task node, 
       mainly used in the tf::ExecutorObserver interface.

*/
class TaskView {
  
  friend class Executor;

  public:

    /**
    @brief constructs an empty task view
    */
    TaskView() = default;

    /**
    @brief constructs a task view from a task
    */
    TaskView(const Task& task);
    
    /**
    @brief constructs the task with the copy of the other task
    */
    TaskView(const TaskView& other);
    
    /**
    @brief replaces the contents with a copy of the other task
    */
    TaskView& operator = (const TaskView& other);
    
    /**
    @brief replaces the contents with another task
    */
    TaskView& operator = (const Task& other);
    
    /**
    @brief replaces the contents with a null pointer
    */
    TaskView& operator = (std::nullptr_t);
    
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
    @brief resets to an empty view
    */
    void reset();

    /**
    @brief queries if the task view is empty
    */
    bool empty() const;
    
  private:
    
    TaskView(Node&);
    TaskView(Node*);

    Node* _node {nullptr};
};

// Constructor
inline TaskView::TaskView(Node& node) : _node {&node} {
}

// Constructor
inline TaskView::TaskView(Node* node) : _node {node} {
}

// Constructor
inline TaskView::TaskView(const TaskView& rhs) : _node {rhs._node} {
}

// Constructor
inline TaskView::TaskView(const Task& task) : _node {task._node} {
}

// Operator =
inline TaskView& TaskView::operator = (const TaskView& rhs) {
  _node = rhs._node;
  return *this;
}

// Operator =
inline TaskView& TaskView::operator = (const Task& rhs) {
  _node = rhs._node;
  return *this;
}

// Operator =
inline TaskView& TaskView::operator = (std::nullptr_t ptr) {
  _node = ptr;
  return *this;
}

// Function: name
inline const std::string& TaskView::name() const {
  return _node->_name;
}

// Function: num_dependents
inline size_t TaskView::num_dependents() const {
  return _node->num_dependents();
}

// Function: num_successors
inline size_t TaskView::num_successors() const {
  return _node->num_successors();
}

// Function: reset
inline void TaskView::reset() {
  _node = nullptr;
}

// Function: empty
inline bool TaskView::empty() const {
  return _node == nullptr;
}

}  // end of namespace tf. ---------------------------------------------------

