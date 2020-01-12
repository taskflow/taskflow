#pragma once

#include "graph.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// Task
// ----------------------------------------------------------------------------

/**
@class Task

@brief task handle to a node in a task dependency graph

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
    @brief compares if two tasks are associated with the same graph node
    */
    bool operator == (const Task& rhs) const;

    /**
    @brief compares if two tasks are not associated with the same graph node
    */
    bool operator != (const Task& rhs) const;
    
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
    @brief queries the number of strong dependents of the task
    */
    size_t num_strong_dependents() const;

    /**
    @brief queries the number of weak dependents of the task
    */
    size_t num_weak_dependents() const;
    
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
    @brief adds precedence links from other tasks to this

    @tparam Ts parameter pack 

    @param tasks one or multiple tasks

    @return @c *this
    */
    template <typename... Ts>
    Task& succeed(Ts&&... tasks);
    
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
  (_node->_precede(tgts._node), ...);
  return *this;
}

// Function: succeed
template <typename... Bs>
Task& Task::succeed(Bs&&... tgts) {
  (tgts._node->_precede(_node), ...);
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

// Operator ==
inline bool Task::operator == (const Task& rhs) const {
  return _node == rhs._node;
}

// Operator !=
inline bool Task::operator != (const Task& rhs) const {
  return _node != rhs._node;
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

// Function: num_strong_dependents
inline size_t Task::num_strong_dependents() const {
  return _node->num_strong_dependents();
}

// Function: num_weak_dependents
inline size_t Task::num_weak_dependents() const {
  return _node->num_weak_dependents();
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

// Function: for_each_successor
template <typename V>
void Task::for_each_successor(V&& visitor) const {
  for(size_t i=0; i<_node->_successors.size(); ++i) {
    visitor(Task(_node->_successors[i]));
  }
}

// Function: for_each_dependent
template <typename V>
void Task::for_each_dependent(V&& visitor) const {
  for(size_t i=0; i<_node->_dependents.size(); ++i) {
    visitor(Task(_node->_dependents[i]));
  }
}

// ----------------------------------------------------------------------------

/**
@class TaskView

@brief an immutable accessor class to a task node, 
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
    @brief compares if two taskviews are associated with the same task
    */
    bool operator == (const TaskView&) const;
    
    /**
    @brief compares if two taskviews are associated with different tasks
    */
    bool operator != (const TaskView&) const;
    
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
    @brief queries the number of strong dependents of the task
    */
    size_t num_strong_dependents() const;

    /**
    @brief queries the number of weak dependents of the task
    */
    size_t num_weak_dependents() const;

    /**
    @brief resets to an empty view
    */
    void reset();

    /**
    @brief queries if the task view is empty
    */
    bool empty() const;
    
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
    
    TaskView(Node*);

    Node* _node {nullptr};
};

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

// Function: num_strong_dependents
inline size_t TaskView::num_strong_dependents() const {
  return _node->num_strong_dependents();
}

// Function: num_weak_dependents
inline size_t TaskView::num_weak_dependents() const {
  return _node->num_weak_dependents();
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

// Operator ==
inline bool TaskView::operator == (const TaskView& rhs) const {
  return _node == rhs._node;
}

// Operator !=
inline bool TaskView::operator != (const TaskView& rhs) const {
  return _node != rhs._node;
}

// Function: for_each_successor
template <typename V>
void TaskView::for_each_successor(V&& visitor) const {
  for(size_t i=0; i<_node->_successors.size(); ++i) {
    visitor(TaskView(_node->_successors[i]));
  }
}

// Function: for_each_dependent
template <typename V>
void TaskView::for_each_dependent(V&& visitor) const {
  for(size_t i=0; i<_node->_dependents.size(); ++i) {
    visitor(TaskView(_node->_dependents[i]));
  }
}

}  // end of namespace tf. ---------------------------------------------------


