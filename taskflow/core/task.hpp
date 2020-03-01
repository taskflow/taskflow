#pragma once

#include "graph.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// Task Traits
// ----------------------------------------------------------------------------

/**
@struct is_static_task

@brief determines if a callable is a static task
*/
template <typename C>
constexpr bool is_static_task_v = is_invocable_r_v<void, C> &&
                                 !is_invocable_r_v<int, C>;

/**
@struct is_dynamic_task

@brief determines if a callable is a dynamic task
*/
template <typename C>
constexpr bool is_dynamic_task_v = is_invocable_r_v<void, C, Subflow&>;

/**
@struct is_condition_task

@brief determines if a callable is a condition task
*/
template <typename C>
constexpr bool is_condition_task_v = is_invocable_r_v<int, C>;

// ----------------------------------------------------------------------------
// Task
// ----------------------------------------------------------------------------

/**
@class Task

@brief handle to a node in a task dependency graph

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
    @brief assigns a static task

    @tparam C callable object type

    @param callable a callable object acceptable to std::function<void()>

    @return @c *this
    */
    template <typename C>
    std::enable_if_t<is_static_task_v<C>, Task>& work(C&& callable);
    
    /**
    @brief assigns a dynamic task

    @tparam C callable object type

    @param callable a callable object acceptable to std::function<void(Subflow&)>

    @return @c *this
    */
    template <typename C>
    std::enable_if_t<is_dynamic_task_v<C>, Task>& work(C&& callable);
    
    /**
    @brief assigns a condition task

    @tparam C callable object type

    @param callable a callable object acceptable to std::function<int()>

    @return @c *this
    */
    template <typename C>
    std::enable_if_t<is_condition_task_v<C>, Task>& work(C&& callable);

    /**
    @brief creates a module task from a taskflow

    @param taskflow a taskflow object for the module

    @return @c *this
    */
    Task& composed_of(Taskflow& taskflow);
    
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
    */
    void reset();

    /**
    @brief resets the associated work to a placeholder
    */
    void reset_work();

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
    
    Task(Node*);

    Node* _node {nullptr};

    template <typename T>
    void _precede(T&&);
    
    template <typename T, typename... Rest>
    void _precede(T&&, Rest&&...);
    
    template <typename T>
    void _succeed(T&&);
    
    template <typename T, typename... Rest>
    void _succeed(T&&, Rest&&...);
};

// Constructor
inline Task::Task(Node* node) : _node {node} {
}

// Constructor
inline Task::Task(const Task& rhs) : _node {rhs._node} {
}

// Function: precede
template <typename... Ts>
Task& Task::precede(Ts&&... tasks) {
  //(_node->_precede(tgts._node), ...);
  _precede(std::forward<Ts>(tasks)...);
  return *this;
}

/// @private
// Procedure: _precede
template <typename T>
void Task::_precede(T&& other) {
  _node->_precede(other._node);
}

/// @private
// Procedure: _precede
template <typename T, typename... Ts>
void Task::_precede(T&& task, Ts&&... others) {
  _precede(std::forward<T>(task));
  _precede(std::forward<Ts>(others)...);
}

// Function: succeed
template <typename... Ts>
Task& Task::succeed(Ts&&... tasks) {
  //(tasks._node->_precede(_node), ...);
  _succeed(std::forward<Ts>(tasks)...);
  return *this;
}

/// @private
// Procedure: succeed
template <typename T>
void Task::_succeed(T&& other) {
  other._node->_precede(_node);
}

/// @private
// Procedure: _succeed
template <typename T, typename... Ts>
void Task::_succeed(T&& task, Ts&&... others) {
  _succeed(std::forward<T>(task));
  _succeed(std::forward<Ts>(others)...);
}

// Function: composed_of
inline Task& Task::composed_of(Taskflow& tf) {
  _node->_handle.emplace<Node::ModuleWork>(&tf);
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
inline void Task::reset() {
  _node = nullptr;
}

// Procedure: reset_work
inline void Task::reset_work() {
  _node->_handle = nstd::monostate{};
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
  return _node ? _node->_handle.index() != 0 : false;
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


