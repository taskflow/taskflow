#pragma once

#include "graph.hpp"

/** 
@file task.hpp
@brief task include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// Task Types
// ----------------------------------------------------------------------------

/**
@enum TaskType

@brief enumeration of all task types
*/
enum TaskType {
  PLACEHOLDER_TASK = Node::PLACEHOLDER_TASK,
  CUDAFLOW_TASK    = Node::CUDAFLOW_TASK,
  STATIC_TASK      = Node::STATIC_TASK,
  DYNAMIC_TASK     = Node::DYNAMIC_TASK,
  CONDITION_TASK   = Node::CONDITION_TASK,
  MODULE_TASK      = Node::MODULE_TASK,
  ASYNC_TASK       = Node::ASYNC_TASK
};

/**
@brief convert a task type to a human-readable string
*/
inline const char* task_type_to_string(TaskType type) {

  const char* val;

  switch(type) {
    case PLACEHOLDER_TASK: val = "placeholder"; break;
    case CUDAFLOW_TASK:    val = "cudaflow";    break;
    case STATIC_TASK:      val = "static";      break;
    case DYNAMIC_TASK:     val = "subflow";     break;
    case CONDITION_TASK:   val = "condition";   break;
    case MODULE_TASK:      val = "module";      break;
    case ASYNC_TASK:       val = "async";       break;
    default:               val = "undefined";   break;
  }

  return val;
}

// ----------------------------------------------------------------------------
// Task Traits
// ----------------------------------------------------------------------------

/**
@brief determines if a callable is a static task

A static task is a callable object constructible from std::function<void()>.
*/
template <typename C>
constexpr bool is_static_task_v = std::is_invocable_r_v<void, C> &&
                                 !std::is_invocable_r_v<int, C>;

/**
@brief determines if a callable is a dynamic task

A dynamic task is a callable object constructible from std::function<void(Subflow&)>.
*/
template <typename C>
constexpr bool is_dynamic_task_v = std::is_invocable_r_v<void, C, Subflow&>;

/**
@brief determines if a callable is a condition task

A condition task is a callable object constructible from std::function<int()>.
*/
template <typename C>
constexpr bool is_condition_task_v = std::is_invocable_r_v<int, C>;

/**
@brief determines if a callable is a cudaflow task

A cudaFlow task is a callable object constructible from 
std::function<void(tf::cudaFlow&)> or std::function<void(tf::cudaFlowCapturer&)>.
*/
template <typename C>
constexpr bool is_cudaflow_task_v = std::is_invocable_r_v<void, C, cudaFlow&> ||
                                    std::is_invocable_r_v<void, C, cudaFlowCapturer&>;



// ----------------------------------------------------------------------------
// Task
// ----------------------------------------------------------------------------

/**
@class Task

@brief handle to a node in a task dependency graph

A Task is handle to manipulate a node in a taskflow graph. 
It provides a set of methods for users to access and modify the attributes of 
the associated graph node without directly touching internal node data.

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
    @brief assigns a callable

    @tparam C callable type

    @param callable callable to construct one of the static, dynamic, condition, and cudaFlow tasks

    @return @c *this
    */
    template <typename C>
    Task& work(C&& callable);
    
    /**
    @brief creates a module task from a taskflow

    @param taskflow a taskflow object for the module

    @return @c *this
    */
    Task& composed_of(Taskflow& taskflow);
    
    /**
    @brief adds precedence links from this to other tasks

    @tparam Ts parameter pack

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
    @brief makes the task release this semaphore
    */
    Task& release(Semaphore& semaphore);

    /**
    @brief makes the task acquire this semaphore
    */
    Task& acquire(Semaphore& semaphore);
    
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

    /**
    @brief obtains a hash value of the underlying node
    */
    size_t hash_value() const;
    
    /**
    @brief returns the task type
    */
    TaskType type() const;

    /**
    @brief dumps the task through an output stream
    */
    void dump(std::ostream& ostream) const;

  private:
    
    Task(Node*);

    Node* _node {nullptr};
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
  (_node->_precede(tasks._node), ...);
  //_precede(std::forward<Ts>(tasks)...);
  return *this;
}

// Function: succeed
template <typename... Ts>
Task& Task::succeed(Ts&&... tasks) {
  (tasks._node->_precede(_node), ...);
  //_succeed(std::forward<Ts>(tasks)...);
  return *this;
}

// Function: composed_of
inline Task& Task::composed_of(Taskflow& tf) {
  _node->_handle.emplace<Node::ModuleTask>(&tf);
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

// Function: acquire
inline Task& Task::acquire(Semaphore& s) {
  if(!_node->_semaphores) {
    _node->_semaphores.emplace();
  }
  _node->_semaphores->to_acquire.push_back(&s);
  return *this;
}

// Function: release
inline Task& Task::release(Semaphore& s) {
  if(!_node->_semaphores) {
    _node->_semaphores.emplace();
  }
  _node->_semaphores->to_release.push_back(&s);
  return *this;
}

// Procedure: reset
inline void Task::reset() {
  _node = nullptr;
}

// Procedure: reset_work
inline void Task::reset_work() {
  _node->_handle.emplace<std::monostate>();
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

// Function: task_type
inline TaskType Task::type() const {
  return static_cast<TaskType>(_node->_handle.index());
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

// Function: hash_value
inline size_t Task::hash_value() const {
  return std::hash<Node*>{}(_node);
}

// Procedure: dump
inline void Task::dump(std::ostream& os) const {
  os << "task ";
  if(name().empty()) os << _node;
  else os << name();
  os << " [type=" << task_type_to_string(type()) << ']';
}

// Function: work
template <typename C>
Task& Task::work(C&& c) {
  if constexpr(is_static_task_v<C>) {
    _node->_handle.emplace<Node::StaticTask>(std::forward<C>(c));
  }
  else if constexpr(is_dynamic_task_v<C>) {
    _node->_handle.emplace<Node::DynamicTask>(std::forward<C>(c));
  }
  else if constexpr(is_condition_task_v<C>) {
    _node->_handle.emplace<Node::ConditionTask>(std::forward<C>(c));
  }
  else if constexpr(is_cudaflow_task_v<C>) {
    _node->_handle.emplace<Node::cudaFlowTask>(std::forward<C>(c));
  }
  else {
    static_assert(dependent_false_v<C>, "invalid task callable");
  }
  return *this;
}

// ----------------------------------------------------------------------------
// global ostream
// ----------------------------------------------------------------------------

/**
@brief overload of ostream inserter operator for cudaTask
*/
inline std::ostream& operator << (std::ostream& os, const Task& task) {
  task.dump(os);
  return os;
}

// ----------------------------------------------------------------------------

/**
@class TaskView

@brief class to access task information from the observer interface
*/
class TaskView {
  
  friend class Executor;

  public:

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
    @brief applies an visitor callable to each successor of the task
    */
    template <typename V>
    void for_each_successor(V&& visitor) const;
    
    /**
    @brief applies an visitor callable to each dependents of the task
    */
    template <typename V>
    void for_each_dependent(V&& visitor) const;

    /**
    @brief queries the task type
    */
    TaskType type() const;
    
  private:
    
    TaskView(const Node&);
    TaskView(const TaskView&) = default;

    const Node& _node;
};

// Constructor
inline TaskView::TaskView(const Node& node) : _node {node} {
}

// Function: name
inline const std::string& TaskView::name() const {
  return _node._name;
}

// Function: num_dependents
inline size_t TaskView::num_dependents() const {
  return _node.num_dependents();
}

// Function: num_strong_dependents
inline size_t TaskView::num_strong_dependents() const {
  return _node.num_strong_dependents();
}

// Function: num_weak_dependents
inline size_t TaskView::num_weak_dependents() const {
  return _node.num_weak_dependents();
}

// Function: num_successors
inline size_t TaskView::num_successors() const {
  return _node.num_successors();
}

// Function: type
inline TaskType TaskView::type() const {
  return static_cast<TaskType>(_node._handle.index());
}

// Function: for_each_successor
template <typename V>
void TaskView::for_each_successor(V&& visitor) const {
  for(size_t i=0; i<_node._successors.size(); ++i) {
    visitor(TaskView(_node._successors[i]));
  }
}

// Function: for_each_dependent
template <typename V>
void TaskView::for_each_dependent(V&& visitor) const {
  for(size_t i=0; i<_node._dependents.size(); ++i) {
    visitor(TaskView(_node._dependents[i]));
  }
}

}  // end of namespace tf. ---------------------------------------------------

namespace std {

/**
@struct hash

@brief hash specialization for std::hash<tf::Task>
*/
template <>
struct hash<tf::Task> {
  auto operator() (const tf::Task& task) const noexcept {
    return task.hash_value();
  }
};

}  // end of namespace std ----------------------------------------------------



