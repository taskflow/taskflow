#pragma once

#include "graph.hpp"

namespace tf {

/**

@class SuccessorsRange

@brief Creates a range of successors of a task/taskview

*/
template <typename T>
class SuccessorsRange {

  static_assert(
    std::is_same_v<T, Task> || std::is_same_v<T, TaskView>,
    "SuccessorsRange takes only Task or TaskView type"
  );
  
  public:

  class Iterator {

    friend class SuccessorsRange;

    public:
      
      /**
      @brief default constructor
      */
      Iterator() = default;
      
      /**
      @brief copy constructor
      */
      Iterator(const Iterator&) = default;
      
      /**
      @brief mutable object accessor 
      */
      T operator * () { return _item; }

      /**
      @brief immutable object accessor
      */
      const T& operator * () const { return _item; }
      
      /**
      @brief mutable object pointer accessor
      */
      T* operator -> () { return &_item; }

      /**
      @brief immutable object pointer accessor
      */
      const T* operator -> () const { return &_item; }

      /**
      @brief compares if two iterators equal each other
      */
      bool operator == (const Iterator& rhs) const { return _cursor == rhs._cursor; }

      /**
      @brief compares if two iterators differ from each other
      */
      bool operator != (const Iterator& rhs) const { return !(*this == rhs); }
      
      /**
      @brief assigns a new iterator
      */
      Iterator& operator = (const Iterator& rhs) {
        _cursor = rhs._cursor;
        _item = rhs._item;
        return *this;
      }
      
      /**
      @brief prefix increment operator
      */
      Iterator& operator ++ () { 
        _item._node = *(++_cursor);
        return *this; 
      }
      
      /**
      @brief postfix increment operator
      */
      Iterator& operator ++ (int n) { 
        _item._node = *(n == 0 ? _cursor++ : _cursor += n);
        return *this; 
      }

      /**
      @brief prefix decrement operator
      */
      Iterator& operator -- () {
        _item._node = *(--_cursor);
        return *this;
      }

      /**
      @brief postfix decrement oeprator
      */
      Iterator& operator -- (int n) {
        _item._node = *(n == 0 ? _cursor-- : _cursor -= n);
        return *this;
      }

    private:

      PassiveVector<Node*>::iterator _cursor {nullptr};
      T _item;
      
      template <typename I>
      Iterator(I in) : _cursor {in}, _item{*in} { }
  };
    
    /**
    @brief constructs a successor range from a task
    */
    SuccessorsRange(T t) : _node {t._node} {}
    
    /**
    @brief returns an iterator to the beginning of the successor range
    */
    Iterator begin() { return _node->_successors.begin(); }

    /**
    @brief returns an iterator to the end of the successor range
    */
    Iterator end() { return _node->_successors.end(); }
    
    /**
    @brief returns the size of the range
    */
    size_t size() const { return _node->_successors.size(); }

  private:

    Node* _node {nullptr};
};


/**

@class DependentsRange

@brief Creates a range of dependents of a task/taskview

*/
template <typename T>
class DependentsRange {

  static_assert(
    std::is_same_v<T, Task> || std::is_same_v<T, TaskView>,
    "DependentsRange takes only Task or TaskView type"
  );
  
  public:

  class Iterator {

    friend class DependentsRange;

    public:
      
      /**
      @brief default constructor
      */
      Iterator() = default;
      
      /**
      @brief copy constructor
      */
      Iterator(const Iterator&) = default;
      
      /**
      @brief mutable object accessor 
      */
      T operator * () { return _item; }

      /**
      @brief immutable object accessor
      */
      const T& operator * () const { return _item; }
      
      /**
      @brief mutable object pointer accessor
      */
      T* operator -> () { return &_item; }

      /**
      @brief immutable object pointer accessor
      */
      const T* operator -> () const { return &_item; }

      /**
      @brief compares if two iterators equal each other
      */
      bool operator == (const Iterator& rhs) const { return _cursor == rhs._cursor; }

      /**
      @brief compares if two iterators differ from each other
      */
      bool operator != (const Iterator& rhs) const { return !(*this == rhs); }
      
      /**
      @brief assigns a new iterator
      */
      Iterator& operator = (const Iterator& rhs) {
        _cursor = rhs._cursor;
        _item = rhs._item;
        return *this;
      }
      
      /**
      @brief prefix increment operator
      */
      Iterator& operator ++ () { 
        _item._node = *(++_cursor);
        return *this; 
      }
      
      /**
      @brief postfix increment operator
      */
      Iterator& operator ++ (int n) { 
        _item._node = *(n == 0 ? _cursor++ : _cursor += n);
        return *this; 
      }

      /**
      @brief prefix decrement operator
      */
      Iterator& operator -- () {
        _item._node = *(--_cursor);
        return *this;
      }

      /**
      @brief postfix decrement oeprator
      */
      Iterator& operator -- (int n) {
        _item._node = *(n == 0 ? _cursor-- : _cursor -= n);
        return *this;
      }

    private:

      PassiveVector<Node*>::iterator _cursor {nullptr};
      T _item;
      
      template <typename I>
      Iterator(I in) : _cursor {in}, _item{*in} { }
  };
    
    /**
    @brief constructs a predecessor range from a task
    */
    DependentsRange(T t) : _node {t._node} {}
    
    /**
    @brief returns an iterator to the beginning of the predecessor range
    */
    Iterator begin() { return _node->_dependents.begin(); }

    /**
    @brief returns an iterator to the end of the predecessor range
    */
    Iterator end() { return _node->_dependents.end(); }
    
    /**
    @brief returns the size of the range
    */
    size_t size() const { return _node->_dependents.size(); }

  private:

    Node* _node {nullptr};
};

// ----------------------------------------------------------------------------
// Task
// ----------------------------------------------------------------------------

/**
@class Task

@brief task handle

A Task is a wrapper of a node in a dependency graph. 
It provides a set of methods for users to access and modify the attributes of 
the task node,
preventing direct access to the internal data storage.

*/
class Task {

  friend class FlowBuilder;
  friend class Taskflow;
  friend class TaskView;
  
  friend class SuccessorsRange<Task>;
  friend class DependentsRange<Task>;

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
    @brief queries the number of non-condition predecessors of the task
    */
    size_t num_strong_dependents() const;

    /**
    @brief queries the number of condition predecessors of the task
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
    @brief returns a range object of successors of this task for iterating
    */
    SuccessorsRange<Task> successors() const;

    /**
    @brief returns a range object of dependents of this task for iterating
    */
    DependentsRange<Task> dependents() const;

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

// Function: successors
inline SuccessorsRange<Task> Task::successors() const {
  return SuccessorsRange<Task>(*this);
}

// Function: dependents
inline DependentsRange<Task> Task::dependents() const {
  return DependentsRange<Task>(*this);
}

// ----------------------------------------------------------------------------

/**
@class TaskView

@brief an immutable accessor class to a task node, 
       mainly used in the tf::ExecutorObserver interface.

*/
class TaskView {
  
  friend class Executor;
  
  friend class SuccessorsRange<TaskView>;
  friend class DependentsRange<TaskView>;

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
    @brief queries the number of non-condition predecessors of the task
    */
    size_t num_strong_dependents() const;

    /**
    @brief queries the number of condition predecessors of the task
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




}  // end of namespace tf. ---------------------------------------------------










