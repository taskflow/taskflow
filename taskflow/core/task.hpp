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
enum class TaskType : int {
  /** @brief placeholder task type */
  PLACEHOLDER = 0,
  /** @brief static task type */
  STATIC,
  /** @brief runtime task type */
  RUNTIME,
  /** @brief dynamic (subflow) task type */
  SUBFLOW,
  /** @brief condition task type */
  CONDITION,
  /** @brief module task type */
  MODULE,
  /** @brief asynchronous task type */
  ASYNC,
  /** @brief undefined task type (for internal use only) */
  UNDEFINED
};

/**
@private
@brief array of all task types (used for iterating task types)
*/
inline constexpr std::array<TaskType, 7> TASK_TYPES = {
  TaskType::PLACEHOLDER,
  TaskType::STATIC,
  TaskType::RUNTIME,
  TaskType::SUBFLOW,
  TaskType::CONDITION,
  TaskType::MODULE,
  TaskType::ASYNC,
};

/**
@brief convert a task type to a human-readable string

The name of each task type is the litte-case string of its characters.
  + TaskType::PLACEHOLDER is of string `placeholder`
  + TaskType::STATIC is of string `static`
  + TaskType::RUNTIME is of string `runtime`
  + TaskType::SUBFLOW is of string `subflow`
  + TaskType::CONDITION is of string `condition`
  + TaskType::MODULE is of string `module`
  + TaskType::ASYNC is of string `async`
*/
inline const char* to_string(TaskType type) {

  const char* val;

  switch(type) {
    case TaskType::PLACEHOLDER: val = "placeholder";     break;
    case TaskType::STATIC:      val = "static";          break;
    case TaskType::RUNTIME:     val = "runtime";         break;
    case TaskType::SUBFLOW:     val = "subflow";         break;
    case TaskType::CONDITION:   val = "condition";       break;
    case TaskType::MODULE:      val = "module";          break;
    case TaskType::ASYNC:       val = "async";           break;
    default:                    val = "undefined";       break;
  }

  return val;
}

// ----------------------------------------------------------------------------
// Static Task Trait
// ----------------------------------------------------------------------------

/**
@private
*/
template <typename C, typename = void>
struct is_static_task : std::false_type {};

/**
@private
*/
template <typename C>
struct is_static_task<C, std::enable_if_t<std::is_invocable_v<C>>>
  : std::is_same<std::invoke_result_t<C>, void> {};

/**
@brief determines if a callable is a static task

A static task is a callable object constructible from std::function<void()>.
*/
template <typename C>
constexpr bool is_static_task_v = is_static_task<C>::value;

// ----------------------------------------------------------------------------
// Subflow Task Trait
// ----------------------------------------------------------------------------

/**
@private
*/
template <typename C, typename = void>
struct is_subflow_task : std::false_type {};

/**
@private
*/
template <typename C>
struct is_subflow_task<C, std::enable_if_t<std::is_invocable_v<C, tf::Subflow&>>>
  : std::is_same<std::invoke_result_t<C, tf::Subflow&>, void> {};

/**
@brief determines if a callable is a subflow task

A subflow task is a callable object constructible from std::function<void(Subflow&)>.
*/
template <typename C>
constexpr bool is_subflow_task_v = is_subflow_task<C>::value;

// ----------------------------------------------------------------------------
// Runtime Task Trait
// ----------------------------------------------------------------------------

/**
@private
*/
template <typename C, typename = void>
struct is_runtime_task : std::false_type {};

/**
@private
*/
template <typename C>
struct is_runtime_task<C, std::enable_if_t<std::is_invocable_v<C, tf::Runtime&>>>
  : std::is_same<std::invoke_result_t<C, tf::Runtime&>, void> {};

/**
@brief determines if a callable is a runtime task

A runtime task is a callable object constructible from std::function<void(Runtime&)>.
*/
template <typename C>
constexpr bool is_runtime_task_v = is_runtime_task<C>::value;


// ----------------------------------------------------------------------------
// Condition Task Trait
// ----------------------------------------------------------------------------

/**
@brief determines if a callable is a condition task

A condition task is a callable object constructible from std::function<int()>.
*/
template <typename C>
constexpr bool is_condition_task_v = std::is_invocable_r_v<int, C>;

/**
@brief determines if a callable is a multi-condition task

A multi-condition task is a callable object constructible from
std::function<tf::SmallVector<int>()>.
*/
template <typename C>
constexpr bool is_multi_condition_task_v = std::is_invocable_r_v<SmallVector<int>, C>;


// ----------------------------------------------------------------------------
// Task
// ----------------------------------------------------------------------------

/**
@class Task

@brief class to create a task handle over a taskflow node

A task points to a node in a taskflow graph and provides a set of methods for users to access and modify 
attributes of the associated node,
such as dependencies, callable, names, and so on.
A task is a very lightweight object (i.e., it only stores a node pointer) and can be trivially 
copied around. 

@code{.cpp}
// create two tasks with one dependency
auto task1 = taskflow.emplace([](){}).name("task1");
auto task2 = taskflow.emplace([](){}).name("task2");
task1.precede(task2);

// dump the task information through std::cout
task1.dump(std::cout);
@endcode

A task created from a taskflow can be one of the following types:
  + tf::TaskType::STATIC - @ref StaticTasking
  + tf::TaskType::CONDITION - @ref ConditionalTasking
  + tf::TaskType::RUNTIME - @ref RuntimeTasking
  + tf::TaskType::SUBFLOW - @ref SubflowTasking
  + tf::TaskType::MODULE - @ref ComposableTasking

@code{.cpp}
tf::Task task1 = taskflow.emplace([](){}).name("static task");
tf::Task task2 = taskflow.emplace([](){ return 3; }).name("condition task");
tf::Task task3 = taskflow.emplace([](tf::Runtime&){}).name("runtime task");
tf::Task task4 = taskflow.emplace([](tf::Subflow& sf){
  tf::Task stask1 = sf.emplace([](){});
  tf::Task stask2 = sf.emplace([](){});
}).name("subflow task");
tf::Task task5 = taskflow.composed_of(taskflow2).name("module task");
@endcode

A tf::Task is polymorphic. 
Once created, you can assign a different task type to it using tf::Task::work.
For example, the code below creates a static task and then reworks it to a subflow task:

@code{.cpp}
tf::Task task = taskflow.emplace([](){}).name("static task");
task.work([](tf::Subflow& sf){
  tf::Task stask1 = sf.emplace([](){});
  tf::Task stask2 = sf.emplace([](){});
}).name("subflow task");
@endcode

@attention
tf::Task does not own the lifetime of the associated node.
Accessing the attributes of the associated node after the taskflow has been destroyed 
can result in undefined behavior.

*/
class Task {

  friend class FlowBuilder;
  friend class Runtime;
  friend class Taskflow;
  friend class TaskView;
  friend class Executor;

  public:

    /**
    @brief constructs an empty task

    An empty task is not associated with any node in a taskflow.
    */
    Task() = default;

    /**
    @brief constructs the task with the copy of the other task

    @param other the other task to copy

    @code{.cpp}
    tf::Taskflow taskflow;
    tf::Task A = taskflow.emplace([](){ std::cout << "Task A\n"; });
    tf::Task B(A);
    assert(B == A); // Now, B and A refer to the same underlying node
    @endcode
    */
    Task(const Task& other);

    /**
    @brief replaces the contents with a copy of the other task

    @param other the other task to copy

    @code{.cpp}
    tf::Task A = taskflow.emplace([](){ std::cout << "A\n"; });
    tf::Task B;
    B = A;  // B now refers to the same node as A
    @endcode
    */
    Task& operator = (const Task& other);

    /**
    @brief replaces the contents with a null pointer

    @code{.cpp}
    tf::Task A = taskflow.emplace([](){ std::cout << "A\n"; });
    A = nullptr;  // A no longer refers to any node
    @endcode
    */
    Task& operator = (std::nullptr_t);

    /**
    @brief compares if two tasks are associated with the same taskflow node

    @param rhs the other task to compare with
    @return true if both tasks refer to the same node; false otherwise

    @code{.cpp}
    tf::Task A = taskflow.emplace([](){ std::cout << "A\n"; });
    tf::Task B = A;
    assert(A == B);  // A and B refer to the same node
    @endcode
    */
    bool operator == (const Task& rhs) const;

    /**
    @brief compares if two tasks are not associated with the same taskflow node

    @param rhs the other task to compare with
    @return true if they refer to different nodes; false otherwise

    @code{.cpp}
    tf::Task A = taskflow.emplace([](){ std::cout << "A\n"; });
    tf::Task B = taskflow.emplace([](){ std::cout << "B\n"; });
    assert(A != B);  // A and B refer to different nodes
    @endcode
    */
    bool operator != (const Task& rhs) const;

    /**
    @brief queries the name of the task

    @return the name of the task as a constant string reference
    
    @code{.cpp}
    tf::Task task = taskflow.emplace([](){});
    task.name("MyTask");
    std::cout << "Task name: " << task.name() << std::endl;
    @endcode
    */
    const std::string& name() const;

    /**
    @brief queries the number of successors of the task

    @return the number of successor tasks.
    
    @code{.cpp}
    tf::Task A = taskflow.emplace([](){});
    tf::Task B = taskflow.emplace([](){});
    A.precede(B);  // B is a successor of A
    std::cout << "A has " << A.num_successors() << " successor(s)." << std::endl;
    @endcode
    */
    size_t num_successors() const;

    /**
    @brief queries the number of predecessors of the task

    @return the number of predecessor tasks
    
    @code{.cpp}
    tf::Task A = taskflow.emplace([](){});
    tf::Task B = taskflow.emplace([](){});
    A.precede(B);  // A is a predecessor of B
    std::cout << "B has " << B.num_predecessors() << " predecessor(s)." << std::endl;
    @endcode
    */
    size_t num_predecessors() const;

    /**
    @brief queries the number of strong dependencies of the task

    @return the number of strong dependencies to this task

    A strong dependency is a preceding link from one non-condition task to another task.
    For instance, task `cond` below has one strong dependency, while tasks `yes` and `no`
    each have one weak dependency.
    
    @code{.cpp}
    auto [init, cond, yes, no] = taskflow.emplace(
     [] () { },
     [] () { return 0; },
     [] () { std::cout << "yes\n"; },
     [] () { std::cout << "no\n"; }
    );
    cond.succeed(init)
        .precede(yes, no);  // executes yes if cond returns 0
                            // executes no  if cond returns 1
    @endcode

    @dotfile images/conditional-tasking-if-else.dot
    
    @note
    To understand how %Taskflow schedule tasks under strong and weak dependencies,
    please refer to @ref ConditionalTasking.
    */
    size_t num_strong_dependencies() const;

    /**
    @brief queries the number of weak dependencies of the task

    @return the number of weak dependencies to this task

    A weak dependency is a preceding link from one condition task to another task.
    For instance, task `cond` below has one strong dependency, while tasks `yes` and `no`
    each have one weak dependency.

    @code{.cpp}
    auto [init, cond, yes, no] = taskflow.emplace(
     [] () { },
     [] () { return 0; },
     [] () { std::cout << "yes\n"; },
     [] () { std::cout << "no\n"; }
    );
    cond.succeed(init)
        .precede(yes, no);  // executes yes if cond returns 0
                            // executes no  if cond returns 1
    @endcode

    @dotfile images/conditional-tasking-if-else.dot
    
    @note
    To understand how %Taskflow schedule tasks under strong and weak dependencies,
    please refer to @ref ConditionalTasking.
    */
    size_t num_weak_dependencies() const;

    /**
    @brief assigns a name to the task

    @param name a @std_string 

    @return @c *this

    @code{.cpp}
    tf::Task task = taskflow.emplace([](){}).name("foo");
    assert(task.name*) == "foo");
    @endcode
    */
    Task& name(const std::string& name);

    /**
    @brief assigns a callable

    @tparam C callable type

    @param callable callable to construct a task

    @return @c *this

    A tf::Task is polymorphic. 
    Once created, you can reassign it to a different callable of a different task type 
    using tf::Task::work.
    For example, the code below creates a static task and reworks it to a subflow task:
    
    @code{.cpp}
    tf::Task task = taskflow.emplace([](){}).name("static task");
    task.work([](tf::Subflow& sf){
      tf::Task stask1 = sf.emplace([](){});
      tf::Task stask2 = sf.emplace([](){});
    }).name("subflow task");
    @endcode
    */
    template <typename C>
    Task& work(C&& callable);

    /**
    @brief creates a module task from a taskflow

    @tparam T object type
    @param object a custom object that defines @c T::graph() method

    @return @c *this

    The example below creates a module task from a taskflow:
    
    @code{.cpp}
    task.composed_of(taskflow);
    @endcode

    To understand how %Taskflow schedules a module task including how to create a schedulable graph,
    pleas refer to @ref CreateACustomComposableGraph.
    */
    template <typename T>
    Task& composed_of(T& object);

    /**
    @brief adds precedence links from this to other tasks

    @tparam Ts parameter pack

    @param tasks one or multiple tasks

    @return @c *this

    The example below creates a taskflow of two tasks, where `task1` runs before `task2`.

    @code{.cpp}
    auto [task1, task2] = taskflow.emplace(
      [](){ std::cout << "task1\n"; },
      [](){ std::cout << "task2\n"; }
    );
    task1.precede(task2);
    @endcode
    */
    template <typename... Ts>
    Task& precede(Ts&&... tasks);

    /**
    @brief adds precedence links from other tasks to this

    @tparam Ts parameter pack

    @param tasks one or multiple tasks

    @return @c *this
    
    The example below creates a taskflow of two tasks, where `task1` runs before `task2`.

    @code{.cpp}
    auto [task1, task2] = taskflow.emplace(
      [](){ std::cout << "task1\n"; },
      [](){ std::cout << "task2\n"; }
    );
    task2.succeed(task1);
    @endcode
    */
    template <typename... Ts>
    Task& succeed(Ts&&... tasks);
	
    /**
    @brief removes predecessor links from other tasks to this

    @tparam Ts parameter pack

    @param tasks one or multiple tasks

    @return @c *this

    This method removes the dependency links where the given tasks are predecessors
    of this task (i.e., tasks -> this). It ensures both sides of the dependency
    are updated to maintain graph consistency.
    
    @code{.cpp}
    tf::Task A = taskflow.emplace([](){});
    tf::Task B = taskflow.emplace([](){});
    tf::Task C = taskflow.emplace([](){});
    // create a linear chain of tasks, A->B->C
    B.succeed(A)
     .precede(C);
    assert(B.num_successors() == 1 && C.num_predecessors() == 1);

    // remove C from B's successor list
    C.remove_predecessors(B);
    assert(B.num_successors() == 0 && C.num_predecessors() == 0);
    @endcode
    */
    template <typename... Ts>
    Task& remove_predecessors(Ts&&... tasks);

    /**
    @brief removes successor links from this to other tasks

    @tparam Ts parameter pack

    @param tasks one or multiple tasks

    @return @c *this

    This method removes the dependency links where this task is a predecessor
    of the given tasks (i.e., this -> tasks). It ensures both sides of the dependency
    are updated to maintain graph consistency.

    @code{.cpp}
    tf::Task A = taskflow.emplace([](){});
    tf::Task B = taskflow.emplace([](){});
    tf::Task C = taskflow.emplace([](){});
    // create a linear chain of tasks, A->B->C
    B.succeed(A)
     .precede(C);
    assert(B.num_successors() == 1 && C.num_predecessors() == 1);

    // remove C from B's successor list
    B.remove_successors(C);
    assert(B.num_successors() == 0 && C.num_predecessors() == 0);
    @endcode
    */
    template <typename... Ts>
    Task& remove_successors(Ts&&... tasks);

    /**
    @brief makes the task release the given semaphore
    
    @note
    To know more about tf::Semaphore, please refer to @ref LimitTheMaximumConcurrency.
    */
    Task& release(Semaphore& semaphore);
    
    /**
    @brief makes the task release the given range of semaphores
    
    @note
    To know more about tf::Semaphore, please refer to @ref LimitTheMaximumConcurrency.
    */
    template <typename I>
    Task& release(I first, I last);

    /**
    @brief makes the task acquire the given semaphore
    
    @note
    To know more about tf::Semaphore, please refer to @ref LimitTheMaximumConcurrency.
    */
    Task& acquire(Semaphore& semaphore);

    /**
    @brief makes the task acquire the given range of semaphores
    
    @note
    To know more about tf::Semaphore, please refer to @ref LimitTheMaximumConcurrency.
    */
    template <typename I>
    Task& acquire(I first, I last);

    /**
    @brief assigns pointer to user data

    @param data pointer to user data
    @return @c *this

    The following example shows how to attach a user data to a task and retrieve it 
    during the execution of the task.

    @code{.cpp}
    tf::Executor executor;
    tf::Taskflow taskflow("attach data to a task");
    
    int data;  // user data

    // create a task and attach it a user data
    auto A = taskflow.placeholder();
    A.data(&data).work([A](){
      auto d = *static_cast<int*>(A.data());
      std::cout << "data is " << d << std::endl;
    });

    // run the taskflow iteratively with changing data
    for(data = 0; data<10; data++){
      executor.run(taskflow).wait();
    }
    @endcode

    */
    Task& data(void* data);
    
    /**
    @brief resets the task handle to null

    Resetting a task will remove its associated taskflow node and make it an empty task.

    @code{.cpp}
    tf::Task task = taskflow.emplace([](){});
    assert(task.empty() == false);
    task.reset();
    assert(task.empty() == true);
    @endcode
    */
    void reset();

    /**
    @brief resets the associated work to a placeholder
    */
    void reset_work();

    /**
    @brief queries if the task handle is associated with a taskflow node

    @return `true` if the task is not associated with any taskflow node; otherwise `false`

    @code{.cpp}
    tf::Task task;
    assert(task.empty() == true);
    @endcode

    Note that an empty task is not equal to a placeholder task.
    A placeholder task is created from tf::Taskflow::placeholder and is associated with
    a taskflow node, but its work is not assigned yet.
    */
    bool empty() const;

    /**
    @brief queries if the task has a work assigned

    @return `true` if the task has a work assigned (not placeholder); otherwise `false`

    @code{.cpp}
    tf::Task task = taskflow.placeholder();
    assert(task.has_work() == false);
    // assign a static task callable to this task
    task.work([](){});
    assert(task.has_work() == true);
    @endcode
    */
    bool has_work() const;

    /**
    @brief applies an visitor callable to each successor of the task
    
    @tparam V a callable type (function, lambda, etc.) that accepts a tf::Task handle
    @param visitor visitor to apply to each subflow task

    This method allows you to traverse and inspect successor tasks of this task.
    For instance, the code below iterates the two successors (`task2` and `task3`) of `task1`.
    
    @code{.cpp}
    auto [task1, task2, task3] = taskflow.emplace(
      [](){ std::cout << "task 1\n"; },
      [](){ std::cout << "task 2\n"; },
      [](){ std::cout << "task 3\n"; }
    });
    task1.precede(task2, task3);
    task1.for_each_successor([](tf::Task successor){
      std::cout << "successor task " << successor.name() << '\n';
    });
    @endcode

    */
    template <typename V>
    void for_each_successor(V&& visitor) const;

    /**
    @brief applies an visitor callable to each predecessor of the task
    
    @tparam V a callable type (function, lambda, etc.) that accepts a tf::Task handle
    @param visitor visitor to apply to each predecessor task

    This method allows you to traverse and inspect predecessor tasks of this task.
    For instance, the code below iterates the two predecessors (`task2` and `task3`) of `task1`.
    
    @code{.cpp}
    auto [task1, task2, task3] = taskflow.emplace(
      [](){ std::cout << "task 1\n"; },
      [](){ std::cout << "task 2\n"; },
      [](){ std::cout << "task 3\n"; }
    });
    task1.succeed(task2, task3);
    task1.for_each_predecessor([](tf::Task predecessor){
      std::cout << "predecessor task " << predecessor.name() << '\n';
    });
    @endcode
    */
    template <typename V>
    void for_each_predecessor(V&& visitor) const;

    /**
    @brief applies an visitor callable to each subflow task

    @tparam V a callable type (function, lambda, etc.) that accepts a tf::Task handle
    @param visitor visitor to apply to each subflow task

    This method allows you to traverse and inspect tasks within a subflow.
    It only applies to a subflow task.

    @code{.cpp}
    tf::Task task = taskflow.emplace([](tf::Subflow& sf){
      tf::Task stask1 = sf.emplace([](){}).name("stask1");
      tf::Task stask2 = sf.emplace([](){}).name("stask2");
    });
    // Iterate tasks in the subflow and print each subflow task.
    task.for_each_subflow_task([](tf::Task stask){
      std::cout << "subflow task " << stask.name() << '\n';
    });
    @endcode
    */
    template <typename V>
    void for_each_subflow_task(V&& visitor) const;

    /**
    @brief obtains a hash value of the underlying node

    @return the hash value of the underlying node

    The method returns std::hash on the underlying node pointer.

    @code{.cpp}
    tf::Task task = taskflow.emplace([](){});
    std::cout << "hash value of task is " << task.hash_value() << '\n';
    @endcode
    */
    size_t hash_value() const;

    /**
    @brief returns the task type

    A task can be one of the types defined in tf::TaskType and can be printed in 
    a human-readable form using tf::to_string.

    @code{.cpp}
    auto task = taskflow.emplace([](){}).name("task");
    std::cout << task.name() << " type=[" << tf::to_string(task.type()) << "]\n";
    @endcode

    */
    TaskType type() const;

    /**
    @brief dumps the task through an output stream

    The method dumps the name and the type of this task through std::cout.

    @code{.cpp}
    task.dump(std::cout);
    @endcode
    */
    void dump(std::ostream& ostream) const;

    /**
    @brief queries pointer to user data

    @return C-styled pointer to the attached user data by tf::Task::data(void* data)
    
    The following example shows how to attach a user data to a task and retrieve it 
    during the execution of the task.

    @code{.cpp}
    tf::Executor executor;
    tf::Taskflow taskflow("attach data to a task");
    
    int data;  // user data

    // create a task and attach it a user data
    auto A = taskflow.placeholder();
    A.data(&data).work([A](){
      auto d = *static_cast<int*>(A.data());
      std::cout << "data is " << d << std::endl;
    });

    // run the taskflow iteratively with changing data
    for(data = 0; data<10; data++){
      executor.run(taskflow).wait();
    }
    @endcode
    */
    void* data() const;

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

// Function: remove_predecessors
template <typename... Ts>
Task& Task::remove_predecessors(Ts&&... tasks) {
  (tasks._node->_remove_successors(_node), ...);
  (_node->_remove_predecessors(tasks._node), ...);
  return *this;
}

// Function: remove_successors
template <typename... Ts>
Task& Task::remove_successors(Ts&&... tasks) {
  (_node->_remove_successors(tasks._node), ...);
  (tasks._node->_remove_predecessors(_node), ...);
  return *this;
}

// Function: composed_of
template <typename T>
Task& Task::composed_of(T& object) {
  _node->_handle.emplace<Node::Module>(object);
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
    _node->_semaphores = std::make_unique<Node::Semaphores>();
  }
  _node->_semaphores->to_acquire.push_back(&s);
  return *this;
}

// Function: acquire
template <typename I>
Task& Task::acquire(I first, I last) {
  if(!_node->_semaphores) {
    _node->_semaphores = std::make_unique<Node::Semaphores>();
  }
  _node->_semaphores->to_acquire.reserve(
    _node->_semaphores->to_acquire.size() + std::distance(first, last)
  );
  for(auto s = first; s != last; ++s){
    _node->_semaphores->to_acquire.push_back(&(*s));
  }
  return *this;
}

// Function: release
inline Task& Task::release(Semaphore& s) {
  if(!_node->_semaphores) {
    _node->_semaphores = std::make_unique<Node::Semaphores>();
  }
  _node->_semaphores->to_release.push_back(&s);
  return *this;
}

// Function: release
template <typename I>
Task& Task::release(I first, I last) {
  if(!_node->_semaphores) {
    _node->_semaphores = std::make_unique<Node::Semaphores>();
  }
  _node->_semaphores->to_release.reserve(
    _node->_semaphores->to_release.size() + std::distance(first, last)
  );
  for(auto s = first; s != last; ++s) {
    _node->_semaphores->to_release.push_back(&(*s));
  }
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

// Function: num_predecessors
inline size_t Task::num_predecessors() const {
  return _node->num_predecessors();
}

// Function: num_strong_dependencies
inline size_t Task::num_strong_dependencies() const {
  return _node->num_strong_dependencies();
}

// Function: num_weak_dependencies
inline size_t Task::num_weak_dependencies() const {
  return _node->num_weak_dependencies();
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
  switch(_node->_handle.index()) {
    case Node::PLACEHOLDER:     return TaskType::PLACEHOLDER;
    case Node::STATIC:          return TaskType::STATIC;
    case Node::RUNTIME:         return TaskType::RUNTIME;
    case Node::SUBFLOW:         return TaskType::SUBFLOW;
    case Node::CONDITION:       return TaskType::CONDITION;
    case Node::MULTI_CONDITION: return TaskType::CONDITION;
    case Node::MODULE:          return TaskType::MODULE;
    case Node::ASYNC:           return TaskType::ASYNC;
    case Node::DEPENDENT_ASYNC: return TaskType::ASYNC;
    default:                    return TaskType::UNDEFINED;
  }
}

// Function: for_each_successor
template <typename V>
void Task::for_each_successor(V&& visitor) const {
  for(size_t i=0; i<_node->_num_successors; ++i) {
    visitor(Task(_node->_edges[i]));
  }
}

// Function: for_each_predecessor
template <typename V>
void Task::for_each_predecessor(V&& visitor) const {
  for(size_t i=_node->_num_successors; i<_node->_edges.size(); ++i) {
    visitor(Task(_node->_edges[i]));
  }
}

// Function: for_each_subflow_task
template <typename V>
void Task::for_each_subflow_task(V&& visitor) const {
  if(auto ptr = std::get_if<Node::Subflow>(&_node->_handle); ptr) {
    for(auto itr = ptr->subgraph.begin(); itr != ptr->subgraph.end(); ++itr) {
      visitor(Task(itr->get()));
    }
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
  os << " [type=" << to_string(type()) << ']';
}

// Function: work
template <typename C>
Task& Task::work(C&& c) {

  if constexpr(is_static_task_v<C>) {
    _node->_handle.emplace<Node::Static>(std::forward<C>(c));
  }
  else if constexpr(is_runtime_task_v<C>) {
    _node->_handle.emplace<Node::Runtime>(std::forward<C>(c));
  }
  else if constexpr(is_subflow_task_v<C>) {
    _node->_handle.emplace<Node::Subflow>(std::forward<C>(c));
  }
  else if constexpr(is_condition_task_v<C>) {
    _node->_handle.emplace<Node::Condition>(std::forward<C>(c));
  }
  else if constexpr(is_multi_condition_task_v<C>) {
    _node->_handle.emplace<Node::MultiCondition>(std::forward<C>(c));
  }
  else {
    static_assert(dependent_false_v<C>, "invalid task callable");
  }
  return *this;
}

// Function: data
inline void* Task::data() const {
  return _node->_data;
}

// Function: data
inline Task& Task::data(void* data) {
  _node->_data = data;
  return *this;
}

// ----------------------------------------------------------------------------
// global ostream
// ----------------------------------------------------------------------------

/**
@brief overload of ostream inserter operator for Task
*/
inline std::ostream& operator << (std::ostream& os, const Task& task) {
  task.dump(os);
  return os;
}

// ----------------------------------------------------------------------------
// Task View
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
    size_t num_predecessors() const;

    /**
    @brief queries the number of strong dependencies of the task
    */
    size_t num_strong_dependencies() const;

    /**
    @brief queries the number of weak dependencies of the task
    */
    size_t num_weak_dependencies() const;

    /**
    @brief applies an visitor callable to each successor of the task
    
    @tparam V a callable type (function, lambda, etc.) that accepts a tf::Task handle
    @param visitor visitor to apply to each subflow task

    This method allows you to traverse and inspect successor tasks of this task.
    */
    template <typename V>
    void for_each_successor(V&& visitor) const;

    /**
    @brief applies an visitor callable to each predecessor of the task
    
    @tparam V a callable type (function, lambda, etc.) that accepts a tf::Task handle
    @param visitor visitor to apply to each predecessor task

    This method allows you to traverse and inspect predecessor tasks of this task.
    */
    template <typename V>
    void for_each_predecessor(V&& visitor) const;

    /**
    @brief queries the task type
    */
    TaskType type() const;

    /**
    @brief obtains a hash value of the underlying node
    */
    size_t hash_value() const;

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

// Function: num_predecessors
inline size_t TaskView::num_predecessors() const {
  return _node.num_predecessors();
}

// Function: num_strong_dependencies
inline size_t TaskView::num_strong_dependencies() const {
  return _node.num_strong_dependencies();
}

// Function: num_weak_dependencies
inline size_t TaskView::num_weak_dependencies() const {
  return _node.num_weak_dependencies();
}

// Function: num_successors
inline size_t TaskView::num_successors() const {
  return _node.num_successors();
}

// Function: type
inline TaskType TaskView::type() const {
  switch(_node._handle.index()) {
    case Node::PLACEHOLDER:     return TaskType::PLACEHOLDER;
    case Node::STATIC:          return TaskType::STATIC;
    case Node::RUNTIME:         return TaskType::RUNTIME;
    case Node::SUBFLOW:         return TaskType::SUBFLOW;
    case Node::CONDITION:       return TaskType::CONDITION;
    case Node::MULTI_CONDITION: return TaskType::CONDITION;
    case Node::MODULE:          return TaskType::MODULE;
    case Node::ASYNC:           return TaskType::ASYNC;
    case Node::DEPENDENT_ASYNC: return TaskType::ASYNC;
    default:                    return TaskType::UNDEFINED;
  }
}

// Function: hash_value
inline size_t TaskView::hash_value() const {
  return std::hash<const Node*>{}(&_node);
}

// Function: for_each_successor
template <typename V>
void TaskView::for_each_successor(V&& visitor) const {
  for(size_t i=0; i<_node._num_successors; ++i) {
    visitor(TaskView(*_node._edges[i]));
  }
  //for(size_t i=0; i<_node._successors.size(); ++i) {
  //  visitor(TaskView(*_node._successors[i]));
  //}
}

// Function: for_each_predecessor
template <typename V>
void TaskView::for_each_predecessor(V&& visitor) const {
  for(size_t i=_node._num_successors; i<_node._edges.size(); ++i) {
    visitor(TaskView(*_node._edges[i]));
  }
  //for(size_t i=0; i<_node._predecessors.size(); ++i) {
  //  visitor(TaskView(*_node._predecessors[i]));
  //}
}

}  // end of namespace tf. ----------------------------------------------------

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

/**
@struct hash

@brief hash specialization for std::hash<tf::TaskView>
*/
template <>
struct hash<tf::TaskView> {
  auto operator() (const tf::TaskView& task_view) const noexcept {
    return task_view.hash_value();
  }
};

}  // end of namespace std ----------------------------------------------------



