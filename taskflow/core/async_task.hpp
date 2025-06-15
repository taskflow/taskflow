#pragma once

#include "graph.hpp"

/**
@file async_task.hpp
@brief asynchronous task include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// AsyncTask
// ----------------------------------------------------------------------------

/**
@brief class to hold a dependent asynchronous task with shared ownership

A tf::AsyncTask is a lightweight handle that retains @em shared ownership
of a dependent asynchronous (dependent-async) task created by an executor.
This shared ownership ensures that the dependent-async task remains alive when
adding it to the dependency list of another dependent-async task, 
thus avoiding the classical [ABA problem](https://en.wikipedia.org/wiki/ABA_problem).

@code{.cpp}
// main thread retains shared ownership of async task A
tf::AsyncTask A = executor.silent_dependent_async([](){});

// task A remains alive (i.e., at least one ref count by the main thread) 
// when being added to the dependency list of async task B
tf::AsyncTask B = executor.silent_dependent_async([](){}, A);
@endcode

tf::AsyncTask is implemented based on the logic of 
C++ smart pointer std::shared_ptr and 
is considered cheap to copy or move as long as only a handful of objects
own it.
When a worker completes an async task, it will remove the task from the executor,
decrementing the number of shared owners by one.
If that counter reaches zero, the task is destroyed.

@note
To know more about dependent-async task, please refer to @ref DependentAsyncTasking.
*/
class AsyncTask {
  
  friend class Executor;
  
  public:
    
    /**
    @brief constructs an empty task handle
    */
    AsyncTask() = default;
    
    /**
    @brief destroys the managed dependent-async task if this is the last owner
    */
    ~AsyncTask();
    
    /**
    @brief constructs a dependent-async task that shares ownership of @c rhs
    */
    AsyncTask(const AsyncTask& rhs);

    /**
    @brief move-constructs an dependent-async task from @c rhs
    */
    AsyncTask(AsyncTask&& rhs);
    
    /**
    @brief copy-assigns the dependent-async task from @c rhs

    Releases the managed object of @c this and retains a new shared ownership
    of @c rhs.
    */
    AsyncTask& operator = (const AsyncTask& rhs);

    /**
    @brief move-assigns the dependent-async task from @c rhs
    
    Releases the managed object of @c this and takes over the ownership of @c rhs.
    */
    AsyncTask& operator = (AsyncTask&& rhs);
    
    /**
    @brief checks if this dependent-async task is associated with any task

    An empty dependent-async task is not associated with any task created 
    from the executor.

    @code{.cpp}
    tf::AsyncTask task;
    assert(task.empty());
    @endcode
    */
    bool empty() const;

    /**
    @brief release the managed object of @c this
    
    Releases the ownership of the managed task, if any. 
    After the call `*this` manages no task.

    @code{.cpp}
    tf::AsyncTask task = executor.silent_dependent_async([](){});
    assert(task.empty() == false);
    task.reset();
    assert(task.empty() == true);
    @endcode
    */
    void reset();
    
    /**
    @brief obtains the hashed value of this dependent-async task
    
    @code{.cpp}
    tf::AsyncTask task = executor.silent_dependent_async([](){});
    std::cout << task.hash_value() << '\n';
    @endcode
    */
    size_t hash_value() const;

    /**
    @brief returns the number of shared owners that are currently managing 
           this dependent-async task
    
    In a multithreaded environment, `use_count` atomically retrieves 
    (with `memory_order_relaxed` load) the number of tf::AsyncTask instances that manage 
    the current task.

    @code{.cpp}
    tf::AsyncTask task;
    assert(task.use_count() == 0);
    @endcode
    */
    size_t use_count() const;

    /**                                                                                                       
    @brief checks if this dependent-async task finishes

    In a multithreaded environment, `is_done` atomically retrieves 
    (with `memory_order_acquire` load) the underlying state bit that indicates
    the completion of this dependent-async task.
    If the dependent-async task is empty, returns `true`.

    @code{.cpp}
    tf::AsyncTask task = executor.silent_dependent_async([](){});
    while(task.is_done() == false);
    std::cout << "dependent-async task finishes\n";

    task.reset();
    assert(task.is_done() == true);
    @endcode

    */
    bool is_done() const; 

  private:

    explicit AsyncTask(Node*);

    Node* _node {nullptr};

    void _incref();
    void _decref();
};

// Constructor
inline AsyncTask::AsyncTask(Node* ptr) : _node{ptr} {
  _incref();
}

// Function: _incref
inline void AsyncTask::_incref() {
  if(_node) {
    std::get_if<Node::DependentAsync>(&(_node->_handle))->use_count.fetch_add(
      1, std::memory_order_relaxed
    );
  }
}

// Function: _decref
inline void AsyncTask::_decref() {
  if(_node && std::get_if<Node::DependentAsync>(&(_node->_handle))->use_count.fetch_sub(
      1, std::memory_order_acq_rel
    ) == 1) {
    recycle(_node);
  }
}

// Copy Constructor
inline AsyncTask::AsyncTask(const AsyncTask& rhs) : 
  _node{rhs._node} {
  _incref();
}

// Move Constructor
inline AsyncTask::AsyncTask(AsyncTask&& rhs) :
  _node {rhs._node} {
  rhs._node = nullptr;
}

// Destructor
inline AsyncTask::~AsyncTask() {
  _decref();
}

// Copy assignment
inline AsyncTask& AsyncTask::operator = (const AsyncTask& rhs) {
  _decref();
  _node = rhs._node;
  _incref();
  return *this;
}

// Move assignment
inline AsyncTask& AsyncTask::operator = (AsyncTask&& rhs) {
  _decref();
  _node = rhs._node;
  rhs._node = nullptr;
  return *this;
}

// Function: empty
inline bool AsyncTask::empty() const {
  return _node == nullptr;
}

// Function: reset
inline void AsyncTask::reset() {
  _decref();
  _node = nullptr;
}

// Function: hash_value
inline size_t AsyncTask::hash_value() const {
  return std::hash<Node*>{}(_node);
}

// Function: use_count
inline size_t AsyncTask::use_count() const {
  return _node == nullptr ? size_t{0} : 
  std::get_if<Node::DependentAsync>(&(_node->_handle))->use_count.load(
    std::memory_order_relaxed
  );
}

// Function: is_done
inline bool AsyncTask::is_done() const {
  return _node == nullptr ? true:
  std::get_if<Node::DependentAsync>(&(_node->_handle))->state.load(
    std::memory_order_acquire
  ) == ASTATE::FINISHED;
}

}  // end of namespace tf ----------------------------------------------------



