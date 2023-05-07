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
@brief class to create a dependent asynchronous task

A tf::AsyncTask is a lightweight handle that retains @em shared ownership
of a dependent async task created by an executor.
This shared ownership ensures that the async task remains alive when
adding it to the dependency list of another async task, 
thus avoiding the classical [ABA problem](https://en.wikipedia.org/wiki/ABA_problem).

@code{.cpp}
// main thread retains shared ownership of async task A
tf::AsyncTask A = executor.silent_dependent_async([](){});

// task A remains alive (i.e., at least one ref count by the main thread) 
// when being added to the dependency list of async task B
tf::AsyncTask B = executor.silent_dependent_async([](){}, A);
@endcode

Currently, tf::AsyncTask is implemented based on C++ smart pointer std::shared_ptr and 
is considered cheap to copy or move as long as only a handful of objects
own it.
When a worker completes an async task, it will remove the task from the executor,
decrementing the number of shared owners by one.
If that counter reaches zero, the task is destroyed.
*/
class AsyncTask {
  
  friend class FlowBuilder;
  friend class Runtime;
  friend class Taskflow;
  friend class TaskView;
  friend class Executor;
  
  public:
    
    /**
    @brief constructs an empty task handle
    */
    AsyncTask() = default;
    
    /**
    @brief destroys the managed asynchronous task if this is the last owner
    */
    ~AsyncTask() = default;
    
    /**
    @brief constructs an task that shares ownership of @c rhs
    */
    AsyncTask(const AsyncTask& rhs) = default;

    /**
    @brief move-constructs an task from @c rhs
    */
    AsyncTask(AsyncTask&& rhs) = default;
    
    /**
    @brief shares ownership of the task managed by @c rhs
    */
    AsyncTask& operator = (const AsyncTask& rhs) = default;

    /**
    @brief move-assigns the task from @c rhs
    */
    AsyncTask& operator = (AsyncTask&& rhs) = default;
    
    /**
    @brief checks if the task stores a non-null shared pointer
    */
    bool empty() const;
    
    /**
    @brief release the ownership 
    */
    void reset();
    
    /**
    @brief obtains a hash value of the underlying node
    */
    size_t hash_value() const;

  private:

    AsyncTask(std::shared_ptr<Node>);

    std::shared_ptr<Node> _node;
};

// Constructor
inline AsyncTask::AsyncTask(std::shared_ptr<Node> ptr) : _node {std::move(ptr)} {
}

// Function: empty
inline bool AsyncTask::empty() const {
  return _node == nullptr;
}

// Function: reset
inline void AsyncTask::reset() {
  _node.reset();
}

// Function: hash_value
inline size_t AsyncTask::hash_value() const {
  return std::hash<std::shared_ptr<Node>>{}(_node);
}

}  // end of namespace tf ----------------------------------------------------



