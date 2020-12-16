#pragma once

#include "declarations.hpp"
#include "tsq.hpp"
#include "notifier.hpp"

/** 
@file worker.hpp
@brief worker include file
*/

namespace tf {

/**
@private
*/
struct Worker {

  friend class Executor;
  friend class WorkerView;

  private:

    size_t _id;
    size_t _vtm;
    Executor* _executor;
    Notifier::Waiter* _waiter;
    std::default_random_engine _rdgen { std::random_device{}() };
    TaskQueue<Node*> _wsq;
};

// ----------------------------------------------------------------------------
// Class Definition: WorkerView
// ----------------------------------------------------------------------------

/**
@class WorkerView

@brief class to create an immutable view of a worker in an executor

An executor keeps a set of internal worker threads to run tasks.
A worker view provides users an immutable interface to observe
when a worker runs a task, and the view object is only accessible
from an observer derived from tf::ObserverInterface.
*/
class WorkerView {
  
  friend class Executor;
  
  public:
    
    /**
    @brief queries the worker id associated with the executor

    A worker id is a unsigned integer in the range <tt>[0, N)</tt>,
    where @c N is the number of workers spawned at the construction
    time of the executor.
    */
    size_t id() const;
    
    /**
    @brief queries the size of the queue (i.e., number of pending tasks to 
           run) associated with the worker
    */
    size_t queue_size() const;

    /**
    @brief queries the current capacity of the queue
    */
    size_t queue_capacity() const;

  private:

    WorkerView(const Worker&);
    WorkerView(const WorkerView&) = default;

    const Worker& _worker;

};

// Constructor
inline WorkerView::WorkerView(const Worker& w) : _worker{w} {
}

// function: id
inline size_t WorkerView::id() const {
  return _worker._id;
}

// Function: queue_size
inline size_t WorkerView::queue_size() const {
  return _worker._wsq.size();
}

// Function: queue_capacity
inline size_t WorkerView::queue_capacity() const {
  return static_cast<size_t>(_worker._wsq.capacity());
}


}  // end of namespact tf -----------------------------------------------------


