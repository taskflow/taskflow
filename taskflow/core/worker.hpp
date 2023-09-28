#pragma once

#include "declarations.hpp"
#include "tsq.hpp"
#include "notifier.hpp"

/**
@file worker.hpp
@brief worker include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// Class Definition: Worker
// ----------------------------------------------------------------------------

/**
@class Worker

@brief class to create a worker in an executor

The class is primarily used by the executor to perform work-stealing algorithm.
Users can access a worker object and alter its property
(e.g., changing the thread affinity in a POSIX-like system)
using tf::WorkerInterface.
*/
class Worker {

  friend class Executor;
  friend class WorkerView;

  public:

    /**
    @brief queries the worker id associated with its parent executor

    A worker id is a unsigned integer in the range <tt>[0, N)</tt>,
    where @c N is the number of workers spawned at the construction
    time of the executor.
    */
    inline size_t id() const { return _id; }

    /**
    @brief acquires a pointer access to the underlying thread
    */
    inline std::thread* thread() const { return _thread; }

    /**
    @brief queries the size of the queue (i.e., number of enqueued tasks to
           run) associated with the worker
    */
    inline size_t queue_size() const { return _wsq.size(); }
    
    /**
    @brief queries the current capacity of the queue
    */
    inline size_t queue_capacity() const { return static_cast<size_t>(_wsq.capacity()); }

  private:

    size_t _id;
    size_t _vtm;
    Executor* _executor;
    std::thread* _thread;
    Notifier::Waiter* _waiter;
    std::default_random_engine _rdgen { std::random_device{}() };
    TaskQueue<Node*> _wsq;
    Node* _cache;
};

// ----------------------------------------------------------------------------
// Class Definition: PerThreadWorker
// ----------------------------------------------------------------------------

/**
@private
*/
//struct PerThreadWorker {
//
//  Worker* worker;
//
//  PerThreadWorker() : worker {nullptr} {}
//
//  PerThreadWorker(const PerThreadWorker&) = delete;
//  PerThreadWorker(PerThreadWorker&&) = delete;
//
//  PerThreadWorker& operator = (const PerThreadWorker&) = delete;
//  PerThreadWorker& operator = (PerThreadWorker&&) = delete;
//};

/**
@private
*/
//inline PerThreadWorker& this_worker() {
//  thread_local PerThreadWorker worker;
//  return worker;
//}


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
    @brief queries the worker id associated with its parent executor

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


