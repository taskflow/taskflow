#pragma once

#include "declarations.hpp"
#include "wsq.hpp"
#include "nonblocking_notifier.hpp"
#include "atomic_notifier.hpp"


/**
@file worker.hpp
@brief worker include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// Default Notifier
// Our experiments show that the performance of atomic notifier is the best.
// ----------------------------------------------------------------------------


/**
@typedef DefaultNotifier

@brief the default notifier type used by %Taskflow

%Taskflow selects the default notifier based on build configuration.
By default, the `NonblockingNotifier` is used. 
We do not use `AtomicNotifier` since on some platforms and compiler versions,
the atomic notification may exhibit suboptimal performance due to buggy wake-up mechanisms.
These issues have been discussed in GCC bug reports and patch threads related to atomic wait/notify
implementations.

See also:
  + [GCC Bugzilla report on atomic wait/notify behavior](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=106772)
  + [GCC patch discussions on refactoring and fixing atomic notify/race issues](https://gcc.gnu.org/pipermail/gcc-patches/2025-May/685050.html)
*/
#ifdef TF_ENABLE_ATOMIC_NOTIFIER
  using DefaultNotifier = AtomicNotifier
#else
  using DefaultNotifier = NonblockingNotifier;
#endif

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
  friend class Runtime;
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
  @brief queries the size of the queue (i.e., number of enqueued tasks to
         run) associated with the worker
  */
  inline size_t queue_size() const { return _wsq.size(); }
  
  /**
  @brief queries the current capacity of the queue
  */
  inline size_t queue_capacity() const { return static_cast<size_t>(_wsq.capacity()); }
  
  /**
  @brief acquires the associated executor
  */
  inline Executor* executor() { return _executor; }

  /**
  @brief acquires the associated thread
  */
  std::thread& thread() { return _thread; }

  private:
  
  std::atomic_flag _done = ATOMIC_FLAG_INIT; 

  size_t _id;
  size_t _vtm;
  Executor* _executor {nullptr};
  std::thread _thread;
  
  std::default_random_engine _rdgen;

  BoundedWSQ<Node*> _wsq;
};

// ----------------------------------------------------------------------------
// Class Definition: WorkerView
// ----------------------------------------------------------------------------

/**
@class WorkerView

@brief class to create an immutable view of a worker 

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

// ----------------------------------------------------------------------------
// Class Definition: WorkerInterface
// ----------------------------------------------------------------------------

/**
@class WorkerInterface

@brief class to configure worker behavior in an executor

The tf::WorkerInterface class allows users to customize worker properties when creating an executor. 
Examples include binding workers to specific CPU cores or 
invoking custom methods before and after a worker enters or leaves the work-stealing loop.
When you create an executor, it spawns a set of workers to execute tasks
with the following logic:

@code{.cpp}
for(size_t n=0; n<num_workers; n++) {
  create_thread([](Worker& worker)

    // pre-processing executor-specific worker information
    // ...

    // enter the scheduling loop
    // Here, WorkerInterface::scheduler_prologue is invoked, if any
    worker_interface->scheduler_prologue(worker);
    
    try {
      while(1) {
        perform_work_stealing_algorithm();
        if(stop) {
          break;
        }
      }
    } catch(...) {
      exception_ptr = std::current_exception();
    }

    // leaves the scheduling loop and joins this worker thread
    // Here, WorkerInterface::scheduler_epilogue is invoked, if any
    worker_interface->scheduler_epilogue(worker, exception_ptr);
  );
}
@endcode

@attention
tf::WorkerInterface::scheduler_prologue and tf::WorkerInterface::scheduler_eiplogue 
are invoked by each worker simultaneously.

*/
class WorkerInterface {

  public:
  
  /**
  @brief default destructor
  */
  virtual ~WorkerInterface() = default;

  /**
  @brief method to call before a worker enters the scheduling loop
  @param worker a reference to the worker

  The method is called by the constructor of an executor.
  */
  virtual void scheduler_prologue(Worker& worker) = 0;

  /**
  @brief method to call after a worker leaves the scheduling loop
  @param worker a reference to the worker
  @param ptr an pointer to the exception thrown by the scheduling loop

  The method is called by the constructor of an executor.
  */
  virtual void scheduler_epilogue(Worker& worker, std::exception_ptr ptr) = 0;

};

/**
@brief helper function to create an instance derived from tf::WorkerInterface

@tparam T type derived from tf::WorkerInterface
@tparam ArgsT argument types to construct @c T

@param args arguments to forward to the constructor of @c T
*/
template <typename T, typename... ArgsT>
std::unique_ptr<T> make_worker_interface(ArgsT&&... args) {
  static_assert(
    std::is_base_of_v<WorkerInterface, T>,
    "T must be derived from WorkerInterface"
  );
  return std::make_unique<T>(std::forward<ArgsT>(args)...);
}


                                                                                 
                                                                                 
}  // end of namespact tf ------------------------------------------------------  


