#pragma once

#include "../core/task.hpp"
#include "../core/worker.hpp"

/**
@file interface.hpp
@brief ObserverInterface include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// common observer types
// ----------------------------------------------------------------------------

/**
@brief default time point type of observers
*/
using observer_stamp_t = std::chrono::time_point<std::chrono::steady_clock>;

// ----------------------------------------------------------------------------
// ObserverInterface
// ----------------------------------------------------------------------------

/**
@class: ObserverInterface

@brief class to derive an executor observer 

The tf::ObserverInterface class allows users to define custom methods to monitor 
the behaviors of an executor. This is particularly useful when you want to 
inspect the performance of an executor and visualize when each thread 
participates in the execution of a task.
To prevent users from direct access to the internal threads and tasks, 
tf::ObserverInterface provides immutable wrappers,
tf::WorkerView and tf::TaskView, over workers and tasks.

Please refer to tf::WorkerView and tf::TaskView for details.

Example usage:

@code{.cpp}

struct MyObserver : public tf::ObserverInterface {

  MyObserver(const std::string& name) {
    std::cout << "constructing observer " << name << '\n';
  }

  void set_up(size_t num_workers) override final {
    std::cout << "setting up observer with " << num_workers << " workers\n";
  }

  void on_entry(WorkerView w, tf::TaskView tv) override final {
    std::ostringstream oss;
    oss << "worker " << w.id() << " ready to run " << tv.name() << '\n';
    std::cout << oss.str();
  }

  void on_exit(WorkerView w, tf::TaskView tv) override final {
    std::ostringstream oss;
    oss << "worker " << w.id() << " finished running " << tv.name() << '\n';
    std::cout << oss.str();
  }
};
  
tf::Taskflow taskflow;
tf::Executor executor;

// insert tasks into taskflow
// ...
  
// create a custom observer
std::shared_ptr<MyObserver> observer = executor.make_observer<MyObserver>("MyObserver");

// run the taskflow
executor.run(taskflow).wait();
@endcode
*/
class ObserverInterface {

  public:

  /**
  @brief virtual destructor
  */
  virtual ~ObserverInterface() = default;
  
  /**
  @brief constructor-like method to call when the executor observer is fully created
  @param num_workers the number of the worker threads in the executor
  */
  virtual void set_up(size_t num_workers) = 0;
  
  /**
  @brief method to call before a worker thread executes a closure 
  @param wv an immutable view of this worker thread 
  @param task_view a constant wrapper object to the task 
  */
  virtual void on_entry(WorkerView wv, TaskView task_view) = 0;
  
  /**
  @brief method to call after a worker thread executed a closure
  @param wv an immutable view of this worker thread
  @param task_view a constant wrapper object to the task
  */
  virtual void on_exit(WorkerView wv, TaskView task_view) = 0;
};

}  // end of namespace tf -----------------------------------------------------
