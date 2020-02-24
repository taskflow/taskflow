// 2019/07/31 - modified by Tsung-Wei Huang
//  - fixed the missing comma in outputing JSON  
//
// 2019/06/13 - modified by Tsung-Wei Huang
//  - added TaskView interface
//
// 2019/04/17 - created by Tsung-Wei Huang

#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <memory>
#include <deque>
#include <thread>
#include <algorithm>
#include <set>
#include <numeric>
#include <cassert>

#include "task.hpp"

namespace tf {

/**
@class: ExecutorObserverInterface

@brief The interface class for creating an executor observer.

The tf::ExecutorObserver class let users define methods to monitor the behaviors
of an executor. 
This is particularly useful when you want to inspect the performance of an executor.
*/
class ExecutorObserverInterface {
  
  public:

  /**
  @brief virtual destructor
  */
  virtual ~ExecutorObserverInterface() = default;
  
  /**
  @brief constructor-like method to call when the executor observer is fully created
  @param num_workers the number of the worker threads in the executor
  */
  virtual void set_up(unsigned num_workers) = 0;
  
  /**
  @brief method to call before a worker thread executes a closure 
  @param worker_id the id of this worker thread 
  @param task_view a constant wrapper object to the task 
  */
  virtual void on_entry(unsigned worker_id, TaskView task_view) = 0;
  
  /**
  @brief method to call after a worker thread executed a closure
  @param worker_id the id of this worker thread 
  @param task_view a constant wrapper object to the task
  */
  virtual void on_exit(unsigned worker_id, TaskView task_view) = 0;
};

// ------------------------------------------------------------------

/**
@class: ExecutorObserver

@brief Default executor observer to dump the execution timelines

*/
class ExecutorObserver : public ExecutorObserverInterface {

  friend class Executor;
  
  // data structure to record each task execution
  struct Execution {

    TaskView task_view;

    std::chrono::time_point<std::chrono::steady_clock> beg;
    std::chrono::time_point<std::chrono::steady_clock> end;

    Execution(
      TaskView tv, 
      std::chrono::time_point<std::chrono::steady_clock> b
    ) :
      task_view {tv}, beg {b} {
    } 

    Execution(
      TaskView tv,
      std::chrono::time_point<std::chrono::steady_clock> b,
      std::chrono::time_point<std::chrono::steady_clock> e
    ) :
      task_view {tv}, beg {b}, end {e} {
    }
  };
  
  // data structure to store the entire execution timeline
  struct Timeline {
    std::chrono::time_point<std::chrono::steady_clock> origin;
    std::vector<std::vector<Execution>> executions;
  };  

  public:
    
    /**
    @brief dump the timelines in JSON format to an ostream
    @param ostream the target std::ostream to dump
    */
    inline void dump(std::ostream& ostream) const;

    /**
    @brief dump the timelines in JSON to a std::string
    @return a JSON string 
    */
    inline std::string dump() const;
    
    /**
    @brief clear the timeline data
    */
    inline void clear();

    /**
    @brief get the number of total tasks in the observer
    @return number of total tasks
    */
    inline size_t num_tasks() const;

  private:
    
    inline void set_up(unsigned num_workers) override final;
    inline void on_entry(unsigned worker_id, TaskView task_view) override final;
    inline void on_exit(unsigned worker_id, TaskView task_view) override final;

    Timeline _timeline;
};  

// Procedure: set_up
inline void ExecutorObserver::set_up(unsigned num_workers) {

  _timeline.executions.resize(num_workers);

  for(unsigned w=0; w<num_workers; ++w) {
    _timeline.executions[w].reserve(1024);
  }
  
  _timeline.origin = std::chrono::steady_clock::now();
}

// Procedure: on_entry
inline void ExecutorObserver::on_entry(unsigned w, TaskView tv) {
  _timeline.executions[w].emplace_back(tv, std::chrono::steady_clock::now());
}

// Procedure: on_exit
inline void ExecutorObserver::on_exit(unsigned w, TaskView tv) {
  static_cast<void>(tv);  // avoid warning from compiler
  assert(_timeline.executions[w].size() > 0);
  _timeline.executions[w].back().end = std::chrono::steady_clock::now();
}

// Function: clear
inline void ExecutorObserver::clear() {
  for(size_t w=0; w<_timeline.executions.size(); ++w) {
    _timeline.executions[w].clear();
  }
}

// Procedure: dump
inline void ExecutorObserver::dump(std::ostream& os) const {

  size_t first;

  for(first = 0; first<_timeline.executions.size(); ++first) {
    if(_timeline.executions[first].size() > 0) { 
      break; 
    }
  }

  os << '[';

  for(size_t w=first; w<_timeline.executions.size(); w++) {

    if(w != first && _timeline.executions[w].size() > 0) {
      os << ',';
    }

    for(size_t i=0; i<_timeline.executions[w].size(); i++) {

      os << '{'
         << "\"cat\":\"ExecutorObserver\","
         << "\"name\":\"" << _timeline.executions[w][i].task_view.name() << "\","
         << "\"ph\":\"X\","
         << "\"pid\":1,"
         << "\"tid\":" << w << ','
         << "\"ts\":" << std::chrono::duration_cast<std::chrono::microseconds>(
                           _timeline.executions[w][i].beg - _timeline.origin
                         ).count() << ','
         << "\"dur\":" << std::chrono::duration_cast<std::chrono::microseconds>(
                           _timeline.executions[w][i].end - _timeline.executions[w][i].beg
                         ).count();

      if(i != _timeline.executions[w].size() - 1) {
        os << "},";
      }
      else {
        os << '}';
      }
    }
  }
  os << "]\n";
}

// Function: dump
inline std::string ExecutorObserver::dump() const {
  std::ostringstream oss;
  dump(oss);
  return oss.str();
}

// Function: num_tasks
inline size_t ExecutorObserver::num_tasks() const {
  return std::accumulate(
    _timeline.executions.begin(), _timeline.executions.end(), size_t{0}, 
    [](size_t sum, const auto& exe){ 
      return sum + exe.size(); 
    }
  );
}


}  // end of namespace tf -------------------------------------------


