// 2020/04/30 - midified by Tsung-Wei Huang
//  - adding TaskflowBoard support
//
// 2019/07/31 - modified by Tsung-Wei Huang
//  - fixed the missing comma in outputing JSON  
//
// 2019/06/13 - modified by Tsung-Wei Huang
//  - added TaskView interface
//
// 2019/04/17 - created by Tsung-Wei Huang

#pragma once

#include "task.hpp"

namespace tf {

/**
@class: ObserverInterface

@brief The interface class for creating an executor observer.

The tf::ExecutorObserver class let users define methods to monitor the behaviors
of an executor. 
This is particularly useful when you want to inspect the performance of an executor.
*/
class ObserverInterface {

  friend class Executor;
  
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
  @param worker_id the id of this worker thread 
  @param task_view a constant wrapper object to the task 
  */
  virtual void on_entry(size_t worker_id, TaskView task_view) = 0;
  
  /**
  @brief method to call after a worker thread executed a closure
  @param worker_id the id of this worker thread 
  @param task_view a constant wrapper object to the task
  */
  virtual void on_exit(size_t worker_id, TaskView task_view) = 0;
};

// ----------------------------------------------------------------------------
// ChromeTracingObserver definition
// ----------------------------------------------------------------------------

/**
@class: ChromeTracingObserver

@brief observer designed based on chrome tracing format

*/
class ChromeTracingObserver : public ObserverInterface {

  friend class Executor;
  
  // data structure to record each task execution
  struct Segment {

    std::string name;

    std::chrono::time_point<std::chrono::steady_clock> beg;
    std::chrono::time_point<std::chrono::steady_clock> end;

    Segment(
      const std::string& n,
      std::chrono::time_point<std::chrono::steady_clock> b
    );

    Segment(
      const std::string& n,
      std::chrono::time_point<std::chrono::steady_clock> b,
      std::chrono::time_point<std::chrono::steady_clock> e
    );
  };
  
  // data structure to store the entire execution timeline
  struct Timeline {
    std::chrono::time_point<std::chrono::steady_clock> origin;
    std::vector<std::vector<Segment>> segments;
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
    
    inline void set_up(size_t num_workers) override final;
    inline void on_entry(size_t worker_id, TaskView task_view) override final;
    inline void on_exit(size_t worker_id, TaskView task_view) override final;

    Timeline _timeline;
};  
    
// constructor
inline ChromeTracingObserver::Segment::Segment(
  const std::string& n,
  std::chrono::time_point<std::chrono::steady_clock> b
) :
  name {n}, beg {b} {
} 

// constructor
inline ChromeTracingObserver::Segment::Segment(
  const std::string& n,
  std::chrono::time_point<std::chrono::steady_clock> b,
  std::chrono::time_point<std::chrono::steady_clock> e
) :
  name {n}, beg {b}, end {e} {
}

// Procedure: set_up
inline void ChromeTracingObserver::set_up(size_t num_workers) {

  _timeline.segments.resize(num_workers);

  for(size_t w=0; w<num_workers; ++w) {
    _timeline.segments[w].reserve(32);
  }
  
  _timeline.origin = std::chrono::steady_clock::now();
}

// Procedure: on_entry
inline void ChromeTracingObserver::on_entry(size_t w, TaskView tv) {
  _timeline.segments[w].emplace_back(
    tv.name(), std::chrono::steady_clock::now()
  );
}

// Procedure: on_exit
inline void ChromeTracingObserver::on_exit(size_t w, TaskView) {
  assert(_timeline.segments[w].size() > 0);
  _timeline.segments[w].back().end = std::chrono::steady_clock::now();
}

// Function: clear
inline void ChromeTracingObserver::clear() {
  for(size_t w=0; w<_timeline.segments.size(); ++w) {
    _timeline.segments[w].clear();
  }
}

// Procedure: dump
inline void ChromeTracingObserver::dump(std::ostream& os) const {

  size_t first;

  for(first = 0; first<_timeline.segments.size(); ++first) {
    if(_timeline.segments[first].size() > 0) { 
      break; 
    }
  }

  os << '[';

  for(size_t w=first; w<_timeline.segments.size(); w++) {

    if(w != first && _timeline.segments[w].size() > 0) {
      os << ',';
    }

    for(size_t i=0; i<_timeline.segments[w].size(); i++) {

      os << '{'
         << "\"cat\":\"ChromeTracingObserver\",";

      // name field
      os << "\"name\":\"";
      if(_timeline.segments[w][i].name.empty()) {
        os << w << '_' << i;
      }
      else {
        os << _timeline.segments[w][i].name;
      }
      os << "\",";
      
      // segment field
      os << "\"ph\":\"X\","
         << "\"pid\":1,"
         << "\"tid\":" << w << ','
         << "\"ts\":" << std::chrono::duration_cast<std::chrono::microseconds>(
                           _timeline.segments[w][i].beg - _timeline.origin
                         ).count() << ','
         << "\"dur\":" << std::chrono::duration_cast<std::chrono::microseconds>(
                           _timeline.segments[w][i].end - _timeline.segments[w][i].beg
                         ).count();

      if(i != _timeline.segments[w].size() - 1) {
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
inline std::string ChromeTracingObserver::dump() const {
  std::ostringstream oss;
  dump(oss);
  return oss.str();
}

// Function: num_tasks
inline size_t ChromeTracingObserver::num_tasks() const {
  return std::accumulate(
    _timeline.segments.begin(), _timeline.segments.end(), size_t{0}, 
    [](size_t sum, const auto& exe){ 
      return sum + exe.size(); 
    }
  );
}

// ----------------------------------------------------------------------------
// TFProfObserver definition
// ----------------------------------------------------------------------------

/**
@class: TFProfObserver

@brief observer designed based on taskflow board format

*/
class TFProfObserver : public ObserverInterface {

  friend class Executor;
  
  // data structure to record each task execution
  struct Segment {

    std::string name;
    TaskType type;

    std::chrono::time_point<std::chrono::steady_clock> beg;
    std::chrono::time_point<std::chrono::steady_clock> end;

    Segment(
      const std::string& n,
      TaskType t,
      std::chrono::time_point<std::chrono::steady_clock> b
    );

    Segment(
      const std::string& n,
      TaskType t,
      std::chrono::time_point<std::chrono::steady_clock> b,
      std::chrono::time_point<std::chrono::steady_clock> e
    );
  };
  
  // data structure to store the entire execution timeline
  struct Timeline {
    std::chrono::time_point<std::chrono::steady_clock> origin;
    std::vector<std::vector<Segment>> segments;
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
    
    inline void set_up(size_t num_workers) override final;
    inline void on_entry(size_t worker_id, TaskView task_view) override final;
    inline void on_exit(size_t worker_id, TaskView task_view) override final;

    Timeline _timeline;

    UUID _uuid;
};  

// constructor
inline TFProfObserver::Segment::Segment(
  const std::string& n,
  TaskType t,
  std::chrono::time_point<std::chrono::steady_clock> b
) :
  name {n}, type {t}, beg {b} {
} 

// constructor
inline TFProfObserver::Segment::Segment(
  const std::string& n,
  TaskType t,
  std::chrono::time_point<std::chrono::steady_clock> b,
  std::chrono::time_point<std::chrono::steady_clock> e
) :
  name {n}, type {t}, beg {b}, end {e} {
}

// Procedure: set_up
inline void TFProfObserver::set_up(size_t num_workers) {

  _timeline.segments.resize(num_workers);

  for(size_t w=0; w<num_workers; ++w) {
    _timeline.segments[w].reserve(32);
  }
  
  _timeline.origin = std::chrono::steady_clock::now();
}

// Procedure: on_entry
inline void TFProfObserver::on_entry(size_t w, TaskView tv) {
  _timeline.segments[w].emplace_back(
    tv.name(), tv.type(), std::chrono::steady_clock::now()
  );
}

// Procedure: on_exit
inline void TFProfObserver::on_exit(size_t w, TaskView) {
  assert(_timeline.segments[w].size() > 0);
  _timeline.segments[w].back().end = std::chrono::steady_clock::now();
}

// Function: clear
inline void TFProfObserver::clear() {
  for(size_t w=0; w<_timeline.segments.size(); ++w) {
    _timeline.segments[w].clear();
  }
}

// Procedure: dump
inline void TFProfObserver::dump(std::ostream& os) const {

  size_t first;

  for(first = 0; first<_timeline.segments.size(); ++first) {
    if(_timeline.segments[first].size() > 0) { 
      break; 
    }
  }
  
  // not timeline data to dump
  if(first == _timeline.segments.size()) {
    os << "{}\n";
    return;
  }

  os << "{\"executor\":\"" << _uuid << "\",\"data\":[";

  for(size_t w=first; w<_timeline.segments.size(); w++) {

    if(_timeline.segments[w].empty()) {
      continue;
    }

    if(w != first) {
      os << ',';
    }

    os << "{\"worker\":\"worker " << w << "\",\"data\":[";
    for(size_t i=0; i<_timeline.segments[w].size(); ++i) {

      const auto& s = _timeline.segments[w][i];

      if(i) os << ',';
      
      // span 
      os << "{\"span\":[" 
         << std::chrono::duration_cast<std::chrono::microseconds>(
              s.beg - _timeline.origin
            ).count() << ","
         << std::chrono::duration_cast<std::chrono::microseconds>(
              s.end - _timeline.origin
            ).count() << "],";
      
      // name
      os << "\"name\":\""; 
      if(s.name.empty()) {
        os << w << '_' << i;
      }
      else {
        os << s.name;
      }
      os << "\",";
  
      // category "type": "Condition Task",
      os << "\"type\":\"" << task_type_to_string(s.type) << "\"";

      os << "}";
    }
    os << "]}";
  }

  os << "]}\n";
}

// Function: dump
inline std::string TFProfObserver::dump() const {
  std::ostringstream oss;
  dump(oss);
  return oss.str();
}

// Function: num_tasks
inline size_t TFProfObserver::num_tasks() const {
  return std::accumulate(
    _timeline.segments.begin(), _timeline.segments.end(), size_t{0}, 
    [](size_t sum, const auto& exe){ 
      return sum + exe.size(); 
    }
  );
}

// ----------------------------------------------------------------------------
// Identifier for Each Built-in Observer
// ----------------------------------------------------------------------------

/** @enum ObserverType

built-in observer types

*/
enum ObserverType {
  TFPROF = 1,
  CHROME = 2
};

/**
@brief convert an observer type to a human-readable string
*/
inline const char* observer_type_to_string(ObserverType type) {
  const char* val;
  switch(type) {
    case TFPROF: val = "TFProf";    break;
    case CHROME: val = "Chrome";    break;
    default:     val = "undefined"; break;
  }
  return val;
}

// ----------------------------------------------------------------------------
// Legacy Alias
// ----------------------------------------------------------------------------
using ExecutorObserverInterface = ObserverInterface;
using ExecutorObserver          = ChromeTracingObserver;


}  // end of namespace tf -----------------------------------------------------


