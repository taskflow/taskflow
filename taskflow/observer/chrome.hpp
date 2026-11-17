#pragma once

#include "interface.hpp"

/**
@file chrome.hpp
@brief ChromeObserver include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// ChromeObserver definition
// ----------------------------------------------------------------------------

/**
@class: ChromeObserver

@brief class to create an observer based on Chrome tracing format

A tf::ChromeObserver inherits tf::ObserverInterface and defines methods to dump
the observed thread activities into a format that can be visualized through
@ChromeTracing.

@code{.cpp}
tf::Taskflow taskflow;
tf::Executor executor;

// insert tasks into taskflow
// ...
  
// create a custom observer
std::shared_ptr<tf::ChromeObserver> observer = executor.make_observer<tf::ChromeObserver>();

// run the taskflow
executor.run(taskflow).wait();

// dump the thread activities to a chrome-tracing format.
observer->dump(std::cout);
@endcode
*/
class ChromeObserver : public ObserverInterface {

  friend class Executor;
  
  // data structure to record each task execution
  struct Segment {

    std::string name;

    observer_stamp_t beg;
    observer_stamp_t end;

    Segment(
      const std::string& n,
      observer_stamp_t b,
      observer_stamp_t e
    );
  };
  
  // data structure to store the entire execution timeline
  struct Timeline {
    observer_stamp_t origin;
    std::vector<std::vector<Segment>> segments;
    std::vector<std::stack<observer_stamp_t>> stacks;
  };  

  public:

    /**
    @brief dumps the timelines into a @ChromeTracing format through 
           an output stream 
    */
    void dump(std::ostream& ostream) const;

    /**
    @brief dumps the timelines into a @ChromeTracing format
    */
    inline std::string dump() const;

    /**
    @brief clears the timeline data
    */
    inline void clear();

    /**
    @brief queries the number of tasks observed
    */
    inline size_t num_tasks() const;

  private:
    
    inline void set_up(size_t num_workers) override final;
    inline void on_entry(WorkerView w, TaskView task_view) override final;
    inline void on_exit(WorkerView w, TaskView task_view) override final;

    Timeline _timeline;
};  
    
// constructor
inline ChromeObserver::Segment::Segment(
  const std::string& n, observer_stamp_t b, observer_stamp_t e
) :
  name {n}, beg {b}, end {e} {
}

// Procedure: set_up
inline void ChromeObserver::set_up(size_t num_workers) {
  _timeline.segments.resize(num_workers);
  _timeline.stacks.resize(num_workers);

  for(size_t w=0; w<num_workers; ++w) {
    _timeline.segments[w].reserve(32);
  }
  
  _timeline.origin = observer_stamp_t::clock::now();
}

// Procedure: on_entry
inline void ChromeObserver::on_entry(WorkerView wv, TaskView) {
  _timeline.stacks[wv.id()].push(observer_stamp_t::clock::now());
}

// Procedure: on_exit
inline void ChromeObserver::on_exit(WorkerView wv, TaskView tv) {

  size_t w = wv.id();

  assert(!_timeline.stacks[w].empty());

  auto beg = _timeline.stacks[w].top();
  _timeline.stacks[w].pop();

  _timeline.segments[w].emplace_back(
    tv.name(), beg, observer_stamp_t::clock::now()
  );
}

// Function: clear
inline void ChromeObserver::clear() {
  for(size_t w=0; w<_timeline.segments.size(); ++w) {
    _timeline.segments[w].clear();
    while(!_timeline.stacks[w].empty()) {
      _timeline.stacks[w].pop();
    }
  }
}

// Procedure: dump
inline void ChromeObserver::dump(std::ostream& os) const {

  using namespace std::chrono;

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

      os << '{'<< "\"cat\":\"ChromeObserver\",";

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
         << "\"ts\":" << duration_cast<microseconds>(
                           _timeline.segments[w][i].beg - _timeline.origin
                         ).count() << ','
         << "\"dur\":" << duration_cast<microseconds>(
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
inline std::string ChromeObserver::dump() const {
  std::ostringstream oss;
  dump(oss);
  return oss.str();
}

// Function: num_tasks
inline size_t ChromeObserver::num_tasks() const {
  return std::accumulate(
    _timeline.segments.begin(), _timeline.segments.end(), size_t{0}, 
    [](size_t sum, const auto& exe){ 
      return sum + exe.size(); 
    }
  );
}

}  // end of namespace tf -----------------------------------------------------
