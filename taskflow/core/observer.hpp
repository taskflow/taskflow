#pragma once

#include "task.hpp"
#include "worker.hpp"

/** 
@file observer.hpp
@brief observer include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// timeline data structure
// ----------------------------------------------------------------------------

/**
@brief default time point type of observers
*/
using observer_stamp_t = std::chrono::time_point<std::chrono::steady_clock>;

/**
@private
*/
struct Segment {

  std::string name;
  TaskType type;

  observer_stamp_t beg;
  observer_stamp_t end;

  template <typename Archiver>
  auto save(Archiver& ar) const {
    return ar(name, type, beg, end);
  }

  template <typename Archiver>
  auto load(Archiver& ar) {
    return ar(name, type, beg, end);
  }

  Segment() = default;

  Segment(
    const std::string& n, TaskType t, observer_stamp_t b, observer_stamp_t e
  ) : name {n}, type {t}, beg {b}, end {e} {
  }

  auto span() const {
    return end-beg;
  } 
};

/**
@private
*/
struct Timeline {

  size_t uid;

  observer_stamp_t origin;
  std::vector<std::vector<std::vector<Segment>>> segments;

  Timeline() = default;

  Timeline(const Timeline& rhs) = delete;
  Timeline(Timeline&& rhs) = default;

  Timeline& operator = (const Timeline& rhs) = delete;
  Timeline& operator = (Timeline&& rhs) = default;

  template <typename Archiver>
  auto save(Archiver& ar) const {
    return ar(uid, origin, segments);
  }

  template <typename Archiver>
  auto load(Archiver& ar) {
    return ar(uid, origin, segments);
  }
};  

/**
@private
 */
struct ProfileData {

  std::vector<Timeline> timelines;

  ProfileData() = default;

  ProfileData(const ProfileData& rhs) = delete;
  ProfileData(ProfileData&& rhs) = default;

  ProfileData& operator = (const ProfileData& rhs) = delete;
  ProfileData& operator = (ProfileData&&) = default;
  
  template <typename Archiver>
  auto save(Archiver& ar) const {
    return ar(timelines);
  }

  template <typename Archiver>
  auto load(Archiver& ar) {
    return ar(timelines);
  }
};

// ----------------------------------------------------------------------------
// observer interface 
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

// ----------------------------------------------------------------------------
// TFProfObserver definition
// ----------------------------------------------------------------------------

/**
@class TFProfObserver

@brief class to create an observer based on the built-in taskflow profiler format

A tf::TFProfObserver inherits tf::ObserverInterface and defines methods to dump
the observed thread activities into a format that can be visualized through
@TFProf.

@code{.cpp}
tf::Taskflow taskflow;
tf::Executor executor;

// insert tasks into taskflow
// ...
  
// create a custom observer
std::shared_ptr<tf::TFProfObserver> observer = executor.make_observer<tf::TFProfObserver>();

// run the taskflow
executor.run(taskflow).wait();

// dump the thread activities to Taskflow Profiler format.
observer->dump(std::cout);
@endcode

*/
class TFProfObserver : public ObserverInterface {

  friend class Executor;
  friend class TFProfManager;

  /** @private overall task summary */
  struct TaskSummary {
    size_t count {0};
    size_t total_span {0};
    size_t min_span;
    size_t max_span;
    
    float avg_span() const { return total_span * 1.0f / count; }
  };

  /** @private worker summary at a level */
  struct WorkerSummary {

    size_t id;
    size_t level;
    size_t count {0};
    size_t total_span {0};
    size_t min_span{0};
    size_t max_span{0};

    std::array<TaskSummary, TASK_TYPES.size()> tsum;

    float avg_span() const { return total_span * 1.0f / count; }
    //return count < 2 ? 0.0f : total_delay * 1.0f / (count-1); 
  };
  
  /** @private */
  struct Summary {
    std::array<TaskSummary, TASK_TYPES.size()> tsum;
    std::vector<WorkerSummary> wsum;
    
    void dump_tsum(std::ostream&) const;
    void dump_wsum(std::ostream&) const;
    void dump(std::ostream&) const;
  };

  public:

    /**
    @brief dumps the timelines into a @TFProf format through 
           an output stream
    */
    void dump(std::ostream& ostream) const;

    /**
    @brief dumps the timelines into a JSON string
    */
    std::string dump() const;

    /**
    @brief shows the summary report through an output stream
    */
    void summary(std::ostream& ostream) const;

    /**
    @brief returns the summary report in a string
    */
    std::string summary() const;

    /**
    @brief clears the timeline data
    */
    void clear();

    /**
    @brief queries the number of tasks observed
    */
    size_t num_tasks() const;
    
    /**
    @brief queries the number of observed workers
    */
    size_t num_workers() const;

  private:
    
    Timeline _timeline;
  
    std::vector<std::stack<observer_stamp_t>> _stacks;
    
    inline void set_up(size_t num_workers) override final;
    inline void on_entry(WorkerView, TaskView) override final;
    inline void on_exit(WorkerView, TaskView) override final;
};  


// dump the task summary
inline void TFProfObserver::Summary::dump_tsum(std::ostream& os) const {

  // task summary
  size_t type_w{10}, count_w{5}, time_w{9}, avg_w{8}, min_w{8}, max_w{8};

  std::for_each(tsum.begin(), tsum.end(), [&](const auto& i){
    if(i.count == 0) return;
    count_w = std::max(count_w, std::to_string(i.count).size());
  });
  
  std::for_each(tsum.begin(), tsum.end(), [&](const auto& i){
    if(i.count == 0) return;
    time_w = std::max(time_w, std::to_string(i.total_span).size());
  });
  
  std::for_each(tsum.begin(), tsum.end(), [&](const auto& i){
    if(i.count == 0) return;
    avg_w = std::max(time_w, std::to_string(i.avg_span()).size());
  });
  
  std::for_each(tsum.begin(), tsum.end(), [&](const auto& i){
    if(i.count == 0) return;
    min_w = std::max(min_w, std::to_string(i.min_span).size());
  });
  
  std::for_each(tsum.begin(), tsum.end(), [&](const auto& i){
    if(i.count == 0) return;
    max_w = std::max(max_w, std::to_string(i.max_span).size());
  });

  os << std::setw(type_w) << "-Task-" 
     << std::setw(count_w+2) << "Count"
     << std::setw(time_w+2) << "Time (us)"
     << std::setw(avg_w+2) << "Avg (us)"
     << std::setw(min_w+2) << "Min (us)"
     << std::setw(max_w+2) << "Max (us)"
     << '\n';

  for(size_t i=0; i<TASK_TYPES.size(); i++) {
    if(tsum[i].count == 0) {
      continue;
    }
    os << std::setw(type_w) << to_string(TASK_TYPES[i])
       << std::setw(count_w+2) << tsum[i].count
       << std::setw(time_w+2) << tsum[i].total_span
       << std::setw(avg_w+2) << std::to_string(tsum[i].avg_span())
       << std::setw(min_w+2) << tsum[i].min_span
       << std::setw(max_w+2) << tsum[i].max_span
       << '\n';
  }
}

// dump the worker summary
inline void TFProfObserver::Summary::dump_wsum(std::ostream& os) const {
  
  // task summary
  size_t w_w{10}, t_w{10}, l_w{5}, c_w{5}, d_w{9}, avg_w{8}, min_w{8}, max_w{8};

  std::for_each(wsum.begin(), wsum.end(), [&](const auto& i){
    if(i.count == 0) return;
    l_w = std::max(l_w, std::to_string(i.level).size());
  });
  
  std::for_each(wsum.begin(), wsum.end(), [&](const auto& i){
    if(i.count == 0) return;
    c_w = std::max(c_w, std::to_string(i.count).size());
  });
  
  std::for_each(wsum.begin(), wsum.end(), [&](const auto& i){
    if(i.count == 0) return;
    d_w = std::max(d_w, std::to_string(i.total_span).size());
  });
  
  std::for_each(wsum.begin(), wsum.end(), [&](const auto& i){
    if(i.count == 0) return;
    avg_w = std::max(avg_w, std::to_string(i.avg_span()).size());
  });
  
  std::for_each(wsum.begin(), wsum.end(), [&](const auto& i){
    if(i.count == 0) return;
    min_w = std::max(min_w, std::to_string(i.min_span).size());
  });
  
  std::for_each(wsum.begin(), wsum.end(), [&](const auto& i){
    if(i.count == 0) return;
    max_w = std::max(max_w, std::to_string(i.max_span).size());
  });
  
  os << std::setw(w_w) << "-Worker-" 
     << std::setw(l_w+2) << "Level"
     << std::setw(t_w) << "Task"
     << std::setw(c_w+2) << "Count"
     << std::setw(d_w+2) << "Time (us)"
     << std::setw(avg_w+2) << "Avg (us)"
     << std::setw(min_w+2) << "Min (us)"
     << std::setw(max_w+2) << "Max (us)"
     << '\n';

  for(const auto& ws : wsum) {

    if(ws.count == 0) {
      continue;
    }

    os << std::setw(w_w) << ws.id
       << std::setw(l_w+2) << ws.level;
    
    bool first = true;
    for(size_t i=0; i<TASK_TYPES.size(); i++) {

      if(ws.tsum[i].count == 0) {
        continue;
      }

      os << (first ? std::setw(t_w) : std::setw(w_w + l_w + 2 + t_w));
      first = false;

      os << to_string(TASK_TYPES[i])
         << std::setw(c_w+2) << ws.tsum[i].count
         << std::setw(d_w+2) << ws.tsum[i].total_span
         << std::setw(avg_w+2) << std::to_string(ws.tsum[i].avg_span())
         << std::setw(min_w+2) << ws.tsum[i].min_span
         << std::setw(max_w+2) << ws.tsum[i].max_span
         << '\n';
    }

    // per-worker summary
    os << std::setw(w_w + l_w + t_w + c_w + 4) << ws.count
       << std::setw(d_w+2) << ws.total_span
       << std::setw(avg_w+2) << std::to_string(ws.avg_span())
       << std::setw(min_w+2) << ws.min_span
       << std::setw(max_w+2) << ws.max_span
       << '\n';
    
    //for(size_t j=0; j<w_w+l_w+t_w+4; j++) os << ' ';
    //for(size_t j=0; j<c_w+d_w+avg_w+min_w+max_w+8; j++) os << '-';
    //os <<'\n';
  }
}

// dump the summary report through an ostream
inline void TFProfObserver::Summary::dump(std::ostream& os) const {
  dump_tsum(os);
  os << '\n';
  dump_wsum(os);
}

// Procedure: set_up
inline void TFProfObserver::set_up(size_t num_workers) {
  _timeline.uid = unique_id<size_t>();
  _timeline.origin = observer_stamp_t::clock::now();
  _timeline.segments.resize(num_workers);
  _stacks.resize(num_workers);
}

// Procedure: on_entry
inline void TFProfObserver::on_entry(WorkerView wv, TaskView) {
  _stacks[wv.id()].push(observer_stamp_t::clock::now());
}

// Procedure: on_exit
inline void TFProfObserver::on_exit(WorkerView wv, TaskView tv) {

  size_t w = wv.id();

  assert(!_stacks[w].empty());
  
  if(_stacks[w].size() > _timeline.segments[w].size()) {
    _timeline.segments[w].resize(_stacks[w].size());
  }

  auto beg = _stacks[w].top();
  _stacks[w].pop();

  _timeline.segments[w][_stacks[w].size()].emplace_back(
    tv.name(), tv.type(), beg, observer_stamp_t::clock::now()
  );
}

// Function: clear
inline void TFProfObserver::clear() {
  for(size_t w=0; w<_timeline.segments.size(); ++w) {
    for(size_t l=0; l<_timeline.segments[w].size(); ++l) {
      _timeline.segments[w][l].clear();
    }
    while(!_stacks[w].empty()) {
      _stacks[w].pop();
    }
  }
}

// Procedure: dump
inline void TFProfObserver::dump(std::ostream& os) const {

  using namespace std::chrono;

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

  os << "{\"executor\":\"" << _timeline.uid << "\",\"data\":[";

  bool comma = false;

  for(size_t w=first; w<_timeline.segments.size(); w++) {
    for(size_t l=0; l<_timeline.segments[w].size(); l++) {

      if(_timeline.segments[w][l].empty()) {
        continue;
      }

      if(comma) {
        os << ',';
      }
      else {
        comma = true;
      }

      os << "{\"worker\":" << w << ",\"level\":" << l << ",\"data\":[";
      for(size_t i=0; i<_timeline.segments[w][l].size(); ++i) {

        const auto& s = _timeline.segments[w][l][i];

        if(i) os << ',';
        
        // span 
        os << "{\"span\":[" 
           << duration_cast<microseconds>(s.beg - _timeline.origin).count() 
           << ","
           << duration_cast<microseconds>(s.end - _timeline.origin).count() 
           << "],";
        
        // name
        os << "\"name\":\""; 
        if(s.name.empty()) {
          os << w << '_' << i;
        }
        else {
          os << s.name;
        }
        os << "\",";
    
        // e.g., category "type": "Condition Task"
        os << "\"type\":\"" << to_string(s.type) << "\"";

        os << "}";
      }
      os << "]}";
    }
  }

  os << "]}\n";
}

// Function: dump
inline std::string TFProfObserver::dump() const {
  std::ostringstream oss;
  dump(oss);
  return oss.str();
}

// Procedure: summary
inline void TFProfObserver::summary(std::ostream& os) const {

  using namespace std::chrono;
  
  Summary summary;
  std::optional<observer_stamp_t> view_beg, view_end;

  // find the first non-empty worker
  size_t first;
  for(first = 0; first<_timeline.segments.size(); ++first) {
    if(_timeline.segments[first].size() > 0) { 
      break; 
    }
  }
  
  // not timeline data to dump
  if(first == _timeline.segments.size()) {
    goto end_of_summary;
  }

  for(size_t w=first; w<_timeline.segments.size(); w++) {
    for(size_t l=0; l<_timeline.segments[w].size(); l++) {

      if(_timeline.segments[w][l].empty()) {
        continue;
      }

      // worker w at level l
      WorkerSummary ws;
      ws.id = w;
      ws.level = l;
      ws.count = _timeline.segments[w][l].size();
      
      // scan all tasks at level l
      for(size_t i=0; i<_timeline.segments[w][l].size(); ++i) {
        
        // update the entire span
        auto& s = _timeline.segments[w][l][i];
        view_beg = view_beg ? std::min(*view_beg, s.beg) : s.beg;
        view_end = view_end ? std::max(*view_end, s.end) : s.end;
        
        // update the task summary
        size_t t = duration_cast<microseconds>(s.end - s.beg).count();

        auto& x = summary.tsum[static_cast<int>(s.type)];
        x.count += 1;
        x.total_span += t;
        x.min_span = (x.count == 1) ? t : std::min(t, x.min_span);
        x.max_span = (x.count == 1) ? t : std::max(t, x.max_span);

        // update the worker summary
        ws.total_span += t;
        ws.min_span = (i == 0) ? t : std::min(t, ws.min_span);
        ws.max_span = (i == 0) ? t : std::max(t, ws.max_span);

        auto&y = ws.tsum[static_cast<int>(s.type)];
        y.count += 1;
        y.total_span += t;
        y.min_span = (y.count == 1) ? t : std::min(t, y.min_span);
        y.max_span = (y.count == 1) ? t : std::max(t, y.max_span);
        
        // update the delay
        //if(i) {
        //  size_t d = duration_cast<nanoseconds>(
        //    s.beg - _timeline.segments[w][l][i-1].end
        //  ).count();
        //  ws.total_delay += d;
        //  ws.min_delay = (i == 1) ? d : std::min(ws.min_delay, d);
        //  ws.max_delay = (i == 1) ? d : std::max(ws.max_delay, d);
        //}
      }
      summary.wsum.push_back(ws);
    }
  }

  end_of_summary:

  size_t view = 0;
  if(view_beg && view_end) {
    view = duration_cast<microseconds>(*view_end - *view_beg).count();
  }

  os << "==Observer " << _timeline.uid << ": "
     << num_workers() << " workers completed "
     << num_tasks() << " tasks in "
     << view << " us\n";

  summary.dump(os);
}

// Procedure: summary
inline std::string TFProfObserver::summary() const {
  std::ostringstream oss;
  summary(oss);
  return oss.str();
}

// Function: num_tasks
inline size_t TFProfObserver::num_tasks() const {
  size_t s = 0;
  for(size_t w=0; w<_timeline.segments.size(); ++w) {
    for(size_t l=0; l<_timeline.segments[w].size(); ++l) {
      s += _timeline.segments[w][l].size();
    }
  }
  return s;
}
  
// Function: num_workers
inline size_t TFProfObserver::num_workers() const {
  size_t w = 0;
  for(size_t i=0; i<_timeline.segments.size(); ++i) {
    w += (!_timeline.segments[i].empty());
  }
  return w;
}


// ----------------------------------------------------------------------------
// TFProfManager
// ----------------------------------------------------------------------------

/**
@private
*/
class TFProfManager {

  friend class Executor;

  public:
    
    ~TFProfManager();
    
    TFProfManager(const TFProfManager&) = delete;
    TFProfManager& operator=(const TFProfManager&) = delete;

    static TFProfManager& get();

    void dump(std::ostream& ostream) const;

  private:
    
    const std::string _fpath;

    std::mutex _mutex;
    std::vector<std::shared_ptr<TFProfObserver>> _observers;
    
    TFProfManager();

    void _manage(std::shared_ptr<TFProfObserver> observer);
};

// constructor
inline TFProfManager::TFProfManager() :
  _fpath {get_env(TF_ENABLE_PROFILER)} {

}

// Procedure: manage
inline void TFProfManager::_manage(std::shared_ptr<TFProfObserver> observer) {
  std::lock_guard lock(_mutex);
  _observers.push_back(std::move(observer));
}

// Procedure: dump
inline void TFProfManager::dump(std::ostream& os) const {
  for(size_t i=0; i<_observers.size(); ++i) {
    if(i) os << ',';
    _observers[i]->dump(os); 
  }
}

// Destructor
inline TFProfManager::~TFProfManager() {
  std::ofstream ofs(_fpath);
  if(ofs) {
    // .tfp
    if(_fpath.rfind(".tfp") != std::string::npos) {
      ProfileData data;
      data.timelines.reserve(_observers.size());
      for(size_t i=0; i<_observers.size(); ++i) {
        data.timelines.push_back(std::move(_observers[i]->_timeline));
      }
      Serializer<std::ofstream> serializer(ofs); 
      serializer(data);
    }
    // .json
    else { // if(_fpath.rfind(".json") != std::string::npos) {
      ofs << "[\n";
      for(size_t i=0; i<_observers.size(); ++i) {
        if(i) ofs << ',';
        _observers[i]->dump(ofs);
      }
      ofs << "]\n";
    }
  }
  // do a summary report in stderr for each observer
  else {
    std::ostringstream oss;
    for(size_t i=0; i<_observers.size(); ++i) {
      _observers[i]->summary(oss);
    }
    fprintf(stderr, "%s", oss.str().c_str());
  }
}
    
// Function: get
inline TFProfManager& TFProfManager::get() {
  static TFProfManager mgr;
  return mgr;
}

// ----------------------------------------------------------------------------
// Identifier for Each Built-in Observer
// ----------------------------------------------------------------------------

/** @enum ObserverType

@brief enumeration of all observer types

*/
enum class ObserverType : int {
  TFPROF = 0,
  CHROME,
  UNDEFINED
};

/**
@brief convert an observer type to a human-readable string
*/
inline const char* to_string(ObserverType type) {
  switch(type) {
    case ObserverType::TFPROF: return "tfprof";
    case ObserverType::CHROME: return "chrome";
    default:                   return "undefined";
  }
}


}  // end of namespace tf -----------------------------------------------------


