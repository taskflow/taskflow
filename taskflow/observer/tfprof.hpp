#pragma once

#include "interface.hpp"
#include "../utility/os.hpp"
#include "../utility/serializer.hpp"

/** 
@file observer/tfprof_observer.hpp
@brief TFProfObserver include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// timeline data structure
// ----------------------------------------------------------------------------

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

  public:


  struct TaskSummary {
    size_t count      {0};
    size_t total_span {0};
    size_t min_span   {0};  // valid when count > 0
    size_t max_span   {0};  // valid when count > 0

    double avg_span() const {
      return count ? static_cast<double>(total_span) / count : 0.0;
    }

    // update with a new observation of duration t (microseconds)
    void update(size_t t) {
      total_span += t;
      min_span = (count == 0) ? t : (std::min)(min_span, t);
      max_span = (count == 0) ? t : (std::max)(max_span, t);
      count += 1;
    }
  };

  /**
  @private
  @brief per-worker aggregated statistics, collapsed across all nesting levels
  
  busy_us  = sum of all task durations on this worker (across all levels)
  idle_us  = wall_us - busy_us  (wall_us = global max(end) - global min(beg))
  util_pct = busy_us / wall_us * 100
  */
  struct WorkerSummary {
    size_t id        {0};
    size_t count     {0};
    size_t busy_us   {0};
    size_t idle_us   {0};
    size_t min_span  {0};  // valid when count > 0
    size_t max_span  {0};  // valid when count > 0

    double avg_span() const {
      return count ? static_cast<double>(busy_us) / count : 0.0;
    }

    double util_pct(size_t wall_us) const {
      return wall_us ? static_cast<double>(busy_us) / wall_us * 100.0 : 0.0;
    }

    void update(size_t t) {
      busy_us += t;
      min_span = (count == 0) ? t : (std::min)(min_span, t);
      max_span = (count == 0) ? t : (std::max)(max_span, t);
      count += 1;
    }
  };

  /**
  @private
  @brief top-level summary container built by summary() and consumed by dump_*
  
  worker_histogram[b] = distinct workers active during bin b (0..num_workers).
  task_histogram[b]   = total concurrent tasks during bin b (can exceed num_workers).
  num_bins is computed at runtime from the 80-char line width and scaled tick label width.
  Each bin covers (wall_us / num_bins) microseconds of wall time.
  */
  struct Summary {
    std::array<TaskSummary,   TASK_TYPES.size()> tsum;
    std::vector<WorkerSummary>                   wsum;
    std::vector<size_t>                          worker_histogram;  // sized at runtime to num_bins
    std::vector<size_t>                          task_histogram;    // sized at runtime to num_bins

    size_t wall_us         {0};
    size_t num_all_workers {0};

    // avg_utilization = sum of (busy_us / wall_us) across ALL workers
    // (including idle ones) divided by num_all_workers.
    // this answers "how well did I use my thread pool overall?" — a worker
    // that ran no tasks contributes 0% to the sum, correctly dragging the
    // average down when the pool was underutilized.
    // result is in [0, 100]%.
    double avg_utilization() const {
      if(wall_us == 0 || num_all_workers == 0) return 0.0;
      double sum = 0.0;
      for(const auto& w : wsum) {
        sum += static_cast<double>(w.busy_us) / static_cast<double>(wall_us) * 100.0;
      }
      return sum / static_cast<double>(num_all_workers);
    }

    void dump_overview  (std::ostream&, size_t uid, size_t num_tasks) const;
    void dump_tsum      (std::ostream&) const;
    void dump_wsum      (std::ostream&) const;
    void dump_histogram (std::ostream&) const;
    void dump           (std::ostream&, size_t uid, size_t num_tasks) const;
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


// ----------------------------------------------------------------------------
// TFProfObserver::Summary output methods
// ----------------------------------------------------------------------------

// helper: emit a horizontal rule of width w
static inline void _tf_rule(std::ostream& os, size_t w, char c = '-') {
  for(size_t i = 0; i < w; ++i) os << c;
  os << '\n';
}

// Helper: _tf_time_scale
// Given a duration in microseconds, returns a human-readable scaled value
// and sets unit to the appropriate suffix string.
// Thresholds:
//   < 10,000 us  -> us  (microseconds)
//   < 10,000 ms  -> ms  (milliseconds)
//   < 10,000 s   -> s   (seconds)
//   otherwise    -> min (minutes)
static inline double _tf_time_scale(size_t us, const char*& unit) {
  if(us < 10000ULL) {
    unit = "us";
    return static_cast<double>(us);
  }
  if(us < 10000ULL * 1000ULL) {
    unit = "ms";
    return static_cast<double>(us) / 1e3;
  }
  if(us < 10000ULL * 1000ULL * 1000ULL) {
    unit = "s";
    return static_cast<double>(us) / 1e6;
  }
  unit = "min";
  return static_cast<double>(us) / 60e6;
}

// Procedure: dump_overview
// Emits the single-line header with wall time, worker count, task count,
// and average worker utilization — the single number that captures how
// well the thread pool was used (mean of busy/wall across active workers).
inline void TFProfObserver::Summary::dump_overview(
  std::ostream& os, size_t uid, size_t num_tasks
) const {
  const char* wall_unit;
  double wall_val = _tf_time_scale(wall_us, wall_unit);

  os << std::string(80, '=') << '\n';
  os << std::fixed << std::setprecision(2);
  os << " Observer "         << uid
     << " | Wall: "          << wall_val << " " << wall_unit
     << " | Workers: "       << num_all_workers
     << " | Tasks: "         << num_tasks
     << " | Avg Utilization: "<< avg_utilization() << "%\n";
  os << std::string(80, '=') << '\n';
}

// Procedure: dump_tsum
// Emits the aggregate task statistics table broken down by task type.
// Only task types that were actually observed are shown.
inline void TFProfObserver::Summary::dump_tsum(std::ostream& os) const {

  // check if there is anything to show
  bool any = false;
  for(const auto& t : tsum) { if(t.count) { any = true; break; } }
  if(!any) return;

  // column widths - compute from data so numbers never overflow their columns
  size_t type_w  = 9;
  size_t count_w = 5;
  size_t tot_w   = 10;
  size_t avg_w   = 10;
  size_t min_w   = 9;
  size_t max_w   = 9;

  for(const auto& t : tsum) {
    if(!t.count) continue;
    count_w = (std::max)(count_w, std::to_string(t.count).size());
    tot_w   = (std::max)(tot_w,   std::to_string(t.total_span).size());
    min_w   = (std::max)(min_w,   std::to_string(t.min_span).size());
    max_w   = (std::max)(max_w,   std::to_string(t.max_span).size());
  }

  os << "\n[Aggregate Task Statistics]\n";
  _tf_rule(os, 2 + type_w + count_w + tot_w + avg_w + min_w + max_w + 12);

  os << std::setw(type_w)  << "Type"
     << std::setw(count_w+2) << "Count"
     << std::setw(tot_w+2)   << "Total(us)"
     << std::setw(avg_w+2)   << "Avg(us)"
     << std::setw(min_w+2)   << "Min(us)"
     << std::setw(max_w+2)   << "Max(us)"
     << '\n';
  _tf_rule(os, 2 + type_w + count_w + tot_w + avg_w + min_w + max_w + 12);

  for(size_t i = 0; i < TASK_TYPES.size(); ++i) {
    const auto& t = tsum[i];
    if(!t.count) continue;
    os << std::setw(type_w)    << to_string(TASK_TYPES[i])
       << std::setw(count_w+2) << t.count
       << std::setw(tot_w+2)   << t.total_span
       << std::setw(avg_w+2)   << std::fixed << std::setprecision(2) << t.avg_span()
       << std::setw(min_w+2)   << t.min_span
       << std::setw(max_w+2)   << t.max_span
       << '\n';
  }
}

// Procedure: dump_wsum
// Emits the per-worker utilization table. Each row shows a worker's task
// count, total busy time, idle time, avg/min/max task duration, and
// utilization percentage. Workers that ran no tasks are skipped.
// A totals row at the bottom summarises across all active workers.
inline void TFProfObserver::Summary::dump_wsum(std::ostream& os) const {

  if(wsum.empty()) return;

  // column widths
  size_t w_w     = 6;
  size_t count_w = 5;
  size_t busy_w  = 9;
  size_t idle_w  = 9;
  size_t avg_w   = 9;
  size_t min_w   = 8;
  size_t max_w   = 8;
  size_t util_w  = 6;

  for(const auto& w : wsum) {
    w_w     = (std::max)(w_w,     std::to_string(w.id).size());
    count_w = (std::max)(count_w, std::to_string(w.count).size());
    busy_w  = (std::max)(busy_w,  std::to_string(w.busy_us).size());
    idle_w  = (std::max)(idle_w,  std::to_string(w.idle_us).size());
    min_w   = (std::max)(min_w,   std::to_string(w.min_span).size());
    max_w   = (std::max)(max_w,   std::to_string(w.max_span).size());
  }

  size_t row_w = w_w + count_w + busy_w + idle_w + avg_w + min_w + max_w + util_w + 16;

  os << "\n[Worker Utilization]\n";
  _tf_rule(os, row_w);

  os << std::setw(w_w+2)     << "Worker"
     << std::setw(count_w+2) << "Tasks"
     << std::setw(busy_w+2)  << "Busy(us)"
     << std::setw(idle_w+2)  << "Idle(us)"
     << std::setw(avg_w+2)   << "Avg(us)"
     << std::setw(min_w+2)   << "Min(us)"
     << std::setw(max_w+2)   << "Max(us)"
     << std::setw(util_w+2)  << "Util%"
     << '\n';
  _tf_rule(os, row_w);

  for(const auto& w : wsum) {
    os << std::setw(w_w+2)     << w.id
       << std::setw(count_w+2) << w.count
       << std::setw(busy_w+2)  << w.busy_us
       << std::setw(idle_w+2)  << w.idle_us
       << std::setw(avg_w+2)   << std::fixed << std::setprecision(2) << w.avg_span()
       << std::setw(min_w+2)   << w.min_span
       << std::setw(max_w+2)   << w.max_span
       << std::setw(util_w+2)  << std::fixed << std::setprecision(1)
                               << w.util_pct(wall_us) << "%"
       << '\n';
  }

  // totals row: Util% shows avg across ALL num_all_workers (including idle),
  // consistent with the header's "Avg Utilization" figure
  _tf_rule(os, row_w);
  size_t total_count = 0;
  size_t total_busy  = 0;
  size_t total_idle  = 0;
  for(const auto& w : wsum) {
    total_count += w.count;
    total_busy  += w.busy_us;
    total_idle  += w.idle_us;
  }
  os << std::setw(w_w+2)     << "Total"
     << std::setw(count_w+2) << total_count
     << std::setw(busy_w+2)  << total_busy
     << std::setw(idle_w+2)  << total_idle
     << std::setw(avg_w+2)   << ""
     << std::setw(min_w+2)   << ""
     << std::setw(max_w+2)   << ""
     << std::setw(util_w+2)  << std::fixed << std::setprecision(1)
                             << avg_utilization() << "% (avg)"
     << '\n';
}

// Helper: _dump_one_histogram
// Draws a single vertical bar chart for one histogram vector.
//   title    : label printed in the section header
//   hist     : histogram data (num_bins entries)
//   num_bins : number of bins (runtime value)
//   y_max    : the true maximum value (e.g. num_workers or peak task count)
//   bin_us   : time width of each bin in microseconds
//   col_w    : display column width per bin
//   bar_width: num_bins * col_w
//   wall_us  : total wall clock duration in microseconds
//
// Y axis is capped at MAX_Y_ROWS terminal rows regardless of y_max so that
// large workloads (e.g. millions of tasks) don't flood the terminal.
// Each display row represents a band of ceil(y_max / MAX_Y_ROWS) units;
// a bin fills row r if its value >= the row's lower threshold.
// Y axis labels show the upper bound of each row's band.
static inline void _dump_one_histogram(
  std::ostream&              os,
  const char*                title,
  const std::vector<size_t>& hist,
  size_t                     num_bins,
  size_t                     y_max,
  size_t                     bin_us,
  size_t                     col_w,
  size_t                     bar_width,
  size_t                     wall_us
) {
  static constexpr size_t MAX_Y_ROWS = 20;

  const char* wall_unit;
  const char* bin_unit;
  double wall_val = _tf_time_scale(wall_us, wall_unit);
  double bin_val  = _tf_time_scale(bin_us,  bin_unit);

  os << std::fixed << std::setprecision(2);
  os << "\n[" << title << "]  bin=" << bin_val << " " << bin_unit
     << "  (wall: 0.." << wall_val << " " << wall_unit << ","
     << " " << num_bins << " bins)\n";

  // scale Y: cap at MAX_Y_ROWS display rows, each covering y_step units
  size_t num_rows = (std::min)(y_max, MAX_Y_ROWS);
  if(num_rows == 0) num_rows = 1;
  size_t y_step = (y_max + num_rows - 1) / num_rows;
  if(y_step == 0) y_step = 1;

  // Y label width driven by y_max so labels never overflow their column
  size_t y_label_w = (std::max)(std::to_string(y_max).size(), size_t(4));
  size_t y_prefix  = y_label_w + 2;  // label + " |"

  // pre-build filled and empty column strings.
  // U+2588 FULL BLOCK encoded as raw UTF-8 bytes \xe2\x96\x88 to avoid
  // MSVC warning C4566 which triggers when the source code page (1252)
  // cannot represent the character named by a universal-character-name.
  std::string filled_col(1, ' ');
  for(size_t i = 0; i < col_w - 1; ++i) {
    filled_col += "\xe2\x96\x88";  // UTF-8 encoding of U+2588 FULL BLOCK
  }
  std::string empty_col(col_w, ' ');

  // bar rows from top (num_rows) down to 1.
  // row r represents the band [(r-1)*y_step+1 .. r*y_step].
  // a bin fills this row if hist[b] >= (r-1)*y_step + 1.
  // label shows r*y_step (capped at y_max for the top row).
  for(size_t r = num_rows; r >= 1; --r) {
    size_t threshold = (r - 1) * y_step + 1;
    size_t label_val = (std::min)(r * y_step, y_max);
    os << std::setw(static_cast<int>(y_label_w)) << label_val << " |";
    for(size_t b = 0; b < num_bins; ++b) {
      os << (hist[b] >= threshold ? filled_col : empty_col);
    }
    os << '\n';
  }

  // X axis rule, aligned to y_prefix
  os << std::string(y_prefix - 1, ' ') << '+'
     << std::string(bar_width, '-') << '\n';

  // tick label row: scale each tick to the bin unit, stamp at b*col_w+1
  double divisor = 1.0;
  const char* tick_unit = bin_unit;
  if     (std::string(bin_unit) == "ms")  divisor = 1e3;
  else if(std::string(bin_unit) == "s")   divisor = 1e6;
  else if(std::string(bin_unit) == "min") divisor = 60e6;

  std::string labels(bar_width, ' ');
  for(size_t b = 0; b < num_bins; ++b) {
    std::string tick;
    if(divisor == 1.0) {
      tick = std::to_string(b * bin_us);
    } else {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(1)
          << static_cast<double>(b * bin_us) / divisor;
      tick = oss.str();
    }
    size_t pos   = b * col_w + 1;
    size_t avail = (pos < bar_width) ? (std::min)(tick.size(), bar_width - pos) : 0;
    for(size_t i = 0; i < avail; ++i) {
      labels[pos + i] = tick[i];
    }
  }
  os << std::string(y_prefix, ' ') << labels << "  (" << tick_unit << ")\n";
}

// Procedure: dump_histogram
// Emits two bar charts stacked vertically:
//   1. Worker Concurrency  - Y = distinct workers active per bin (0..num_workers)
//   2. Task Parallelism    - Y = concurrent tasks per bin
//
// num_bins is computed at runtime:
//   1. Scale wall_us to find the display unit (us/ms/s/min)
//   2. Format the last tick label in that unit to find col_w (label width + 1 padding)
//   3. num_bins = (80 - Y_PREFIX) / col_w — fills the available 80-char width
// This makes the resolution as high as the terminal width allows.
inline void TFProfObserver::Summary::dump_histogram(std::ostream& os) const {

  if(wall_us == 0 || num_all_workers == 0 || worker_histogram.empty()) return;

  size_t num_bins = worker_histogram.size();  // set during build pass

  size_t bin_us = (wall_us + num_bins - 1) / num_bins;
  if(bin_us == 0) bin_us = 1;

  // col_w was already chosen when num_bins was computed in summary();
  // re-derive it here from the last tick width so the two agree exactly.
  const char* bin_unit;
  _tf_time_scale(bin_us, bin_unit);
  double divisor = 1.0;
  if     (std::string(bin_unit) == "ms")  divisor = 1e3;
  else if(std::string(bin_unit) == "s")   divisor = 1e6;
  else if(std::string(bin_unit) == "min") divisor = 60e6;

  std::string last_tick;
  if(divisor == 1.0) {
    last_tick = std::to_string((num_bins - 1) * bin_us);
  } else {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1)
        << static_cast<double>((num_bins - 1) * bin_us) / divisor;
    last_tick = oss.str();
  }

  size_t col_w     = (std::max)(last_tick.size() + 1, size_t(3));
  size_t bar_width = num_bins * col_w;

  size_t task_peak = *std::max_element(task_histogram.begin(), task_histogram.end());
  if(task_peak == 0) task_peak = 1;

  _dump_one_histogram(os,
    "Worker Concurrency  (Y=active workers)",
    worker_histogram, num_bins, num_all_workers, bin_us, col_w, bar_width, wall_us
  );

  _dump_one_histogram(os,
    "Task Parallelism  (Y=concurrent tasks)",
    task_histogram, num_bins, task_peak, bin_us, col_w, bar_width, wall_us
  );
}

// Procedure: dump
// Emits all four sections in order: overview header, task type stats,
// worker utilization table, concurrency histogram.
inline void TFProfObserver::Summary::dump(
  std::ostream& os, size_t uid, size_t num_tasks
) const {
  dump_overview  (os, uid, num_tasks);
  dump_tsum      (os);
  dump_wsum      (os);
  dump_histogram (os);
  os << '\n';
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
  summary.num_all_workers = _timeline.segments.size();

  // --- pass 1: scan all segments to compute time range, task stats,
  //             per-worker busy time, and global task type breakdown ---
  //
  // per-worker busy time is computed via a merge-intervals line sweep across
  // ALL nesting levels to correctly handle recursive tasking. a worker that
  // spawns subflow children has overlapping segments across levels — simply
  // summing durations would double-count the overlapping wall-clock time.
  // the sweep merges [beg, end] intervals per worker into non-overlapping
  // spans and sums only those, giving true wall-clock busy time in [0, wall_us].
  std::optional<observer_stamp_t> view_beg, view_end;

  // per-worker accumulators indexed by worker id; sized to num_workers
  std::vector<WorkerSummary> wmap(_timeline.segments.size());
  for(size_t w = 0; w < wmap.size(); ++w) {
    wmap[w].id = w;
  }

  // collect flat interval lists per worker (across all levels) for line sweep
  // intervals stored as (beg_us, end_us) relative to timeline origin
  // (we use the raw time points for sorting, convert to us after merging)
  using Interval = std::pair<observer_stamp_t, observer_stamp_t>;
  std::vector<std::vector<Interval>> worker_intervals(_timeline.segments.size());

  for(size_t w = 0; w < _timeline.segments.size(); ++w) {
    for(size_t l = 0; l < _timeline.segments[w].size(); ++l) {
      for(const auto& s : _timeline.segments[w][l]) {

        // update global time bounds (all levels)
        view_beg = view_beg ? (std::min)(*view_beg, s.beg) : s.beg;
        view_end = view_end ? (std::max)(*view_end, s.end) : s.end;

        size_t t = duration_cast<microseconds>(s.end - s.beg).count();

        // global per-type stats count all levels
        summary.tsum[static_cast<int>(s.type)].update(t);

        // update count/min/max on WorkerSummary (these are task-level stats,
        // not time stats, so all levels count)
        wmap[w].count += 1;
        wmap[w].min_span = (wmap[w].count == 1) ? t : (std::min)(wmap[w].min_span, t);
        wmap[w].max_span = (wmap[w].count == 1) ? t : (std::max)(wmap[w].max_span, t);

        // collect interval for line sweep
        worker_intervals[w].emplace_back(s.beg, s.end);
      }
    }
  }

  // nothing to show
  if(!view_beg || !view_end) {
    os << "==Observer " << _timeline.uid << ": no tasks recorded\n";
    return;
  }

  summary.wall_us = duration_cast<microseconds>(*view_end - *view_beg).count();
  if(summary.wall_us == 0) summary.wall_us = 1;

  // line sweep per worker: sort intervals by start time, merge overlaps,
  // sum merged span durations to get true wall-clock busy time
  for(size_t w = 0; w < worker_intervals.size(); ++w) {
    auto& ivs = worker_intervals[w];
    if(ivs.empty()) continue;

    // sort by beg
    std::sort(ivs.begin(), ivs.end(),
      [](const Interval& a, const Interval& b){ return a.first < b.first; });

    size_t busy_us = 0;
    auto cur_beg = ivs[0].first;
    auto cur_end = ivs[0].second;

    for(size_t i = 1; i < ivs.size(); ++i) {
      if(ivs[i].first <= cur_end) {
        // overlapping or adjacent: extend current merged interval
        cur_end = (std::max)(cur_end, ivs[i].second);
      }
      else {
        // gap: emit completed merged interval and start a new one
        busy_us += duration_cast<microseconds>(cur_end - cur_beg).count();
        cur_beg = ivs[i].first;
        cur_end = ivs[i].second;
      }
    }
    busy_us += duration_cast<microseconds>(cur_end - cur_beg).count();  // emit last

    wmap[w].busy_us = busy_us;
    wmap[w].idle_us = (summary.wall_us > busy_us) ? (summary.wall_us - busy_us) : 0;
    summary.wsum.push_back(wmap[w]);
  }

  // --- pass 2: build concurrency histograms ---
  //
  // num_bins is derived from the available 80-char line width and the width
  // of the widest scaled tick label, so the chart fills the terminal naturally.
  //
  // worker_histogram[b] = distinct workers with at least one task active in bin b.
  // task_histogram[b]   = total concurrent tasks in bin b.
  static constexpr size_t Y_PREFIX   = 6;
  static constexpr size_t LINE_WIDTH = 80;

  // compute num_bins: scale wall_us, format the last tick, derive col_w and num_bins
  {
    // start with a provisional num_bins to get the unit; iterate once to converge.
    // in practice the unit is stable after the first estimate.
    size_t probe_bins = 20;
    size_t probe_bin_us = (summary.wall_us + probe_bins - 1) / probe_bins;
    if(probe_bin_us == 0) probe_bin_us = 1;

    const char* bin_unit;
    _tf_time_scale(probe_bin_us, bin_unit);
    double divisor = 1.0;
    if     (std::string(bin_unit) == "ms")  divisor = 1e3;
    else if(std::string(bin_unit) == "s")   divisor = 1e6;
    else if(std::string(bin_unit) == "min") divisor = 60e6;

    // format the last tick at probe_bins-1 to get its rendered width
    std::string last_tick;
    if(divisor == 1.0) {
      last_tick = std::to_string((probe_bins - 1) * probe_bin_us);
    } else {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(1)
          << static_cast<double>((probe_bins - 1) * probe_bin_us) / divisor;
      last_tick = oss.str();
    }

    // col_w = label width + 1 space padding, minimum 3
    size_t col_w    = (std::max)(last_tick.size() + 1, size_t(3));
    size_t num_bins = (LINE_WIDTH - Y_PREFIX) / col_w;
    if(num_bins < 2) num_bins = 2;  // need at least 2 bins to be meaningful

    summary.worker_histogram.assign(num_bins, 0);
    summary.task_histogram.assign(num_bins, 0);
  }

  size_t num_bins = summary.worker_histogram.size();
  size_t bin_us   = (summary.wall_us + num_bins - 1) / num_bins;
  if(bin_us == 0) bin_us = 1;

  size_t nw = _timeline.segments.size();

  // active[b][w] = true if worker w had any task overlapping bin b
  std::vector<std::vector<bool>> active(num_bins, std::vector<bool>(nw, false));

  for(size_t w = 0; w < nw; ++w) {
    for(size_t l = 0; l < _timeline.segments[w].size(); ++l) {
      for(const auto& s : _timeline.segments[w][l]) {

        size_t s_beg = duration_cast<microseconds>(s.beg - *view_beg).count();
        size_t s_end = duration_cast<microseconds>(s.end - *view_beg).count();

        size_t first_bin = s_beg / bin_us;
        size_t last_bin  = (s_end == 0) ? 0 : (s_end - 1) / bin_us;
        first_bin = (std::min)(first_bin, num_bins - 1);
        last_bin  = (std::min)(last_bin,  num_bins - 1);

        for(size_t b = first_bin; b <= last_bin; ++b) {
          active[b][w] = true;
          summary.task_histogram[b] += 1;
        }
      }
    }
  }

  for(size_t b = 0; b < num_bins; ++b) {
    for(size_t w = 0; w < nw; ++w) {
      summary.worker_histogram[b] += active[b][w] ? 1 : 0;
    }
  }

  summary.dump(os, _timeline.uid, num_tasks());
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
