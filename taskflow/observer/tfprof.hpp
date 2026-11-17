#pragma once

#include "interface.hpp"
#include "../utility/os.hpp"

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
    void dump           (std::ostream&, size_t uid, size_t num_tasks) const;
  };

  public:

    /**
    @brief dumps the timelines into a @TFProf format through 
           an output stream
    */

    /**
    @brief dumps the timelines into a JSON string
    */

    /**
    @brief dumps this executor's data as a self-contained .tfp executor block.
    Writes its own string table followed by all worker-level segment data.
    Called by TFProfManager::dump after the file header is written.
    */
    void dump(std::ostream& ostream) const;

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

    // -----------------------------------------------------------------------
    // .tfp binary format helpers
    // Kept private so they have direct access to _timeline without
    // any friend declarations or public/protected exposure.
    // -----------------------------------------------------------------------

    // Portable little-endian write helpers
    static void _tfp_write_u8 (std::ostream& os, uint8_t  v) {
      os.put(static_cast<char>(v));
    }
    static void _tfp_write_u16(std::ostream& os, uint16_t v) {
      uint8_t buf[2] = { uint8_t(v), uint8_t(v >> 8) };
      os.write(reinterpret_cast<const char*>(buf), 2);
    }
    static void _tfp_write_u32(std::ostream& os, uint32_t v) {
      uint8_t buf[4] = {
        uint8_t(v),       uint8_t(v >> 8),
        uint8_t(v >> 16), uint8_t(v >> 24)
      };
      os.write(reinterpret_cast<const char*>(buf), 4);
    }
    static void _tfp_write_u64(std::ostream& os, uint64_t v) {
      uint8_t buf[8] = {
        uint8_t(v),       uint8_t(v >> 8),
        uint8_t(v >> 16), uint8_t(v >> 24),
        uint8_t(v >> 32), uint8_t(v >> 40),
        uint8_t(v >> 48), uint8_t(v >> 56)
      };
      os.write(reinterpret_cast<const char*>(buf), 8);
    }

    // Variable-length unsigned integer (varint) — LEB128 encoding.
    // Values 0-127 encode in 1 byte. Each byte uses 7 bits of data;
    // the MSB is a continuation flag (1 = more bytes follow).
    // Typical delta timestamps (1-1000 µs) encode in 1-2 bytes vs 8 bytes fixed.
    static void _tfp_write_varint(std::ostream& os, uint64_t v) {
      do {
        uint8_t byte = v & 0x7F;
        v >>= 7;
        if(v) byte |= 0x80;   // set continuation bit
        os.put(static_cast<char>(byte));
      } while(v);
    }

    // TaskType -> uint8 (must match JS TYPE_NAMES array order:
    //   0=static 1=subflow 2=condition 3=module 4=async)
    static uint8_t _tfp_type_byte(TaskType t) {
      switch(t) {
        case TaskType::STATIC:    return 0;
        case TaskType::SUBFLOW:   return 1;
        case TaskType::CONDITION: return 2;
        case TaskType::MODULE:    return 3;
        case TaskType::ASYNC:     return 4;
        default:                  return 0;
      }
    }

    // Deduplicating string table builder (used by dump)
    struct _StringTable {
      std::vector<char>                         data;
      std::unordered_map<std::string, uint32_t> idx;

      // Returns {byte_offset, clamped_length}.
      // Empty name => {0, 0} meaning anonymous; viewer generates "W{w}_{i}".
      std::pair<uint32_t, uint8_t> intern(const std::string& name) {
        if(name.empty()) return { uint32_t{0}, uint8_t{0} };
        const uint8_t len = static_cast<uint8_t>(
          name.size() > 31 ? 31 : name.size()
        );
        auto it = idx.find(name);
        if(it != idx.end()) return { it->second, len };
        const uint32_t off = static_cast<uint32_t>(data.size());
        data.insert(data.end(), name.begin(), name.begin() + len);
        idx.emplace(name, off);
        return { off, len };
      }
    };
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

// Procedure: dump
// Emits overview header, task type stats, and worker utilization table.
inline void TFProfObserver::Summary::dump(
  std::ostream& os, size_t uid, size_t num_tasks
) const {
  dump_overview  (os, uid, num_tasks);
  dump_tsum      (os);
  dump_wsum      (os);
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
// Writes this executor's data as a self-contained block:
//
// FILE FORMAT (all integers little-endian):
//   Executor header (24 bytes):
//     uid          : u64   — unique executor id
//     origin_us    : u64   — wall-clock origin (µs since epoch)
//     str_table_len: u32   — byte length of the string table below
//     num_wl       : u32   — number of non-empty (worker, level) blocks
//   String table   : str_table_len bytes of flat UTF-8 task names
//   Per non-empty (worker, level):
//     Worker-level header (12 bytes): worker_id(u32) level(u32) num_segs(u32)
//     Per segment (variable length, delta+varint encoded):
//       delta_beg  : varint  — beg_us minus previous beg_us (0 for first)
//       duration   : varint  — (end_us - beg_us), always non-negative
//       name_off   : u32     — byte offset into this executor's string table
//       type|nlen  : u8      — upper 3 bits = task type, lower 5 = name length
//
// Each executor block is fully self-contained with its own string table,
// so TFProfManager::dump can simply write the file header and call dump()
// on each observer with no shared state.
inline void TFProfObserver::dump(std::ostream& os) const {
  using namespace std::chrono;
  using SegRef = std::pair<uint32_t, uint8_t>;

  // ---- pass 1: build this executor's string table ----
  _StringTable strtab;
  const size_t nw = _timeline.segments.size();
  std::vector<std::vector<std::vector<SegRef>>> segRefs(nw);
  for(size_t w = 0; w < nw; ++w) {
    segRefs[w].resize(_timeline.segments[w].size());
    for(size_t l = 0; l < _timeline.segments[w].size(); ++l) {
      const auto& segs = _timeline.segments[w][l];
      segRefs[w][l].resize(segs.size());
      for(size_t s = 0; s < segs.size(); ++s) {
        segRefs[w][l][s] = strtab.intern(segs[s].name);
      }
    }
  }

  // ---- count non-empty worker-level blocks ----
  uint32_t numWL = 0;
  for(size_t w = 0; w < nw; ++w)
    for(size_t l = 0; l < _timeline.segments[w].size(); ++l)
      if(!_timeline.segments[w][l].empty()) ++numWL;

  // ---- executor header (24 bytes) ----
  const int64_t origin_us = duration_cast<microseconds>(
    _timeline.origin.time_since_epoch()
  ).count();
  _tfp_write_u64(os, static_cast<uint64_t>(_timeline.uid));
  _tfp_write_u64(os, static_cast<uint64_t>(origin_us));
  _tfp_write_u32(os, static_cast<uint32_t>(strtab.data.size())); // str_table_len
  _tfp_write_u32(os, numWL);

  // ---- string table ----
  if(!strtab.data.empty()) {
    os.write(strtab.data.data(), static_cast<std::streamsize>(strtab.data.size()));
  }

  // ---- worker-level blocks ----
  for(size_t w = 0; w < nw; ++w) {
    for(size_t l = 0; l < _timeline.segments[w].size(); ++l) {
      const auto& segs = _timeline.segments[w][l];
      if(segs.empty()) continue;

      _tfp_write_u32(os, static_cast<uint32_t>(w));
      _tfp_write_u32(os, static_cast<uint32_t>(l));
      _tfp_write_u32(os, static_cast<uint32_t>(segs.size()));

      int64_t prev_beg = 0;
      for(size_t s = 0; s < segs.size(); ++s) {
        const auto& seg  = segs[s];
        const auto& ref  = segRefs[w][l][s];
        const int64_t beg_us  = duration_cast<microseconds>(seg.beg - _timeline.origin).count();
        const int64_t end_us  = duration_cast<microseconds>(seg.end - _timeline.origin).count();
        _tfp_write_varint(os, static_cast<uint64_t>(beg_us - prev_beg)); // delta_beg
        _tfp_write_varint(os, static_cast<uint64_t>(end_us - beg_us));   // duration
        _tfp_write_u32(os, ref.first);                                    // name_off
        _tfp_write_u8 (os, static_cast<uint8_t>(
          (_tfp_type_byte(seg.type) << 5) | (ref.second & 0x1F)
        ));                                                               // type|name_len
        prev_beg = beg_us;
      }
    }
  }
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


  private:
    
    const std::string _fpath;

    std::mutex _mutex;
    std::vector<std::shared_ptr<TFProfObserver>> _observers;
    
    TFProfManager();

    void _manage(std::shared_ptr<TFProfObserver> observer);

    // Writes a complete .tfp file covering all observers to 'os'.
    // Builds a single global string table across all observers (maximises
    // deduplication), then writes the file header, string table, and each
    // executor block by calling into TFProfObserver's private helpers.
    void dump(std::ostream& os) const;
};

// Constructor
inline TFProfManager::TFProfManager() :
  _fpath {get_env(TF_ENABLE_PROFILER)} {
}

// Procedure: _manage
inline void TFProfManager::_manage(std::shared_ptr<TFProfObserver> observer) {
  std::lock_guard lock(_mutex);
  _observers.push_back(std::move(observer));
}

// Procedure: dump (JSON, used for streaming / manual calls)


// Procedure: dump
// Writes a complete .tfp file to os.
// File header (12 bytes): magic(4) version(u16) flags(u16) num_exec(u32)
// Followed by each executor's self-contained block (see TFProfObserver::dump).
inline void TFProfManager::dump(std::ostream& os) const {
  // File header — 12 bytes
  os.write("TFPX", 4);
  TFProfObserver::_tfp_write_u16(os, 1);                                        // version
  TFProfObserver::_tfp_write_u16(os, 0);                                        // flags
  TFProfObserver::_tfp_write_u32(os, static_cast<uint32_t>(_observers.size())); // num_exec

  // Each observer writes its own self-contained executor block
  for(size_t i = 0; i < _observers.size(); ++i) {
    _observers[i]->dump(os);
  }
}

// Destructor
// If a file path was given, write the binary .tfp file.
// Otherwise print a text summary to stderr.
inline TFProfManager::~TFProfManager() {
  std::ofstream ofs(_fpath, std::ios::binary);
  if(ofs) {
    dump(ofs);
  }
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
