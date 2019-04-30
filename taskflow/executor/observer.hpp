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
#include <optional>
#include <thread>
#include <algorithm>
#include <set>
#include <numeric>
#include <cassert>

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
  virtual void set_up(unsigned num_workers) {};

  /**
  @brief method to call before a worker thread executes a closure
  @param worker_id the id of this worker thread
  */
  virtual void on_entry(unsigned worker_id, std::string_view name) {};

  /**
  @brief method to call after a worker thread executed a closure
  @param worker_id the id of this worker thread
  */
  virtual void on_exit(unsigned worker_id) {};
};

// ------------------------------------------------------------------

/**
@class: ExecutorObserver

@brief A default executor observer to dump the execution timelines

*/
class ExecutorObserver : public ExecutorObserverInterface {

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
    inline void on_entry(unsigned worker_id, std::string_view name) override final;
    inline void on_exit(unsigned worker_id) override final;

    std::chrono::time_point<std::chrono::steady_clock> _origin {std::chrono::steady_clock::now()};

    std::vector<std::vector<std::chrono::time_point<std::chrono::steady_clock>>> _begs;
    std::vector<std::vector<std::chrono::time_point<std::chrono::steady_clock>>> _ends;
    std::vector<std::vector<std::string_view>> _names;
};

// Procedure: set_up
inline void ExecutorObserver::set_up(unsigned num_workers) {
  _begs.resize(num_workers);
  _ends.resize(num_workers);
  _names.resize(num_workers);
  for(unsigned w=0; w<num_workers; ++w) {
    _begs[w].reserve(1024);
    _ends[w].reserve(1024);
    _names[w].reserve(1024);
  }
}

// Procedure: on_entry
inline void ExecutorObserver::on_entry(unsigned w, std::string_view name) {
  _begs[w].emplace_back(std::chrono::steady_clock::now());
  _names[w].emplace_back(name);
}

// Procedure: on_exit
inline void ExecutorObserver::on_exit(unsigned w) {
  _ends[w].emplace_back(std::chrono::steady_clock::now());
}

// Function: clear
inline void ExecutorObserver::clear() {
  for(size_t w=0; w<_begs.size(); ++w) {
    _begs[w].clear();
    _ends[w].clear();
    _names[w].clear();
  }
}

// Procedure: dump
inline void ExecutorObserver::dump(std::ostream& os) const {

  os << '[';

  bool needComma = false;
  for(size_t w=0; w<_begs.size(); w++) {

    assert(_begs[w].size() == _ends[w].size());

    if(needComma && _begs[w].size() > 0) {
      os << ',';
      needComma = false;
    }

    for(size_t i=0; i<_begs[w].size(); i++) {
      os << '{'
         << "\"cat\":\"ExecutorObserver\","
         << "\"name\":\"" << _names[w][i] << "\","
         << "\"ph\":\"X\","
         << "\"pid\":1,"
         << "\"tid\":" << w << ','
         << "\"ts\":" << std::chrono::duration_cast<std::chrono::microseconds>(_begs[w][i] - _origin).count() << ','
         << "\"dur\":" << std::chrono::duration_cast<std::chrono::microseconds>(_ends[w][i] - _begs[w][i]).count();
      if(i != _begs[w].size() - 1) {
        os << "},";
      }
      else {
        os << '}';
        needComma = true;
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
  return std::accumulate(_begs.begin(), _begs.end(), size_t{0},
    [](size_t sum, const auto& vec){
      return sum + vec.size();
    }
  );
}


}  // end of namespace tf -------------------------------------------


