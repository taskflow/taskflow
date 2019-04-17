#pragma once

#include "../executor/observer.hpp"

namespace tf {

// TODO: use typename T (taskflow)
template <typename T>
class BasicTaskflowObserver: public tf::ExecutorObserver {
  public:

    void set_up(const ExecutorObserver::ExecutorInfo&);
    void on_entry(void* data);
    void on_exit(void* data);

    // TODO: dump
    void dump(std::ostream&) const;
    std::string dump() const;

  private:

    // TODO: steady clock
    struct Duration {
      Duration(Node* n) {
        if(n == nullptr) {
          name = "Sleep";
          return ;
        }
        if(!n->name().empty()) {
          name = n->name();
        }
        else {
          std::ostringstream oss;
          oss << (void*)this;
          name = oss.str();
        }
      }
      std::chrono::time_point<std::chrono::steady_clock> beg {std::chrono::steady_clock::now()};
      std::chrono::time_point<std::chrono::steady_clock> end {std::chrono::steady_clock::now()};
      std::string name;
    };  

    std::unordered_map<std::thread::id, std::vector<Duration>> _durations;
    std::chrono::time_point<std::chrono::steady_clock> _beg {std::chrono::steady_clock::now()};
};  

template <typename T>
void BasicTaskflowObserver<T>::set_up(const ExecutorObserver::ExecutorInfo& info) {
  _durations.reserve(info.workers.size());
  for(auto &w: info.workers) {
    _durations.emplace(w, std::vector<Duration>());
  }
}

template <typename T>
void BasicTaskflowObserver<T>::on_entry(void* data) {
  auto closure = static_cast<typename T::Closure*>(data);
  _durations.at(std::this_thread::get_id()).emplace_back(closure->node);
}

template <typename T>
void BasicTaskflowObserver<T>::on_exit(void* data) {
  _durations.at(std::this_thread::get_id()).back().end = std::chrono::steady_clock::now();
}

template <typename T>
void BasicTaskflowObserver<T>::dump(std::ostream& os) const {
  size_t i {0};
  os << '[';
  for(auto& [tid, ds]: _durations) {
    if(ds.empty()) continue;
    if(i != 0 && !ds.empty()) {
      os << ',';
    }
    size_t j {0};
    for(auto &d : ds) {
      if(d.beg == d.end) {
        continue;
      }
      os << '{';

      // Complete event format 
      os << "\"name\":\"" << d.name << "\", ";
      os << "\"cat\":\"Cpp-Taskflow\"," ;
      os << "\"ph\":\"X\",";
      os << "\"pid\":1,";
      os << "\"tid\":" << i << ',';
      os << "\"ts\":"  << std::chrono::duration_cast<std::chrono::microseconds>(d.beg - _beg).count() << ',';
      os << "\"dur\":"  << std::chrono::duration_cast<std::chrono::microseconds>(d.end - d.beg).count();

      if(j != ds.size() - 1)
        os << "},";
      else
        os << '}';
      j ++;
    }
    i ++;
  }
  os << ']';
}

// TODO: hard-code the json output
template <typename T>
std::string BasicTaskflowObserver<T>::dump() const {
  std::ostringstream oss;
  dump(oss);
  return oss.str();
}

}  // end of namespace tf ----------------------------------------------------

