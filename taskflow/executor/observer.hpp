#pragma once

namespace tf {

struct ExecutorObserver {
  
  struct ExecutorInfo {
    std::thread::id owner;
    std::vector<std::thread::id> workers;
  };

  virtual ~ExecutorObserver() = default;
  
  virtual void set_up(const ExecutorInfo&) {}
  virtual void on_entry(void* data) {}
  virtual void on_exit(void* data) {}
  virtual void tear_down() {}
};

}
