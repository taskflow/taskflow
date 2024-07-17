#pragma once


// use tf::Latch
#include <condition_variable>
#include <limits>
#include <mutex>

namespace tf {

class Latch {

private:

  std::ptrdiff_t _counter;
  mutable std::condition_variable _cv;
  mutable std::mutex _mutex;

public:

  static constexpr ptrdiff_t (max)() noexcept
  {
    return (std::numeric_limits<ptrdiff_t>::max)();
  }

  explicit Latch(std::ptrdiff_t expected)
    : _counter(expected)
  {
    assert(0 <= expected && expected < (max)());
  }

  ~Latch() = default;

  Latch(const Latch&) = delete;
  Latch& operator=(const Latch&) = delete;

  void count_down(std::ptrdiff_t update = 1)
  {
    std::lock_guard<decltype(_mutex)> lk(_mutex);
    assert(0 <= update && update <= _counter);
    _counter -= update;
    if (_counter == 0) {
      _cv.notify_all();
    }
  }

  bool try_wait() const noexcept
  {
    std::lock_guard<decltype(_mutex)> lk(_mutex);
    // no spurious failure
    return (_counter == 0);
  }

  void wait() const
  {
    std::unique_lock<decltype(_mutex)> lk(_mutex);
    while (_counter != 0) {
      _cv.wait(lk);
    }
  }

  void arrive_and_wait(std::ptrdiff_t update = 1)
  {
    std::unique_lock<decltype(_mutex)> lk(_mutex);
    // equivalent to { count_down(update); wait(); }
    assert(0 <= update && update <= _counter);
    _counter -= update;
    if (_counter == 0) {
      _cv.notify_all();
    }
    while (_counter != 0) {
      _cv.wait(lk);
    }
  }
};

} // namespace tf -------------------------------------------------------------
