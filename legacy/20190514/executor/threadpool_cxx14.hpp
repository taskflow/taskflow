// A C++-14 based threadpool implementation inspired by Taskflow Threadpool.
// 
// 2018/09/02 - contributed by Glen Fraser
//   - added wait_for_all method
//
// 2018/08/27 - contributed by Glen Fraser
// taskflow.hpp was modified by Glen Fraser to produce this file
// (threadpool_cxx14.hpp), which is a "light" version of the library with
// restricted functionality -- it only exposes the tf::Threadpool class.
// However, it has also been reworked to support compilation with C++14
// (instead of requiring C++17, as the main Taskflow library does).
// It is designed to be used in cases where only the Threadpool
// functionality is required, in projects that are reliant on a (slightly)
// older version of C++.
//
// NOTE: if you are using a fully C++17-compliant compiler, you should
//       be including "taskflow.hpp" rather than this file!

#pragma once

#include <deque>
#include <vector>
#include <thread>
#include <future>
#include <unordered_set>
#include <type_traits>
#include <utility>

//-------------------------------------------------------------------------------------------------
// C++14 implementation of C++17's std::invoke_result, taken from:
//   https://en.cppreference.com/w/cpp/types/result_of
//-------------------------------------------------------------------------------------------------

namespace std {

  namespace detail {
    template <class T>
    struct is_reference_wrapper : std::false_type {};
    template <class U>
    struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type {};

    template<class T>
    struct invoke_impl {
      template<class F, class... Args>
      static auto call(F&& f, Args&&... args)
        -> decltype(std::forward<F>(f)(std::forward<Args>(args)...));
    };

    template<class B, class MT>
    struct invoke_impl<MT B::*> {
      template<class T, class Td = typename std::decay<T>::type,
        class = typename std::enable_if<std::is_base_of<B, Td>::value>::type
      >
      static auto get(T&& t)->T&&;

      template<class T, class Td = typename std::decay<T>::type,
        class = typename std::enable_if<is_reference_wrapper<Td>::value>::type
      >
      static auto get(T&& t) -> decltype(t.get());

      template<class T, class Td = typename std::decay<T>::type,
        class = typename std::enable_if<!std::is_base_of<B, Td>::value>::type,
        class = typename std::enable_if<!is_reference_wrapper<Td>::value>::type
      >
      static auto get(T&& t) -> decltype(*std::forward<T>(t));

      template<class T, class... Args, class MT1,
        class = typename std::enable_if<std::is_function<MT1>::value>::type
      >
      static auto call(MT1 B::*pmf, T&& t, Args&&... args)
        -> decltype((invoke_impl::get(std::forward<T>(t)).*pmf)(std::forward<Args>(args)...));

      template<class T>
      static auto call(MT B::*pmd, T&& t)
        -> decltype(invoke_impl::get(std::forward<T>(t)).*pmd);
    };

    template<class F, class... Args, class Fd = typename std::decay<F>::type>
    auto INVOKE(F&& f, Args&&... args)
      -> decltype(invoke_impl<Fd>::call(std::forward<F>(f), std::forward<Args>(args)...));

  } // namespace detail

  // Conforming C++14 implementation (is also a valid C++11 implementation):
  namespace detail {
    template <typename AlwaysVoid, typename, typename...>
    struct invoke_result { };
    template <typename F, typename...Args>
    struct invoke_result<decltype(void(detail::INVOKE(std::declval<F>(), std::declval<Args>()...))),
      F, Args...> {
      using type = decltype(detail::INVOKE(std::declval<F>(), std::declval<Args>()...));
    };
  } // namespace detail

  template <class F, class... ArgTypes>
  struct invoke_result : detail::invoke_result<void, F, ArgTypes...> {};

  template< class F, class... ArgTypes>
  using invoke_result_t = typename invoke_result<F, ArgTypes...>::type;
}

// ------------------------------------------------------------------------------------------------

namespace tf {

//-------------------------------------------------------------------------------------------------
// Utility
//-------------------------------------------------------------------------------------------------

// Struct: MoC
template <typename T>
struct MoC {

  MoC(T&& rhs) : object(std::move(rhs)) {}
  MoC(const MoC& other) : object(std::move(other.object)) {}

  T& get() { return object; }

  mutable T object;
};

//-------------------------------------------------------------------------------------------------
// Threadpool definition
//-------------------------------------------------------------------------------------------------

// Class: Threadpool
class Threadpool {

  enum class Signal {
    STANDARD,
    SHUTDOWN
  };

  public:

    inline Threadpool(unsigned);
    inline ~Threadpool();

    template<typename C>
    std::enable_if_t<
      std::is_same<void, std::invoke_result_t<C>>::value,
      std::future<std::invoke_result_t<C>>
    >
    async(C&&, Signal = Signal::STANDARD);

    template<typename C>
    std::enable_if_t<
      !std::is_same<void, std::invoke_result_t<C>>::value,
      std::future<std::invoke_result_t<C>>
    >
    async(C&&, Signal = Signal::STANDARD);

    template <typename C>
    auto silent_async(C&&, Signal = Signal::STANDARD);

    inline void shutdown();
    inline void spawn(unsigned);

    inline void wait_for_all();
    inline size_t num_tasks() const;
    inline size_t num_workers() const;

    inline bool is_worker() const;

  private:

    mutable std::mutex _mutex;

    std::condition_variable _worker_signal;
    std::deque<std::function<Signal()>> _task_queue;
    std::vector<std::thread> _threads;
    std::unordered_set<std::thread::id> _worker_ids;
};

// Constructor
inline Threadpool::Threadpool(unsigned N) {
  spawn(N);
}

// Destructor
inline Threadpool::~Threadpool() {
  shutdown();
}

// Function: num_tasks
// Return the number of "unfinished" tasks. Notice that this value is not necessary equal to
// the size of the task_queue since the task can be popped out from the task queue while
// not yet finished.
inline size_t Threadpool::num_tasks() const {
  return _task_queue.size();
}

inline size_t Threadpool::num_workers() const {
  return _threads.size();
}

inline bool Threadpool::is_worker() const {
  std::lock_guard<std::mutex> lock(_mutex);
  return _worker_ids.find(std::this_thread::get_id()) != _worker_ids.end();
}

// Procedure: spawn
// The procedure spawns "n" threads monitoring the task queue and executing each task. After the
// task is finished, the thread reacts to the returned signal.
inline void Threadpool::spawn(unsigned N) {

  if(is_worker()) {
    throw std::runtime_error("Worker thread cannot spawn threads");
  }

  for(size_t i=0; i<N; ++i) {

    _threads.emplace_back([this] () -> void {

      {  // Acquire lock
        std::lock_guard<std::mutex> lock(_mutex);
        _worker_ids.insert(std::this_thread::get_id());
      }

      bool stop {false};

      while(!stop) {
        decltype(_task_queue)::value_type task;

        { // Acquire lock. --------------------------------
          std::unique_lock<std::mutex> lock(_mutex);
          _worker_signal.wait(lock, [this] () { return _task_queue.size() != 0; });
          task = std::move(_task_queue.front());
          _task_queue.pop_front();
        } // Release lock. --------------------------------

        // Execute the task and react to the returned signal.
        switch(task()) {
          case Signal::SHUTDOWN:
            stop = true;
          break;

          default:
          break;
        };

      } // End of worker loop.

      {  // Acquire lock
        std::lock_guard<std::mutex> lock(_mutex);
        _worker_ids.erase(std::this_thread::get_id());
      }

    });
  }
}

// Function: silent_async
// Insert a task without giving future.
template <typename C>
auto Threadpool::silent_async(C&& c, Signal sig) {

  // No worker, do this right away.
  if(num_workers() == 0) {
    c();
  }
  // Dispatch this to a thread.
  else {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _task_queue.emplace_back(
        [c=std::forward<C>(c), ret=sig] () mutable {
          c();
          return ret;
        }
      );
    }
    _worker_signal.notify_one();
  }
}

// Function: async
// Insert a callable task and return a future representing the task.
// Version for tasks returning void.
template<typename C>
std::enable_if_t<
  std::is_same<void, std::invoke_result_t<C>>::value,
  std::future<std::invoke_result_t<C>>
> Threadpool::async(C&& c, Signal sig) {

  using R = std::invoke_result_t<C>;

  std::promise<R> p;
  auto fu = p.get_future();

  // No worker, do this immediately.
  if(_threads.empty()) {
    c();
    p.set_value();
  }
  // Schedule a thread to do this.
  else {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _task_queue.emplace_back(
        [p = MoC<decltype(p)>(std::move(p)), c = std::forward<C>(c), ret = sig]() mutable {
          c();
          p.get().set_value();
          return ret;
        }
      );
    }
    _worker_signal.notify_one();
  }
  return fu;
}

// Function: async
// Version for tasks returning anything other than void.
template<typename C>
std::enable_if_t<
  !std::is_same<void, std::invoke_result_t<C>>::value,
  std::future<std::invoke_result_t<C>>
> Threadpool::async(C&& c, Signal sig) {

  using R = std::invoke_result_t<C>;

  std::promise<R> p;
  auto fu = p.get_future();

  // No worker, do this immediately.
  if (_threads.empty()) {
    p.set_value(c());
  }
  // Schedule a thread to do this.
  else {
    {
      std::lock_guard<std::mutex> lock(_mutex);

      _task_queue.emplace_back(
        [p=MoC<decltype(p)>(std::move(p)), c=std::forward<C>(c), ret=sig]() mutable {
          p.get().set_value(c());
          return ret;
        }
      );
    }
    _worker_signal.notify_one();
  }
  return fu;
}

// Procedure: wait_for_all
// After this method returns, all previously-scheduled tasks in the pool
// will have been executed.
inline void Threadpool::wait_for_all() {

  if(is_worker()) {
    throw std::runtime_error("Worker thread cannot wait for all");
  }

  std::mutex barrier_mutex;
  std::condition_variable barrier_cv;
  auto threads_to_sync{_threads.size()};
  std::vector<std::future<void>> futures;

  auto barrier_task = [&] {
    std::unique_lock<std::mutex> lock(barrier_mutex);
    if (--threads_to_sync == 0) {
      barrier_cv.notify_all();
    }
    else {
      barrier_cv.wait(lock, [&threads_to_sync] {
        return threads_to_sync == 0;
      });
    }
  };

  for(size_t i=0; i<_threads.size(); ++i) {
    futures.emplace_back(async(barrier_task));
  }

  // Wait for all threads to have finished synchronization
  for (auto& fu : futures) {
    fu.get();
  }
}

// Procedure: shutdown
// Remove a given number of workers. Notice that only the master can call this procedure.
inline void Threadpool::shutdown() {

  if(is_worker()) {
    throw std::runtime_error("Worker thread cannot shut down the thread pool");
  }

  for(size_t i=0; i<_threads.size(); ++i) {
    silent_async([](){}, Signal::SHUTDOWN);
  }

  for(auto& t : _threads) {
    t.join();
  }

  _threads.clear();
}

};  // end of namespace tf. ---------------------------------------------------
