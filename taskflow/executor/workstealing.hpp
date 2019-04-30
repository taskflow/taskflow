// 2019/04/19 - modified by Tsung-Wei Huang
//   - added delay yielding strategy
//   - added mailbox strategy to balance the wakeup calls
//   - TODO: need to add mailbox strategy to batch
//   - TODO: need a notify_n to wake up n threads
//   - TODO: can we skip the --mailbox in emplace?
//
// 2019/04/11 - modified by Tsung-Wei Huang
//   - renamed to executor
//
// 2019/03/30 - modified by Tsung-Wei Huang
//   - added consensus sleep-loop stragety
//
// 2019/03/21 - modified by Tsung-Wei Huang
//   - removed notifier
//   - implemented a new scheduling strategy
//     (a thread will sleep as long as all threads meet the constraints)
//
// 2019/02/15 - modified by Tsung-Wei Huang
//   - batch to take reference not move
//
// 2019/02/10 - modified by Tsung-Wei Huang
//   - modified WorkStealingExecutor with notifier
//   - modified the stealing loop
//   - improved the performance
//
// 2019/01/03 - modified by Tsung-Wei Huang
//   - updated the load balancing strategy
//
// 2018/12/24 - modified by Tsung-Wei Huang
//   - refined the work balancing strategy
//
// 2018/12/06 - modified by Tsung-Wei Huang
//   - refactored the code
//   - added load balancing strategy
//   - removed the storage alignment in WorkStealingQueue
//
// 2018/12/03 - created by Tsung-Wei Huang
//   - added WorkStealingQueue class

#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <atomic>
#include <memory>
#include <deque>
#include <optional>
#include <thread>
#include <algorithm>
#include <set>
#include <numeric>
#include <cassert>

#include "notifier.hpp"
#include "observer.hpp"

namespace tf {

/**
@class: WorkStealingQueue

@tparam T data type

@brief Lock-free unbounded single-producer multiple-consumer queue.

This class implements the work stealing queue described in the paper,
"Dynamic Circular Work-stealing Deque," SPAA, 2015.
Only the queue owner can perform pop and push operations,
while others can steal data from the queue.

PPoPP implementation paper
"Correct and Efficient Work-Stealing for Weak Memory Models"
https://www.di.ens.fr/~zappa/readings/ppopp13.pdf
*/
template <typename T>
class WorkStealingQueue {

  //constexpr static int64_t cacheline_size = 64;

  //using storage_type = std::aligned_storage_t<sizeof(T), cacheline_size>;

  struct Array {

    int64_t C;
    int64_t M;
    //storage_type* S;
    T* S;

    Array(int64_t c) :
      C {c},
      M {c-1},
      //S {new storage_type[C]} {
      S {new T[static_cast<size_t>(C)]} {
      //for(int64_t i=0; i<C; ++i) {
      //  ::new (std::addressof(S[i])) T();
      //}
    }

    ~Array() {
      //for(int64_t i=0; i<C; ++i) {
      //  reinterpret_cast<T*>(std::addressof(S[i]))->~T();
      //}
      delete [] S;
    }

    int64_t capacity() const noexcept {
      return C;
    }

    template <typename O>
    void push(int64_t i, O&& o) noexcept {
      //T* ptr = reinterpret_cast<T*>(std::addressof(S[i & M]));
      //*ptr = std::forward<O>(o);
      S[i & M] = std::forward<O>(o);
    }

    T pop(int64_t i) noexcept {
      //return *reinterpret_cast<T*>(std::addressof(S[i & M]));
      return S[i & M];
    }

    Array* resize(int64_t b, int64_t t) {
      Array* ptr = new Array {2*C};
      for(int64_t i=t; i!=b; ++i) {
        ptr->push(i, pop(i));
      }
      return ptr;
    }

  };

  std::atomic<int64_t> _top;
  std::atomic<int64_t> _bottom;
  std::atomic<Array*> _array;
  std::vector<Array*> _garbage;
  //char _padding[cacheline_size];

  public:

    /**
    @brief constructs the queue with a given capacity

    @param capacity the capacity of the queue (must be power of 2)
    */
    WorkStealingQueue(int64_t capacity = 4096);

    /**
    @brief destructs the queue
    */
    ~WorkStealingQueue();

    /**
    @brief queries if the queue is empty at the time of this call
    */
    bool empty() const noexcept;

    /**
    @brief queries the number of items at the time of this call
    */
    size_t size() const noexcept;

    /**
    @brief queries the capacity of the queue
    */
    int64_t capacity() const noexcept;

    /**
    @brief inserts an item to the queue

    Only the owner thread can insert an item to the queue.
    The operation can trigger the queue to resize its capacity
    if more space is required.

    @tparam O data type

    @param item the item to perfect-forward to the queue
    */
    template <typename O>
    void push(O&& item);

    /**
    @brief pops out an item from the queue

    Only the owner thread can pop out an item from the queue.
    The return can be a @std_nullopt if this operation failed (not necessary empty).
    */
    std::optional<T> pop();

    std::optional<T> pop(int64_t&);

    /**
    @brief steals an item from the queue

    Any threads can try to steal an item from the queue.
    The return can be a @std_nullopt if this operation failed (not necessary empty).
    */
    std::optional<T> steal();
};

// Constructor
template <typename T>
WorkStealingQueue<T>::WorkStealingQueue(int64_t c) {
  assert(c && (!(c & (c-1))));
  _top.store(0, std::memory_order_relaxed);
  _bottom.store(0, std::memory_order_relaxed);
  _array.store(new Array{c}, std::memory_order_relaxed);
  _garbage.reserve(32);
}

// Destructor
template <typename T>
WorkStealingQueue<T>::~WorkStealingQueue() {
  for(auto a : _garbage) {
    delete a;
  }
  delete _array.load();
}

// Function: empty
template <typename T>
bool WorkStealingQueue<T>::empty() const noexcept {
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_relaxed);
  return b <= t;
}

// Function: size
template <typename T>
size_t WorkStealingQueue<T>::size() const noexcept {
  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_relaxed);
  return b >= t ? b - t : 0;
}

// Function: push
template <typename T>
template <typename O>
void WorkStealingQueue<T>::push(O&& o) {
  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_acquire);
  Array* a = _array.load(std::memory_order_relaxed);

  // queue is full
  if(a->capacity() - 1 < (b - t)) {
    Array* tmp = a->resize(b, t);
    _garbage.push_back(a);
    std::swap(a, tmp);
    _array.store(a, std::memory_order_relaxed);
  }

  a->push(b, std::forward<O>(o));
  std::atomic_thread_fence(std::memory_order_release);
  _bottom.store(b + 1, std::memory_order_relaxed);
}

// Function: pop
template <typename T>
std::optional<T> WorkStealingQueue<T>::pop() {
  int64_t b = _bottom.load(std::memory_order_relaxed) - 1;
  Array* a = _array.load(std::memory_order_relaxed);
  _bottom.store(b, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t t = _top.load(std::memory_order_relaxed);

  std::optional<T> item;

  if(t <= b) {
    item = a->pop(b);
    if(t == b) {
      // the last item just got stolen
      if(!_top.compare_exchange_strong(t, t+1,
                                       std::memory_order_seq_cst,
                                       std::memory_order_relaxed)) {
        item = std::nullopt;
      }
      _bottom.store(b + 1, std::memory_order_relaxed);
    }
  }
  else {
    _bottom.store(b + 1, std::memory_order_relaxed);
  }

  return item;
}

// Function: pop
template <typename T>
std::optional<T> WorkStealingQueue<T>::pop(int64_t& N) {
  int64_t b = _bottom.load(std::memory_order_relaxed) - 1;
  Array* a = _array.load(std::memory_order_relaxed);
  _bottom.store(b, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t t = _top.load(std::memory_order_relaxed);

  std::optional<T> item;

  if(t <= b) {
    item = a->pop(b);
    if(t == b) {
      // the last item just got stolen
      if(!_top.compare_exchange_strong(t, t+1,
                                       std::memory_order_seq_cst,
                                       std::memory_order_relaxed)) {
        item = std::nullopt;
      }
      _bottom.store(b + 1, std::memory_order_relaxed);
      N = 0;
    }
    else {
      N = b - t;
    }
  }
  else {
    _bottom.store(b + 1, std::memory_order_relaxed);
    N = b + 1 - t;
  }

  return item;
}

// Function: steal
template <typename T>
std::optional<T> WorkStealingQueue<T>::steal() {
  int64_t t = _top.load(std::memory_order_acquire);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = _bottom.load(std::memory_order_acquire);

  std::optional<T> item;

  if(t < b) {
    Array* a = _array.load(std::memory_order_consume);
    item = a->pop(t);
    if(!_top.compare_exchange_strong(t, t+1,
                                     std::memory_order_seq_cst,
                                     std::memory_order_relaxed)) {
      return std::nullopt;
    }
  }

  return item;
}

// Function: capacity
template <typename T>
int64_t WorkStealingQueue<T>::capacity() const noexcept {
  return _array.load(std::memory_order_relaxed)->capacity();
}

// ----------------------------------------------------------------------------

/**
@class: WorkStealingExecutor

@brief Executor that implements an efficient work stealing algorithm.

@tparam Closure closure type
*/
template <typename Closure>
class WorkStealingExecutor {

  struct Worker {
    std::minstd_rand rdgen { std::random_device{}() };
    std::atomic<int64_t> mailbox {0};
    std::optional<Closure> cache;
    WorkStealingQueue<Closure> queue;
  };

  struct PerThread {
    WorkStealingExecutor* pool {nullptr};
    int worker_id {-1};
  };

  struct Consensus {
    const unsigned threshold;
    std::atomic<unsigned> count;
    Consensus(unsigned N) : threshold {N/2}, count {N} {}
    void consent() { ++count; }
    void dissent() { --count; }
    operator bool () const { return count > threshold; }
  };

  public:

    /**
    @brief constructs the executor with a given number of worker threads

    @param N the number of worker threads
    */
    explicit WorkStealingExecutor(unsigned N);

    /**
    @brief destructs the executor

    Destructing the executor will immediately force all worker threads to stop.
    The executor does not guarantee all tasks to finish upon destruction.
    */
    ~WorkStealingExecutor();

    /**
    @brief queries the number of worker threads
    */
    size_t num_workers() const;

    /**
    @brief queries if the caller is the owner of the executor
    */
    bool is_owner() const;

    /**
    @brief constructs the closure in place in the executor

    @tparam ArgsT... argument parameter pack

    @param args arguments to forward to the constructor of the closure
    */
    template <typename... ArgsT>
    void emplace(ArgsT&&... args);

    /**
    @brief moves a batch of closures to the executor

    @param closures a vector of closures
    */
    void batch(std::vector<Closure>& closures);

    /**
    @brief constructs an observer to inspect the activities of worker threads

    Each executor manages at most one observer at a time through std::unique_ptr.
    Createing multiple observers will only keep the lastest one.

    @tparam Observer observer type derived from tf::ExecutorObserverInterface
    @tparam ArgsT... argument parameter pack

    @param args arguments to forward to the constructor of the observer

    @return a raw pointer to the observer associated with this executor
    */
    template<typename Observer, typename... Args>
    Observer* make_observer(Args&&... args);

  private:

    const std::thread::id _owner {std::this_thread::get_id()};

    std::mutex _mutex;

    std::vector<Worker> _workers;
    std::vector<Notifier::Waiter> _waiters;
    std::vector<std::thread> _threads;

    WorkStealingQueue<Closure> _queue;

    std::atomic<size_t> _num_thieves {0};
    std::atomic<size_t> _num_idlers {0};
    std::atomic<bool> _done {false};

    Notifier _notifier;

    std::unique_ptr<ExecutorObserverInterface> _observer;

    void _spawn(unsigned);
    void _balance_load(unsigned);
    void _exploit_task(unsigned, std::optional<Closure>&);
    void _explore_task(unsigned, std::optional<Closure>&);

    unsigned _randomize(uint64_t&) const;
    unsigned _fast_modulo(unsigned, unsigned) const;
    unsigned _find_victim(unsigned);

    PerThread& _per_thread() const;

    bool _wait_for_tasks(unsigned, std::optional<Closure>&);
};

// Constructor
template <typename Closure>
WorkStealingExecutor<Closure>::WorkStealingExecutor(unsigned N) :
  _workers   {N},
  _waiters   {N},
  _notifier  {_waiters} {
  _spawn(N);
}

// Destructor
template <typename Closure>
WorkStealingExecutor<Closure>::~WorkStealingExecutor() {

  _done = true;
  _notifier.notify(true);

  for(auto& t : _threads){
    t.join();
  }
}

// Function: _per_thread
template <typename Closure>
typename WorkStealingExecutor<Closure>::PerThread&
WorkStealingExecutor<Closure>::_per_thread() const {
  thread_local PerThread pt;
  return pt;
}

// Function: _randomize
template <typename Closure>
unsigned WorkStealingExecutor<Closure>::_randomize(uint64_t& state) const {
  uint64_t current = state;
  state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
  // Generate the random output (using the PCG-XSH-RS scheme)
  return static_cast<unsigned>((current ^ (current >> 22)) >> (22 + (current >> 61)));
}

// http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
template <typename Closure>
unsigned WorkStealingExecutor<Closure>::_fast_modulo(unsigned x, unsigned N) const {
  return ((uint64_t) x * (uint64_t) N) >> 32;
}

// Procedure: _spawn
template <typename Closure>
void WorkStealingExecutor<Closure>::_spawn(unsigned N) {

  // Lock to synchronize all workers before creating _worker_mapss
  for(unsigned i=0; i<N; ++i) {
    _threads.emplace_back([this, i] () -> void {

      PerThread& pt = _per_thread();
      pt.pool = this;
      pt.worker_id = i;

      std::optional<Closure> t;

      // must use 1 as condition instead of !done
      while(1) {

        // execute the tasks.
        run_task:
        _exploit_task(i, t);

        // steal loop
        if(_explore_task(i, t); t) {
          goto run_task;
        }

        // wait for tasks
        if(_wait_for_tasks(i, t) == false) {
          break;
        }
      }

    });
  }
}

// Function: is_owner
template <typename Closure>
bool WorkStealingExecutor<Closure>::is_owner() const {
  return std::this_thread::get_id() == _owner;
}

// Function: num_workers
template <typename Closure>
size_t WorkStealingExecutor<Closure>::num_workers() const {
  return _workers.size();
}

// Function: _find_victim
template <typename Closure>
unsigned WorkStealingExecutor<Closure>::_find_victim(unsigned thief) {

  unsigned l = 0;
  unsigned r = _workers.size() - 1;
  unsigned vtm = std::uniform_int_distribution<unsigned>{l, r}(
    _workers[thief].rdgen
  );

  // try to look for a task from other workers
  for(unsigned i=0; i<_workers.size(); ++i){

    if((thief == vtm && !_queue.empty()) ||
       (thief != vtm && !_workers[vtm].queue.empty())) {
      return vtm;
    }

    if(++vtm; vtm == _workers.size()) {
      vtm = 0;
    }
  }

  return _workers.size();
}

// Function: _explore_task
template <typename Closure>
void WorkStealingExecutor<Closure>::_explore_task(
  unsigned thief,
  std::optional<Closure>& t
) {

  //assert(_workers[thief].queue.empty());
  assert(!t);

  //const unsigned l = 0;
  //const unsigned r = _workers.size() - 1;

  size_t f = 0;
  size_t F = (num_workers() + 1) << 1;
  size_t y = 0;

  _workers[thief].mailbox = 0;
  ++_num_thieves;

  // explore
  while(!_done) {

    /*unsigned vtm = std::uniform_int_distribution<unsigned>{l, r}(
      _workers[thief].rdgen
    );

    t = (vtm == thief) ? _queue.steal() : _workers[vtm].queue.steal();

    if(t) {
      _workers[thief].mailbox.fetch_add(1);
      if(auto N = --_num_thieves; N == 0) {
        _notifier.notify(false);
      }
      return;
    }
    // wasteful thief
    else if(vtm != thief) {
      auto C = _workers[vtm].mailbox.load(std::memory_order_acquire);
      if(C > 0 && _workers[vtm].mailbox.compare_exchange_strong(C, C - 1,
                                                            std::memory_order_seq_cst,
                                                            std::memory_order_relaxed)) {
        _notifier.notify(false);
      }
    }
    //else {
    //  assert(_workers[vtm].mailbox == 0);
    //}

    if(f++ > F) {
      if(std::this_thread::yield(); y++ > 100) {
        break;
      }
    } */

    if(auto vtm = _find_victim(thief); vtm != _workers.size()) {
      t = (vtm == thief) ? _queue.steal() : _workers[vtm].queue.steal();
      //printf("%lu\n", _workers[vtm].mailbox.load(std::memory_order_relaxed));
      // successful thief
      if(t) {
        ++_workers[thief].mailbox;
        if(auto N = --_num_thieves; N == 0) {
          _notifier.notify(false);
        }
        return;
      }
      // wasteful thief
      else if(vtm != thief) {
        auto C = _workers[vtm].mailbox.load(std::memory_order_acquire);
        if(C > 0 && _workers[vtm].mailbox.compare_exchange_strong(C, C - 1,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
          _notifier.notify(false);
        }
      }
      //else {
      //  assert(_workers[vtm].mailbox == 0);
      //}
    }
    else {
      if(f++ > F) {
        if(std::this_thread::yield(); y++ > 100) {
          break;
        }
      }
    }
  }

  --_num_thieves;
}

// Procedure: _exploit_task
template <typename Closure>
void WorkStealingExecutor<Closure>::_exploit_task(
  unsigned i,
  std::optional<Closure>& t
) {

  if(t) {
    auto& worker = _workers[i];

    do {
      if(_observer && t->node) {

        _observer->on_entry(i, t->node->name());
      }

      (*t)();

      if(_observer) {
        _observer->on_exit(i);
      }

      if(worker.cache) {
        t = std::move(worker.cache);
        worker.cache = std::nullopt;
      }
      else {
        t = worker.queue.pop();
      }
    } while(t);
  }
}

template <>
inline void WorkStealingExecutor<std::function<void()>>::_exploit_task(
  unsigned i,
  std::optional<std::function<void()>>& t
) {

  if(t) {
    auto& worker = _workers[i];

    do {
      if(_observer) {

        _observer->on_entry(i, std::to_string(i));
      }

      (*t)();

      if(_observer) {
        _observer->on_exit(i);
      }

      if(worker.cache) {
        t = std::move(worker.cache);
        worker.cache = std::nullopt;
      }
      else {
        t = worker.queue.pop();
      }
    } while(t);
  }
}

// Function: _wait_for_tasks
template <typename Closure>
bool WorkStealingExecutor<Closure>::_wait_for_tasks(
  unsigned me,
  std::optional<Closure>& t
) {

  assert(!t);

  _notifier.prepare_wait(&_waiters[me]);

  // check again.
  if(auto vtm = _find_victim(me); vtm != _workers.size()) {
    _notifier.cancel_wait(&_waiters[me]);
    t = (vtm == me) ? _queue.steal() : _workers[vtm].queue.steal();
    return true;
  }

  if(size_t I = ++_num_idlers; _done && I == _workers.size()) {
    _notifier.cancel_wait(&_waiters[me]);
    if(_find_victim(me) != _workers.size()) {
      --_num_idlers;
      return true;
    }
    _notifier.notify(true);
    return false;
  }

  // After we update the idler count, we need to check the mailbox
  // again to ensure there is no new wakeup requests.
  //if(_num_idlers >= _workers.size() / 2) {
  for(unsigned w = 0; w<_workers.size(); ++w) {
    if(w == me) {
      //assert(_workers[w].mailbox.load(std::memory_order_relaxed) == 0);
      //_workers[w].mailbox.store(0, std::memory_order_relaxed);
      continue;
    }
    else if(_workers[w].mailbox.load(std::memory_order_relaxed) > 0) {
      _notifier.cancel_wait(&_waiters[me]);
      --_num_idlers;
      return true;
    }
  }
  //}

  // Now I really need to relinguish my self to others
  _notifier.commit_wait(&_waiters[me]);
  --_num_idlers;

  return true;
}

// Procedure: emplace
template <typename Closure>
template <typename... ArgsT>
void WorkStealingExecutor<Closure>::emplace(ArgsT&&... args){

  //no worker thread available
  if(num_workers() == 0){
    Closure{std::forward<ArgsT>(args)...}();
    return;
  }

  auto& pt = _per_thread();

  // caller is a worker to this pool
  if(pt.pool == this) {
    if(!_workers[pt.worker_id].cache) {
      _workers[pt.worker_id].cache.emplace(std::forward<ArgsT>(args)...);
      return;
    }
    else {
      _workers[pt.worker_id].queue.push(Closure{std::forward<ArgsT>(args)...});

      // We only do the wake-up when this thread is the only worker!
      // Notice that incrementing the mailbox should come before if-statement.
      ++_workers[pt.worker_id].mailbox;
      if(_num_idlers != num_workers() - 1) {
        return;
      }
    }
  }
  // other threads
  else {
    std::scoped_lock lock(_mutex);
    _queue.push(Closure{std::forward<ArgsT>(args)...});
  }

  _notifier.notify(false);
}

// Procedure: batch
template <typename Closure>
void WorkStealingExecutor<Closure>::batch(std::vector<Closure>& tasks) {

  if(tasks.empty()) {
    return;
  }

  //no worker thread available
  if(num_workers() == 0){
    for(auto &t: tasks){
      t();
    }
    return;
  }

  auto& pt = _per_thread();

  if(pt.pool == this) {

    size_t i = 0;

    if(!_workers[pt.worker_id].cache) {
      _workers[pt.worker_id].cache = std::move(tasks[i++]);
    }

    // TODO: need to implement the mailbox strategy as well
    for(; i<tasks.size(); ++i) {
      _workers[pt.worker_id].queue.push(std::move(tasks[i]));
      _notifier.notify(false);
    }

    return;
  }

  {
    std::scoped_lock lock(_mutex);

    for(size_t k=0; k<tasks.size(); ++k) {
      _queue.push(std::move(tasks[k]));
    }
  }

  size_t N = std::max(size_t{1}, std::min(_num_idlers.load(), tasks.size()));

  if(N >= _workers.size()) {
    _notifier.notify(true);
  }
  else {
    for(size_t k=0; k<N; ++k) {
      _notifier.notify(false);
    }
  }
}

// Function: make_observer
template <typename Closure>
template<typename Observer, typename... Args>
Observer* WorkStealingExecutor<Closure>::make_observer(Args&&... args) {
  _observer = std::make_unique<Observer>(std::forward<Args>(args)...);
  _observer->set_up(_threads.size());
  return static_cast<Observer*>(_observer.get());
}

}  // end of namespace tf. ---------------------------------------------------





