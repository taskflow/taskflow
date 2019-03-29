// 2019/03/29 - created by Tsung-Wei Huang
//  - modified from Eigen's unsupported threadpool

#pragma once

#include "notifier.hpp"

namespace tf {

template <typename T, unsigned N = 4096>
class BoundedWorkStealingQueue {

  static_assert((N & (N-1)) == 0, "size must be power of two");
  static_assert(N > 2, "size must be larger than two");
  static_assert(N <= (64 << 10), "size must be smaller than 65536");

  public:

  BoundedWorkStealingQueue() : _front(0), _back(0) {
    for (unsigned i = 0; i < N; i++) {
      _array[i].state.store(kEmpty, std::memory_order_relaxed);
    }
  }

  ~BoundedWorkStealingQueue() { assert(size() == 0); }

  // push inserts w at the beginning of the queue.
  // If queue is full returns w, otherwise returns default-constructed Item.

  bool push(T& w) {

    unsigned front = _front.load(std::memory_order_relaxed);
    Entry* e = &_array[front & MASK];
    uint8_t s = e->state.load(std::memory_order_relaxed);

    // queue is full
    if (s != kEmpty ||
        !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire)) {
      return false;
    }

    _front.store(front + 1 + (N << 1), std::memory_order_relaxed);
    e->w = std::move(w);
    e->state.store(kReady, std::memory_order_release);

    return true;
  }

  // pop removes and returns the first element in the queue.
  // If the queue was empty returns default-constructed T.
  std::optional<T> pop() {
    unsigned front = _front.load(std::memory_order_relaxed);
    Entry* e = &_array[(front - 1) & MASK];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    
    if (s != kReady ||
        !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire)) {
      return std::nullopt;
    }

    T w = std::move(e->w);
    e->state.store(kEmpty, std::memory_order_release);
    front = ((front - 1) & MASK2) | (front & ~MASK2);
    _front.store(front, std::memory_order_relaxed);

    return w;
  }

  // assign adds w at the end of the queue.
  // If queue is full returns w, otherwise returns default-constructed T.

  bool assign(T& w) {
    std::unique_lock<std::mutex> lock(_mutex);
    unsigned back = _back.load(std::memory_order_relaxed);
    Entry* e = &_array[(back - 1) & MASK];
    uint8_t s = e->state.load(std::memory_order_relaxed);

    // queue is full
    if (s != kEmpty ||
        !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire)) {
      return false;
    }

    back = ((back - 1) & MASK2) | (back & ~MASK2);
    _back.store(back, std::memory_order_relaxed);
    e->w = std::move(w);
    e->state.store(kReady, std::memory_order_release);

    return true;
  }

  // steal removes and returns the last elements in the queue.
  // Can fail spuriously.
  std::optional<T> steal() {

    if (empty()) {
      return std::nullopt;
    }

    std::unique_lock<std::mutex> lock(_mutex, std::try_to_lock);

    if (!lock) {
      return std::nullopt;
    }

    unsigned back = _back.load(std::memory_order_relaxed);
    Entry* e = &_array[back & MASK];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    if (s != kReady ||
        !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire)) {
      return std::nullopt;
    }

    T w = std::move(e->w);
    e->state.store(kEmpty, std::memory_order_release);
    _back.store(back + 1 + (N << 1), std::memory_order_relaxed);

    return w;
  }

  // steal_half removes and returns half last elements in the queue.
  // Returns number of elements removed. But can also fail spuriously.
  unsigned steal_half(std::vector<T>& result) {
    if (empty()) return 0;
    std::unique_lock<std::mutex> lock(_mutex, std::try_to_lock);
    if (!lock) return 0;
    unsigned back = _back.load(std::memory_order_relaxed);
    unsigned s = size();
    unsigned mid = back;
    if (s > 1) mid = back + (s - 1) / 2;
    unsigned n = 0;
    unsigned start = 0;
    for (; static_cast<int>(mid - back) >= 0; mid--) {
      Entry* e = &_array[mid & MASK];
      uint8_t s = e->state.load(std::memory_order_relaxed);
      if (n == 0) {
        if (s != kReady ||
            !e->state.compare_exchange_strong(s, kBusy,
                                              std::memory_order_acquire))
          continue;
        start = mid;
      } else {
        // Note: no need to store temporal kBusy, we exclusively own these
        // elements.
        assert(s == kReady);
      }
      result.push_back(std::move(e->w));
      e->state.store(kEmpty, std::memory_order_release);
      n++;
    }
    if (n != 0)
      _back.store(start + 1 + (N << 1), std::memory_order_relaxed);
    return n;
  }

  // Size returns current queue size.
  // Can be called by any thread at any time.
  unsigned size() const {
    // Emptiness plays critical role in thread pool blocking. So we go to great
    // effort to not produce false positives (claim non-empty queue as empty).
    for (;;) {
      // Capture a consistent snapshot of front/tail.
      unsigned front = _front.load(std::memory_order_acquire);
      unsigned back = _back.load(std::memory_order_acquire);
      unsigned front1 = _front.load(std::memory_order_relaxed);
      if (front != front1) continue;
      int s = (front & MASK2) - (back & MASK2);
      // Fix overflow.
      if (s < 0) s += 2 * N;
      // Order of modification in push/pop is crafted to make the queue look
      // larger than it is during concurrent modifications. E.g. pop can
      // decrement size before the corresponding push has incremented it.
      // So the computed size can be up to N + 1, fix it.
      if (s > static_cast<int>(N)) s = N;
      return s;
    }
  }

  // empty tests whether container is empty.
  // Can be called by any thread at any time.
  bool empty() const { return size() == 0; }

 private:

  static const unsigned MASK = N - 1;
  static const unsigned MASK2 = (N << 1) - 1;

  struct Entry {
    std::atomic<uint8_t> state;
    T w;
  };

  enum {
    kEmpty,
    kBusy,
    kReady,
  };

  std::mutex _mutex;

  // Low log(N) + 1 bits in _front and _back contain rolling index of
  // front/back, repsectively. The remaining bits contain modification counters
  // that are incremented on Push operations. This allows us to (1) distinguish
  // between empty and full conditions (if we would use log(N) bits for
  // position, these conditions would be indistinguishable); (2) obtain
  // consistent snapshot of _front/_back for Size operation using the
  // modification counters.
  std::atomic<unsigned> _front;
  std::atomic<unsigned> _back;

  Entry _array[N];

  BoundedWorkStealingQueue(const BoundedWorkStealingQueue&) = delete;
  void operator=(const BoundedWorkStealingQueue&) = delete;
};

// ----------------------------------------------------------------------------

/** 
@class: BoundedWorkStealingThreadpool

@brief Executor that implements an efficient work stealing algorithm.

@tparam Closure closure type
*/
template <typename Closure>
class BoundedWorkStealingThreadpool {
    
  struct Worker {
    BoundedWorkStealingQueue<Closure> queue;
    std::optional<Closure> cache;
  };
    
  struct PerThread {
    BoundedWorkStealingThreadpool* pool {nullptr}; 
    int worker_id {-1};
    uint64_t seed {std::hash<std::thread::id>()(std::this_thread::get_id())};
  };

  public:
    
    /**
    @brief constructs the executor with a given number of worker threads

    @param N the number of worker threads
    */
    explicit BoundedWorkStealingThreadpool(unsigned N);

    /**
    @brief destructs the executor

    Destructing the executor will immediately force all worker threads to stop.
    The executor does not guarantee all tasks to finish upon destruction.
    */
    ~BoundedWorkStealingThreadpool();
    
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

    @param args... arguments to forward to the constructor of the closure
    */
    template <typename... ArgsT>
    void emplace(ArgsT&&... args);
    
    /**
    @brief moves a batch of closures to the executor

    @param closures a vector of closures
    */
    void batch(std::vector<Closure>& closures);

  private:
    
    const std::thread::id _owner {std::this_thread::get_id()};

    std::vector<Worker> _workers;
    std::vector<Notifier::Waiter> _waiters;
    std::vector<unsigned> _coprimes;
    std::vector<std::thread> _threads;

    std::atomic<unsigned> _num_idlers {0};
    std::atomic<bool> _done {false};
    std::atomic<bool> _spinning {false};

    Notifier _notifier;
    
    void _spawn(unsigned);

    unsigned _randomize(uint64_t&) const;
    unsigned _fast_modulo(unsigned, unsigned) const;
    unsigned _find_victim(unsigned) const;
    
    PerThread& _per_thread() const;

    std::optional<Closure> _steal();
};

// Constructor
template <typename Closure>
BoundedWorkStealingThreadpool<Closure>::BoundedWorkStealingThreadpool(unsigned N) : 
  _workers  {N},
  _waiters  {N},
  _notifier {_waiters} {

  for(unsigned i = 1; i <= N; i++) {
    unsigned a = i;
    unsigned b = N;
    // If GCD(a, b) == 1, then a and b are coprimes.
    if(std::gcd(a, b) == 1) {
      _coprimes.push_back(i);
    }
  }

  _spawn(N);
}

// Destructor
template <typename Closure>
BoundedWorkStealingThreadpool<Closure>::~BoundedWorkStealingThreadpool() {

  _done = true;
  _notifier.notify(true);

  for(auto& t : _threads){
    t.join();
  } 
  
}

// Function: _per_thread
template <typename Closure>
typename BoundedWorkStealingThreadpool<Closure>::PerThread& 
BoundedWorkStealingThreadpool<Closure>::_per_thread() const {
  thread_local PerThread pt;
  return pt;
}

// Function: _randomize
template <typename Closure>
unsigned BoundedWorkStealingThreadpool<Closure>::_randomize(uint64_t& state) const {
  uint64_t current = state;
  state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
  // Generate the random output (using the PCG-XSH-RS scheme)
  return static_cast<unsigned>((current ^ (current >> 22)) >> (22 + (current >> 61)));
}

// http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
template <typename Closure>
unsigned BoundedWorkStealingThreadpool<Closure>::_fast_modulo(unsigned x, unsigned N) const {
  return ((uint64_t) x * (uint64_t) N) >> 32;
}

// Procedure: _spawn
template <typename Closure>
void BoundedWorkStealingThreadpool<Closure>::_spawn(unsigned N) {
  
  // Lock to synchronize all workers before creating _worker_mapss
  for(unsigned i=0; i<N; ++i) {
    _threads.emplace_back([this, i, N] () -> void {

      PerThread& pt = _per_thread();  
      pt.pool = this;
      pt.worker_id = i;
    
      auto& worker = _workers[i];
      auto& waiter = _waiters[i];

      std::optional<Closure> t;

      while(1) {
        
        // execute the tasks.
        run_task:
        
        while(t) {
          (*t)();
          if(worker.cache) {
            t = std::move(worker.cache);
            worker.cache = std::nullopt;
          }
          else {
            t = worker.queue.pop();
          }
        }

        // stealing loop
        assert(!t);

        while (!t) {
          if(auto victim = _find_victim(i); victim != N) {
            if(t = _workers[victim].queue.steal(); t) {
              goto run_task;
            }
          }
          else break;
        }

        // wait for more tasks
        assert(!t);
        _notifier.prepare_wait(&waiter);
        
        if(auto victim = _find_victim(i); victim != N) {
          _notifier.cancel_wait(&waiter);
          t = _workers[victim].queue.steal();
          goto run_task;
        }

        if(auto I = ++_num_idlers; _done && I == N) {
          _notifier.cancel_wait(&waiter);
          if(_find_victim(i) != N) {
            --_num_idlers;
            goto run_task;
          }
          _notifier.notify(true);
          break;
        }
        _notifier.commit_wait(&waiter);
        --_num_idlers;
      }
      
    });     
  }
}

// Function: is_owner
template <typename Closure>
bool BoundedWorkStealingThreadpool<Closure>::is_owner() const {
  return std::this_thread::get_id() == _owner;
}

// Function: num_workers
template <typename Closure>
size_t BoundedWorkStealingThreadpool<Closure>::num_workers() const { 
  return _workers.size();  
}

// Function: _non_empty_queue
template <typename Closure>
unsigned BoundedWorkStealingThreadpool<Closure>::_find_victim(unsigned thief) const {

  //assert(_workers[thief].queue.empty());

  // try to find a victim candidate
  auto &pt = _per_thread();
  auto rnd = _randomize(pt.seed);
  auto inc = _coprimes[_fast_modulo(rnd, _coprimes.size())];
  auto vtm = _fast_modulo(rnd, _workers.size());

  for(unsigned i=0; i<_workers.size(); ++i) {
    if(!_workers[vtm].queue.empty()) {
      return vtm;
    }
    if(vtm += inc; vtm >= _workers.size()) {
      vtm -= _workers.size();
    }
  }

  return _workers.size();
}

// Function: _steal
template <typename Closure>
std::optional<Closure> BoundedWorkStealingThreadpool<Closure>::_steal() {
  
  auto &pt = _per_thread();
  auto rnd = _randomize(pt.seed);
  auto inc = _coprimes[_fast_modulo(rnd, _coprimes.size())];
  auto vtm = _fast_modulo(rnd, _workers.size());

  for(unsigned i=0; i<_workers.size(); ++i) {
    if(auto task = _workers[vtm].queue.steal(); task) {
      return task;
    }
    if(vtm += inc; vtm >= _workers.size()) {
      vtm -= _workers.size();
    }
  }
    
  return std::nullopt; 
}

// Procedure: emplace
template <typename Closure>
template <typename... ArgsT>
void BoundedWorkStealingThreadpool<Closure>::emplace(ArgsT&&... args){
  
  //no worker thread available
  if(num_workers() == 0){
    Closure{std::forward<ArgsT>(args)...}();
    return;
  }

  // caller is a worker to this pool
  if(auto& pt = _per_thread(); pt.pool == this) {
    if(!_workers[pt.worker_id].cache) {
      _workers[pt.worker_id].cache.emplace(std::forward<ArgsT>(args)...);
      return;
    }
    else {
      Closure c{std::forward<ArgsT>(args)...};
      if(_workers[pt.worker_id].queue.push(c) == true) {
        _notifier.notify(false);
      }
      // TODO: sure this?
      else {
        std::invoke(c);
      }
    }
  }
  // other threads
  else {
    //std::scoped_lock lock(_mutex);
    //_queue.push(Closure{std::forward<ArgsT>(args)...});
    Closure c{std::forward<ArgsT>(args)...};

    auto victim = _fast_modulo(_randomize(pt.seed), _workers.size()); 

    if(_workers[victim].queue.assign(c) == true) {
      _notifier.notify(false);
    }
    else {
      std::invoke(c);
    }
  }

  //_cv.notify_one();
}

// Procedure: batch
template <typename Closure>
void BoundedWorkStealingThreadpool<Closure>::batch(std::vector<Closure>& tasks) {

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
  
  // schedule the work
  if(auto& pt = _per_thread(); pt.pool == this) {
    
    size_t i = 0;

    if(!_workers[pt.worker_id].cache) {
      _workers[pt.worker_id].cache = std::move(tasks[i++]);
    }

    for(; i<tasks.size(); ++i) {
      if(_workers[pt.worker_id].queue.push(tasks[i]) == true) {
        _notifier.notify(false);
      }
      // TODO: execute the task in the last?
      else {
        std::invoke(tasks[i]);
      }
    }
  }
  // free-standing thread
  else {
    for(size_t k=0; k<tasks.size(); ++k) {
      auto victim = _fast_modulo(_randomize(pt.seed), _workers.size()); 
      if(_workers[victim].queue.assign(tasks[k]) == true) {
        _notifier.notify(false);
      }
      else {
        std::invoke(tasks[k]);
      }
    }
  }

} 

}  // end of namespace tf. ---------------------------------------------------





