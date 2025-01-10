#include <cstdint>
#include <limits>

using eidType = std::uint64_t;
using vidType = std::uint64_t;
using weight_type = std::uint64_t;

struct BaseGraph {};

// ---- start here

#include <taskflow/taskflow.hpp>

// ----------------------------------------------------------------------------
// Threadpool
// ----------------------------------------------------------------------------
  
struct Worker {
  std::vector<vidType> local_frontiers;

  std::thread thread;
  std::condition_variable cv;
  std::mutex mutex;
  std::function<bool()> task;

  void wait_for_task() {
    std::function<bool()> local;
    while(true) {
      {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&]{ return task != nullptr; });
        local = std::move(task);
      }
      if(local() == false) {
        break;
      }
    }
  }

  template <typename C>
  auto async(C&& callable) {

    std::promise<void> promise;
    auto fu = promise.get_future();

    {
      std::scoped_lock lock(mutex);
      task = [callable=std::forward<C>(callable), p=tf::MoC{std::move(promise)}]() mutable {
        auto res = callable();
        p.object.set_value(); 
        return res;
      };
    }

    cv.notify_one();

    return fu;
  }

  //Worker() : thread ([this](){ wait_for_task(); }) { }
};

struct Threadpool {
  Threadpool() : 
    workers(std::thread::hardware_concurrency())
  {
    //std::cout << "threadpool created!!! " << this << std::endl;
    ////spawn threads
    for(size_t i=0; i<workers.size(); i++) {
      workers[i].thread = std::thread([&w=workers[i]](){
        w.wait_for_task();
      });
    }
  }

  ~Threadpool() {
    // stop workers
    for(size_t i=0; i<workers.size(); i++) {
      workers[i].async([](){ return false; });
    }
    
    // join threads
    for(size_t i=0; i<workers.size(); i++) {
      workers[i].thread.join();
    }
  }

  std::vector<Worker> workers;

  Threadpool(const Threadpool&) = delete;
  Threadpool(Threadpool&&) = delete;

  Threadpool& operator = (const Threadpool&) = delete;
  Threadpool& operator = (Threadpool&&) = delete;
};

Threadpool& get_threadpool() {
  static Threadpool threadpool;
  return threadpool;
};

// ----------------------------------------------------------------------------
// Frontiers
// ----------------------------------------------------------------------------

struct Frontiers {

  Frontiers() = default;

  std::vector<vidType> this_frontiers;
  std::vector<vidType> next_frontiers;
  void resize(size_t N) {
    this_frontiers.resize(N);
    next_frontiers.resize(N);
  }
  
  Frontiers(const Frontiers&) = delete;
  Frontiers(Frontiers&&) = delete;

  Frontiers& operator = (const Frontiers&) = delete;
  Frontiers& operator = (Frontiers&&) = delete;
};

Frontiers& get_frontiers() {
  static Frontiers frontiers;
  return frontiers;
}

// ----------------------------------------------------------------------------
// Graph
// ----------------------------------------------------------------------------
class Graph : public BaseGraph {
  
  eidType* rowptr;
  vidType* col;
  uint64_t N;
  uint64_t M;
  
  //constexpr static size_t alpha {65536};
  constexpr static size_t alpha {1};
  constexpr static size_t gamma {64};  // minimum chunk size

  size_t W;

  vidType* this_frontiers;
  vidType* next_frontiers;

  public:
  

  Graph(eidType* in_rowptr, vidType* in_col, uint64_t in_N, uint64_t in_M) :
    rowptr (in_rowptr), 
    col    (in_col), 
    N      (in_N), 
    M      (in_M),
    W      ((N + M) < alpha ? 0 : get_threadpool().workers.size()) {

    auto& frontiers = get_frontiers();
    frontiers.resize(N);
    this_frontiers = frontiers.this_frontiers.data();
    next_frontiers = frontiers.next_frontiers.data();
  }
  
  ~Graph() {
    // destructor logic.
    // If you perform any memory allocations with malloc, new, etc. you must free
    //   them here to avoid memory leaks.
  }
  
  void BFS(vidType source, weight_type * distances) {

    ((N + M) < alpha) ? BFS_sequential(source, distances) : BFS_parallel(source, distances);
  }

  size_t num_edges(vidType v) const {
    return rowptr[v+1] - rowptr[v];
  }
  
  // step - static scheduling
  void step(
    size_t w, 
    size_t beg, 
    size_t end, 
    weight_type* distances, 
    std::atomic<size_t>& num_next_frontiers
  ) {
    auto& local_frontiers = get_threadpool().workers[w].local_frontiers;
    local_frontiers.clear();
    
    for(size_t i=beg; i<end; i++) {
      auto u = this_frontiers[i];
      for (eidType k = rowptr[u]; k < rowptr[u+1]; k++) {
        auto v = col[k];
        if(v != u && __sync_bool_compare_and_swap(&distances[v], std::numeric_limits<weight_type>::max(), distances[u] + 1)) {
          local_frontiers.push_back(v);
        }
      }
    }
    size_t next_beg = num_next_frontiers.fetch_add(local_frontiers.size(), std::memory_order_relaxed);
    std::memcpy(next_frontiers + next_beg, local_frontiers.data(), local_frontiers.size()*sizeof(vidType));
  }
  
  // step - dynamic scheduling
  void step(
    size_t w, 
    weight_type* distances, 
    size_t num_this_frontiers, 
    std::atomic<size_t>& next, 
    std::atomic<size_t>& num_next_frontiers
  ) {
    auto& local_frontiers = get_threadpool().workers[w].local_frontiers;
    local_frontiers.clear();
    
    size_t beg = next.fetch_add(gamma, std::memory_order_relaxed);

    while(beg < num_this_frontiers) {
      size_t end = std::min(beg + gamma, num_this_frontiers);
      for(size_t i=beg; i<end; i++) {
        auto u = this_frontiers[i];
        for (eidType k = rowptr[u]; k < rowptr[u+1]; k++) {
          auto v = col[k];
          if(v != u && __sync_bool_compare_and_swap(&distances[v], std::numeric_limits<weight_type>::max(), distances[u] + 1)) {
            local_frontiers.push_back(v);
          }
        }
      }
      beg = next.fetch_add(gamma, std::memory_order_relaxed);
    }
      
    size_t next_beg = num_next_frontiers.fetch_add(local_frontiers.size(), std::memory_order_relaxed);
    std::memcpy(next_frontiers + next_beg, local_frontiers.data(), local_frontiers.size()*sizeof(vidType));
  }
  
  void BFS_parallel(vidType source, weight_type * distances) {

    //printf("parallel BFS\n");
    
    std::atomic<size_t> num_next_frontiers(0);
    this_frontiers[0] = source;
    distances[source] = 0;
    size_t num_this_frontiers  = 1;
    size_t num_remaining_nodes = N;

    std::vector<std::future<void>> futures;
  
    while(num_this_frontiers) {

      if(num_remaining_nodes <= 1024) {
        step_through(num_this_frontiers, distances);
        break;
      }
      
      // distribute the work among workers using static scheduling algorithm
      size_t chunk_size = std::max(gamma, (num_this_frontiers + W - 1) / W);
      for(size_t w=0, beg=0, end; w<W && beg < num_this_frontiers; beg = end, w++) {
        end = std::min(beg + chunk_size, num_this_frontiers);
        // last task
        if(w + 1 == W || end == num_this_frontiers) {
          step(w, beg, end, distances, num_next_frontiers);
          break;
        }
        else {
          //executor.silent_async(task);
          futures.push_back( get_threadpool().workers[w].async(
            [this, w, beg, end, distances, &num_next_frontiers](){
              step(w, beg, end, distances, num_next_frontiers);
              return true;
            })
          );
        }
      }

      //// distribute work using dynamic scheduler
      //std::atomic<size_t> next(0);

      //for(size_t w=0; w<W;) {
      //  // last task
      //  if(++w == W) {
      //    step(...);
      //    break;
      //  }
      //  else {
      //    //executor.silent_async(task);
      //    futures.push_back( workers[w-1].async(task) );
      //  }
      //}

  
      //executor.wait_for_all();
      for(auto& fu : futures) {
        fu.get();
      }
      futures.clear();

      // swap this and next frontiers
      num_remaining_nodes -= num_this_frontiers; 
      std::swap(this_frontiers, next_frontiers);
      num_this_frontiers = num_next_frontiers.exchange(0, std::memory_order_relaxed); 
    }


  }

  void BFS_sequential(vidType source, weight_type * distances) {
    
    this_frontiers[0] = source;
    distances[source] = 0;
    size_t num_this_frontiers = 1;

    while (num_this_frontiers) {
      size_t num_next_frontiers = 0;
      for(size_t t = 0; t<num_this_frontiers; t++) {
        auto u = this_frontiers[t];
        for(auto i = rowptr[u]; i < rowptr[u+1]; i++) {
          auto v = col[i];
          if (v != u && distances[u] + 1 < distances[v]) {
            distances[v] = distances[u] + 1;
            next_frontiers[num_next_frontiers++] = v;
          }
        }
      }
      std::swap(this_frontiers, next_frontiers);
      num_this_frontiers = num_next_frontiers;
    }
  }
  
  void step_through(size_t num_this_frontiers, weight_type * distances) {
    
    while (num_this_frontiers) {
      size_t num_next_frontiers = 0;
      for(size_t t = 0; t<num_this_frontiers; t++) {
        auto u = this_frontiers[t];
        for(auto i = rowptr[u]; i < rowptr[u+1]; i++) {
          auto v = col[i];
          if (v != u && distances[u] + 1 < distances[v]) {
            distances[v] = distances[u] + 1;
            next_frontiers[num_next_frontiers++] = v;
          }
        }
      }
      std::swap(this_frontiers, next_frontiers);
      num_this_frontiers = num_next_frontiers;
    }
  }

};  // end of Graph

int main() {
  
  const size_t N = 5;
  const size_t M = 12;
  std::array<eidType, N+1> row = {0, 2, 5, 7, 10, 12};
  std::array<vidType, M> col = {1, 4, 0, 2, 3, 1, 3, 1, 2, 4, 0, 3};
  std::vector<weight_type> distances_p(N, std::numeric_limits<weight_type>::max());
  std::vector<weight_type> distances_s(N, std::numeric_limits<weight_type>::max());
  
  Graph graph(row.data(), col.data(), N, M);

  //graph.BFS_parallel(0, distances_p.data());
  graph.BFS(0, distances_p.data());
  graph.BFS_sequential(0, distances_s.data());

  //BFS(0, distances.data(), row.data(), col.data(), N);

  for(size_t i=0; i<N; i++) {
    printf("distances[%5zu] = (%5llu, %5llu)\n", i, distances_p[i], distances_s[i]);
    //if(distances_p[i] != distances_s[i]) {
    //  throw std::runtime_error("distance mismatch");
    //}
  }
  

}









