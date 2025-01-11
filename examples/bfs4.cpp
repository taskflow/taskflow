#include <cstdint>
#include <limits>

using eidType = std::uint64_t;
using vidType = std::uint64_t;
using weight_type = std::uint64_t;

struct BaseGraph {};

// ---- start here

#include <taskflow/taskflow.hpp>


//constexpr static size_t alpha {1<<20};
constexpr static size_t alpha {0};        // threshold to parallelism

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

  size_t num_workers() const { return workers.size(); }
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
// Remainders
// ----------------------------------------------------------------------------

struct Remainders {
  
  size_t num_remainders;
  vidType head, tail;
  std::vector<vidType> next, prev;
  std::vector<std::future<void>> futures;

  void init(size_t N) {

    // N should be greater than 1
    if(N == 0) {
      throw std::runtime_error("N must be greater than 1");
    }

    num_remainders = N;

    head = 0;
    tail = N+1;
    next.resize(N+2);
    prev.resize(N+2);
    
    // sequential initialization
    if(N <= alpha) {
      #pragma omp for simd
      for(size_t i=1; i<=N; i++) {
        next[i] = i+1;
        prev[i] = i-1;
      }
    }
    // parallel initialization
    else {
      auto& threadpool = get_threadpool();
      size_t W = threadpool.num_workers();
      size_t chunk_size = (N + W - 1) / W;
      for(size_t w=0, beg=0, end; w<W && beg < N; beg = end, w++) {
        end = std::min(beg + chunk_size, N);
        // last task
        if(w + 1 == W || end == N) {
          #pragma omp for simd
          for(size_t i=beg; i<end; i++) { 
            next[i+1] = i+2; 
            prev[i+1] = i; 
          }
          break;
        }
        else {
          futures.push_back(get_threadpool().workers[w].async(
            [this, beg, end](){ 
              #pragma omp for simd
              for(size_t i=beg; i<end; i++) { 
                next[i+1] = i+2; 
                prev[i+1] = i; 
              }
              return true; 
            }
          ));
        }
      }

      // sync
      for(auto& fu : futures) {
        fu.get();
      }
      futures.clear();
    }

    next[head] = 1;
    prev[head] = head;

    next[tail] = tail;
    prev[tail] = N;
  }

  auto front() { return next[head]; }

  void remove(vidType u) {
    // vertex u means u + 1 here
    ++u;
    auto p = prev[u];
    auto n = next[u];
    next[p] = n;
    prev[n] = p;
    --num_remainders;

    // this seems redundant given the property of this problem
    prev[u] = next[u] = u;
  }
  
  size_t size() const {
    return num_remainders;
  }

  void partition(size_t chunk_size) {
    
    size_t c = 0;
    auto begu = next[head];

    for(auto beg=next[head]; beg != tail; beg = next[beg]) {
      if(++c >= chunk_size) {

        auto endu = next[beg];

        // got a new partition - can check if this is the last here
        std::cout << "partition [" << begu-1 << ", " << endu-1 <<")\n";

        // reset for the next partition
        c = 0;
        begu = endu;
      }
    }

    if(begu != tail) {
      std::cout << "partition [" << begu-1 << ", " << tail-1 << ")\n"; 
    }
  }

  void dump(std::ostream& ostream) {
    
    std::ostringstream oss;
    oss << "head->";
    for(auto beg = next[head]; beg != tail; beg = next[beg]) {
      oss << beg-1 << "->";
    }
    oss << "tail [size=" << size() << "]\n";
    ostream << oss.str();
  }

};

Remainders& get_remainders() {
  static Remainders remainders;
  return remainders;
}

// ----------------------------------------------------------------------------
// Graph
// ----------------------------------------------------------------------------
class Graph : public BaseGraph {
  
  eidType* rowptr;
  vidType* col;
  uint64_t N;
  uint64_t M;
  
  size_t W;

  vidType* this_frontiers;
  vidType* next_frontiers;

  public:
  

  Graph(eidType* in_rowptr, vidType* in_col, uint64_t in_N, uint64_t in_M) :
    rowptr (in_rowptr), 
    col    (in_col), 
    N      (in_N), 
    M      (in_M),
    W      ((N + M) <= alpha ? 0 : get_threadpool().workers.size()) {

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
    ((N + M) <= alpha) ? BFS_sequential(source, distances) : BFS_parallel(source, distances);
  }

  size_t num_edges(vidType v) const {
    return rowptr[v+1] - rowptr[v];
  }
  
  // step_td - parallel
  void step_td(
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
        //auto expect = std::numeric_limits<weight_type>::max();
        //auto target = distances[u] + 1;
        //if(v != u && __atomic_compare_exchange(&distances[v], &expect, &target, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
        //  local_frontiers.push_back(v);
        //}
      }
    }
    size_t next_beg = num_next_frontiers.fetch_add(local_frontiers.size(), std::memory_order_relaxed);
    std::memcpy(next_frontiers + next_beg, local_frontiers.data(), local_frontiers.size()*sizeof(vidType));
  }
  
  // step_td - sequential
  size_t step_td(size_t num_this_frontiers, weight_type * distances) {
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
    return num_next_frontiers;
  }
  
  // step_bu - parallel
  void step_bu(
    size_t frontier_level,
    size_t w, 
    vidType begu, 
    vidType endu, 
    weight_type* distances, 
    const Remainders& remainders,
    std::atomic<size_t>& num_next_frontiers
  ) {
    auto& local_frontiers = get_threadpool().workers[w].local_frontiers;
    local_frontiers.clear();
    
    for(auto i=begu; i!=endu; i=remainders.next[i]) {
      auto u = i-1;
      for (auto k = rowptr[u]; k < rowptr[u+1]; k++) {
        auto v = col[k];
        if(distances[v] == frontier_level) {
          distances[u] = frontier_level + 1;
          local_frontiers.push_back(u);
          break;
        }
      }
    }
    size_t next_beg = num_next_frontiers.fetch_add(local_frontiers.size(), std::memory_order_relaxed);
    std::memcpy(next_frontiers + next_beg, local_frontiers.data(), local_frontiers.size()*sizeof(vidType));
  }
  
  // step_bu - sequential
  size_t step_bu(size_t frontier_level, weight_type* distances, const Remainders& remainders) {
    size_t num_next_frontiers = 0;
    for(auto i=remainders.next[remainders.head]; i!=remainders.tail; i=remainders.next[i]) {
      auto u = i-1;
      for (auto k = rowptr[u]; k < rowptr[u+1]; k++) {
        auto v = col[k];
        if(distances[v] == frontier_level) {
          distances[u] = frontier_level + 1;
          next_frontiers[num_next_frontiers++] = u; 
          break;
        }
      }
    }
    return num_next_frontiers;
  }
  
  // step - dynamic scheduling
  //void step(
  //  size_t w, 
  //  weight_type* distances, 
  //  size_t num_this_frontiers, 
  //  std::atomic<size_t>& next, 
  //  std::atomic<size_t>& num_next_frontiers
  //) {
  //  auto& local_frontiers = get_threadpool().workers[w].local_frontiers;
  //  local_frontiers.clear();
  //  
  //  size_t beg = next.fetch_add(gamma, std::memory_order_relaxed);

  //  while(beg < num_this_frontiers) {
  //    size_t end = std::min(beg + gamma, num_this_frontiers);
  //    for(size_t i=beg; i<end; i++) {
  //      auto u = this_frontiers[i];
  //      for (eidType k = rowptr[u]; k < rowptr[u+1]; k++) {
  //        auto v = col[k];
  //        if(v != u && __sync_bool_compare_and_swap(&distances[v], std::numeric_limits<weight_type>::max(), distances[u] + 1)) {
  //          local_frontiers.push_back(v);
  //        }
  //      }
  //    }
  //    beg = next.fetch_add(gamma, std::memory_order_relaxed);
  //  }
  //    
  //  size_t next_beg = num_next_frontiers.fetch_add(local_frontiers.size(), std::memory_order_relaxed);
  //  std::memcpy(next_frontiers + next_beg, local_frontiers.data(), local_frontiers.size()*sizeof(vidType));
  //}
  
  void BFS_parallel(vidType source, weight_type * distances) {

    //printf("parallel BFS\n");
    
    // initialize source nodes
    std::atomic<size_t> num_next_frontiers(0);
    this_frontiers[0] = source;
    distances[source] = 0;
    size_t num_this_frontiers  = 1;
    size_t frontier_level = 0;
    
    // initialize remainders
    auto& remainders = get_remainders();
    remainders.init(N);

    std::vector<std::future<void>> futures;
    
    // initialize unexplored edge size
    size_t mu = M;
  
    while(num_this_frontiers) {
      
      // update the remainders 
      size_t mf = 0;
      for(size_t i=0; i<num_this_frontiers; i++) {
        auto f = this_frontiers[i];
        mf += num_edges(f);
        remainders.remove(f);
      }
      if(mu < mf) {
        throw std::runtime_error("bug\n");
      }
      mu -= mf;
      
      // number of scans for top-down and buttom-up bfs
      size_t td_scans = mf + num_this_frontiers; 
      size_t bu_scans = remainders.size() + mu;
      
      //remainders.dump(std::cout);
      //printf("ntf=%zu, remainders=%zu, mf=%zu, mu=%zu (td_scans=%zu, bu_scans=%zu)\n", num_this_frontiers, remainders.size(), mf, mu, td_scans, bu_scans);

      
      // if the remaining work is fewer than the parallel threshold, do sequential bfs
      if(remainders.size() <= 1024) {
        step_through(num_this_frontiers, distances);
        break;
      }

      // case 2: if the number of bottom-up scans is fewer -> do bottom-up scan
      if(bu_scans < td_scans) {
        //printf("bottom up bfs\n");
        //size_t chunk_size = (remainders.size() + W - 1) / W;

        if(bu_scans <= alpha) {
          num_this_frontiers = step_bu(frontier_level, distances, remainders);
        }
        else {
          const size_t chunk_size = (bu_scans + W - 1) / W;
          size_t w = 0;
          size_t c = 0;
          auto begu = remainders.next[remainders.head];
          for(auto beg=begu; beg != remainders.tail; beg = remainders.next[beg]) {
            auto endu = remainders.next[beg];
            c += (num_edges(beg-1) + 1);
            // a new partition is formed
            if(c >= chunk_size || endu == remainders.tail) {
              //std::cout << "partition [" << begu-1 << ", " << endu-1 <<") = " << c << " scans\n";
              // if this is the last partition
              if(w + 1 == W || endu == remainders.tail) {
                step_bu(frontier_level, w, begu, endu, distances, remainders, num_next_frontiers);
                break;
              }
              // otherwise, invoke a task to handle it
              else {
                futures.push_back( get_threadpool().workers[w].async(
                  [this, frontier_level, w, begu, endu, distances, &remainders, &num_next_frontiers](){
                    step_bu(frontier_level, w, begu, endu, distances, remainders, num_next_frontiers);
                    return true;
                  })
                );
              }
              // reset for the next partition
              c = 0;
              begu = endu;
              ++w;
            }
          }
          // sync futures
          for(auto& fu : futures) {
            fu.get();
          }
          futures.clear();
          num_this_frontiers = num_next_frontiers.exchange(0, std::memory_order_relaxed);
        }
      }
      // case 3: otherwise, we do top down
      // distribute the work among workers using static scheduling algorithm
      else {
        //printf("top down bfs\n");
        
        if(td_scans <= alpha) {
          num_this_frontiers = step_td(num_this_frontiers, distances);
        }
        else { 
          const size_t chunk_size = (td_scans + W - 1) / W;
          size_t w = 0;
          size_t c = 0;
          size_t begu = 0;

          for(auto beg=begu; beg < num_this_frontiers; ++beg) {

            c += (num_edges(this_frontiers[beg]) + 1);
            size_t endu = beg + 1;

            // a new partition is formed
            if(c >= chunk_size || endu == num_this_frontiers) {
              //std::cout << "partition [" << begu << ", " << endu <<") = " << c << " scans\n";
              // if this is the last partition
              if(w + 1 == W || endu == num_this_frontiers) {
                step_td(w, begu, endu, distances, num_next_frontiers);
                break;
              }
              // otherwise, invoke a task to handle ti
              else {
                futures.push_back( get_threadpool().workers[w].async(
                  [this, w, begu, endu, distances, &num_next_frontiers](){
                    step_td(w, begu, endu, distances, num_next_frontiers);
                    return true;
                  })
                );
              }

              // reset for the next partition
              c = 0;
              begu = endu;
              ++w;
            }
          }
          // sync futures
          for(auto& fu : futures) {
            fu.get();
          }
          futures.clear();
          num_this_frontiers = num_next_frontiers.exchange(0, std::memory_order_relaxed);
        }

        //size_t chunk_size = (num_this_frontiers + W - 1) / W;
        //for(size_t w=0, beg=0, end; w<W && beg < num_this_frontiers; beg = end, w++) {
        //  end = std::min(beg + chunk_size, num_this_frontiers);
        //  // last task
        //  if(w + 1 == W || end == num_this_frontiers) {
        //    step_td(w, beg, end, distances, num_next_frontiers);
        //    break;
        //  }
        //  else {
        //    //executor.silent_async(task);
        //    futures.push_back( get_threadpool().workers[w].async(
        //      [this, w, beg, end, distances, &num_next_frontiers](){
        //        step_td(w, beg, end, distances, num_next_frontiers);
        //        return true;
        //      })
        //    );
        //  }
        //}
      }


      std::swap(this_frontiers, next_frontiers);
      frontier_level++;
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
  


  
  void dump(std::ostream& os) {
    os << "graph  {\n";
    for(size_t u=0; u<N; u++) {
      for(size_t i=rowptr[u]; i<rowptr[u+1]; i++) {
        if(u <= col[i]) {
          os << "  " << u << " -- " << col[i] << ";\n";
        }
      }
    }
    os << "}\n";
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
  graph.dump(std::cout);

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
  

  //auto remainders = get_remainders();
  //remainders.init(10);
  //remainders.dump(std::cout);

  //remainders.remove(5);
  //remainders.dump(std::cout);
  //
  //remainders.remove(9);
  //remainders.dump(std::cout);

  //remainders.partition(4);

  //remainders.remove(0);
  //remainders.dump(std::cout);
  //
  //remainders.remove(6);
  //remainders.dump(std::cout);
  //
  //remainders.remove(2);
  //remainders.dump(std::cout);
  //
  //remainders.remove(8);
  //remainders.dump(std::cout);
  //
  //remainders.remove(1);
  //remainders.dump(std::cout);
  //
  //remainders.remove(3);
  //remainders.dump(std::cout);
  //
  //remainders.remove(4);
  //remainders.dump(std::cout);
  //
  //remainders.remove(5);
  //remainders.dump(std::cout);
  //
  //remainders.remove(7);
  //remainders.dump(std::cout);
  //
  //remainders.remove(9);
  //remainders.dump(std::cout);

}









