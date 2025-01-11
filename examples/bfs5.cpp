#include <cstdint>
#include <limits>

using eidType = std::uint64_t;
using vidType = std::uint64_t;
using weight_type = std::uint64_t;

struct BaseGraph {};

// ---- start here

#include <taskflow/taskflow.hpp>


//constexpr static size_t ALPHA {1<<20};
constexpr static size_t ALPHA {0};        // threshold to parallel BFS
constexpr static size_t GAMMA {0};        // threshold to parallel step

namespace tf {

// Function: make_for_each_index_task
template <typename R, typename C, typename P = DefaultPartitioner>
auto make_for_each_index_task(R range, C c, P part = P()){
  
  using range_type = std::decay_t<unwrap_ref_decay_t<R>>;

  return [=] (Runtime& rt) mutable {
      
    // fetch the iterator values
    range_type r = range;
    
    // nothing to be done if the range is invalid
    if(is_index_range_invalid(r.begin(), r.end(), r.step_size())) {
      return;
    }

    size_t W = rt.executor().num_workers();
    size_t N = r.size();

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= part.chunk_size()) {

      part([&](){ c(r); })();
      return;
    }

    PreemptionGuard preemption_guard(rt);
    
    if(N < W) {
      W = N;
    }
    
    // static partitioner
    if constexpr(part.type() == PartitionerType::STATIC) {
      for(size_t w=0, curr_b=0; w<W && curr_b < N;) {
        auto chunk_size = part.adjusted_chunk_size(N, W, w);
        auto task = part([=] () mutable {
          part.loop(N, W, curr_b, chunk_size, [=] (size_t part_b, size_t part_e) {
            c(r.discrete_domain(part_b, part_e));
          });
        });
        (++w == W || (curr_b += chunk_size) >= N) ? task() : rt.silent_async(task);
      }
    }
    // dynamic partitioner
    else {
      auto next = std::make_shared<std::atomic<size_t>>(0);
      for(size_t w=0; w<W;) {
        auto task = part([=] () mutable {
          part.loop(N, W, *next, [=] (size_t part_b, size_t part_e) {
            c(r.discrete_domain(part_b, part_e));
          });
        });
        (++w == W) ? task() : rt.silent_async(task);
      }
    }
  };
}

}  // end of namespace tf -----------------------------------------------------


// ----------------------------------------------------------------------------
// Threadpool
// ----------------------------------------------------------------------------
  
struct Worker {
  std::vector<vidType> local_frontiers;
};

tf::Executor& get_executor() {
  static tf::Executor executor;
  return executor;
};

Worker& per_worker() {
  thread_local Worker worker;
  return worker;
}

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
  
  size_t W;

  vidType* this_frontiers;
  vidType* next_frontiers;
    
  size_t num_remainders;
  size_t num_this_frontiers;
  size_t frontier_level;
  size_t mf;
  size_t mu;

  public:
  

  Graph(eidType* in_rowptr, vidType* in_col, uint64_t in_N, uint64_t in_M) :
    rowptr    (in_rowptr), 
    col       (in_col), 
    N         (in_N), 
    M         (in_M),
    W         ((N + M) <= ALPHA ? 0 : get_executor().num_workers()) {

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
    ((N + M) <= ALPHA) ? BFS_sequential(source, distances) : BFS_parallel(source, distances);
  }

  size_t num_edges(vidType v) const {
    return rowptr[v+1] - rowptr[v];
  }
  
  // step_td - parallel
  void step_td(
    size_t beg, 
    size_t end, 
    weight_type* distances, 
    std::atomic<size_t>& num_next_frontiers
  ) {
    auto& local_frontiers = per_worker().local_frontiers;
    local_frontiers.clear();
    
    size_t lmf = 0;
    for(size_t i=beg; i<end; i++) {
      auto u = this_frontiers[i];
      for (eidType k = rowptr[u]; k < rowptr[u+1]; k++) {
        auto v = col[k];
        if(v != u && __sync_bool_compare_and_swap(&distances[v], std::numeric_limits<weight_type>::max(), distances[u] + 1)) {
          local_frontiers.push_back(v);
          lmf += num_edges(v);
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
    __atomic_fetch_add(&mf, lmf, __ATOMIC_RELAXED);
  }
  
  // step_td - sequential
  size_t step_td(weight_type * distances) {
    size_t num_next_frontiers = 0;
    for(size_t t = 0; t<num_this_frontiers; t++) {
      auto u = this_frontiers[t];
      for(auto i = rowptr[u]; i < rowptr[u+1]; i++) {
        auto v = col[i];
        if (v != u && distances[u] + 1 < distances[v]) {
          distances[v] = distances[u] + 1;
          next_frontiers[num_next_frontiers++] = v;
          mf += num_edges(v);
        }
      }
    }
    return num_next_frontiers;
  }
  
  // step_bu - parallel
  void step_bu(size_t begu, size_t endu,  weight_type* distances, std::atomic<size_t>& num_next_frontiers) {
    auto& local_frontiers = per_worker().local_frontiers;
    local_frontiers.clear();
    
    for(vidType u=begu; u<endu; ++u) {
      // u is unexplored
      if(distances[u] == std::numeric_limits<weight_type>::max()) {
        for (auto k = rowptr[u]; k < rowptr[u+1]; k++) {
          auto v = col[k];
          if(distances[v] == frontier_level) {
            distances[u] = frontier_level + 1;
            local_frontiers.push_back(u);
            break;
          }
        }
      }
    }
    size_t next_beg = num_next_frontiers.fetch_add(local_frontiers.size(), std::memory_order_relaxed);
    std::memcpy(next_frontiers + next_beg, local_frontiers.data(), local_frontiers.size()*sizeof(vidType));
  }
  
  // step_bu - sequential
  size_t step_bu(weight_type* distances) {
    size_t num_next_frontiers = 0;
    for(vidType u=0; u<N; u++) {
      // u is unexplored
      if(distances[u] == std::numeric_limits<weight_type>::max()) {
        for(auto k = rowptr[u]; k < rowptr[u+1]; k++) {
          auto v = col[k];
          if(distances[v] == frontier_level) {
            distances[u] = frontier_level + 1;
            next_frontiers[num_next_frontiers++] = u; 
            break;
          }
        }
      }
    }
    return num_next_frontiers;
  }
  
  void BFS_parallel(vidType source, weight_type * distances) {

    //printf("parallel BFS\n");

    auto& executor = get_executor();

    // initialize source nodes
    std::atomic<size_t> num_next_frontiers(0);
    this_frontiers[0] = source;
    distances[source] = 0;
    num_remainders = N;
    num_this_frontiers  = 1;
    frontier_level = 0;
    mf = num_edges(source);
    
    // initialize unexplored edge size
    mu = M;
  
    while(num_this_frontiers) {
      
      // update the remainders 
      if(mu < mf) {
        throw std::runtime_error("bug\n");
      }
      mu -= mf;
      num_remainders -= num_this_frontiers;
      
      // number of scans for top-down and buttom-up bfs
      size_t td_scans = mf + num_this_frontiers; 
      size_t bu_scans = N + mu;
      
      //printf("ntf=%zu, remainders=%zu, mf=%zu, mu=%zu (td_scans=%zu, bu_scans=%zu)\n", num_this_frontiers, num_remainders, mf, mu, td_scans, bu_scans);
      
      // reset data before moving on
      mf = 0;
      
      // if the remaining work is fewer than the parallel threshold, do sequential bfs
      //if(num_remainders <= 1024) {
      //  step_through(distances);
      //  break;
      //}

      // case 2: if the number of bottom-up scans is fewer -> do bottom-up scan
      if(bu_scans < td_scans) {
        //printf("bottom up bfs\n");

        if(bu_scans <= GAMMA) {
          num_this_frontiers = step_bu(distances);
        }
        else {
          executor.silent_async(tf::make_for_each_index_task(
            tf::IndexRange<size_t>(0, N, 1),
            [this, distances, &num_next_frontiers](tf::IndexRange<size_t> subrange) {
              step_bu(subrange.begin(), subrange.end(), distances, num_next_frontiers);
            },
            tf::StaticPartitioner(GAMMA)
          ));
          executor.wait_for_all();
          num_this_frontiers = num_next_frontiers.exchange(0, std::memory_order_relaxed);
        }
      }
      // case 3: otherwise, we do top down
      // distribute the work among workers using static scheduling algorithm
      else {
        //printf("top down bfs\n");
        if(td_scans <= GAMMA) {
          num_this_frontiers = step_td(distances);
        }
        else { 
          executor.silent_async(tf::make_for_each_index_task(
            tf::IndexRange<size_t>(0, num_this_frontiers, 1),
            [this, distances, &num_next_frontiers](tf::IndexRange<size_t> subrange){
              step_td(subrange.begin(), subrange.end(), distances, num_next_frontiers);
            },
            tf::StaticPartitioner(GAMMA)
          ));   
          executor.wait_for_all();
          num_this_frontiers = num_next_frontiers.exchange(0, std::memory_order_relaxed);
        }
      }

      std::swap(this_frontiers, next_frontiers);
      frontier_level++;
    }


  }

  void BFS_sequential(vidType source, weight_type * distances) {
    
    this_frontiers[0] = source;
    distances[source] = 0;
    num_this_frontiers = 1;

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
  
  void step_through(weight_type * distances) {
    
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









