#include <taskflow/taskflow.hpp>

using eidType = uint64_t;
using vidType = uint64_t;
using weight_type = uint64_t;

constexpr auto MAX_DISTANCE = std::numeric_limits<weight_type>::max();

struct BaseGraph {};

// ---- start here

class Graph : public BaseGraph {

  eidType* rowptr;
  vidType* col;
  uint64_t N;
  uint64_t M;

  public:

  Graph(eidType* in_rowptr, vidType* in_col, uint64_t in_N, uint64_t in_M) :
    rowptr(in_rowptr), col(in_col), N(in_N), M(in_M) {
  }
  
  ~Graph() {
    // destructor logic.
    // If you perform any memory allocations with malloc, new, etc. you must free
    //   them here to avoid memory leaks.
  }


  enum FrontierType : int {
    THIS = 0,
    NEXT = 1,
    MAXN = 2
  };
  
  struct Worker {
    std::array<std::vector<vidType>, FrontierType::MAXN> frontiers;
    size_t prefix_sum[FrontierType::MAXN];
  };
  
  // find min
  size_t find_from(std::vector<Worker>& workers, size_t value, FrontierType type) {
  
    size_t beg = 0;
    size_t end = workers.size() - 1;
    size_t mid;
    size_t best = end;
  
    do {
      mid = (beg+end)/2;
  
      if(value < workers[mid].prefix_sum[type]) {
        best = std::min(best, mid);
        end = mid;
      }
      else {
        beg = mid + 1;
      }
  
    }while(beg < end);
  
    return best;
  }
  
  // find min
  size_t find_to(std::vector<Worker>& workers, size_t value, FrontierType type) {
  
    size_t beg = 0;
    size_t end = workers.size() - 1;
    size_t mid;
    size_t best = end;
  
    do {
      mid = (beg+end)/2;
  
      if(value <= workers[mid].prefix_sum[type]) {
        best = std::min(best, mid);
        end = mid;
      }
      else {
        beg = mid + 1;
      }
  
    }while(beg < end);
  
    return best;
  }
  
  
  
  size_t num_frontiers(std::vector<Worker>& workers, FrontierType type) {
    size_t n = 0;
    for(auto& w : workers) {
      n += w.frontiers[type].size();
      w.prefix_sum[type] = n;
    }
    return n;
  }
  
  void BFS(vidType source, weight_type * distances) {
  
    const size_t W = std::thread::hardware_concurrency();
    const size_t alpha = 1;
  
    // W must be greater than 0
    if(W == 0) {
      throw std::runtime_error("not enough worker to do the work");
    }
    
    tf::Executor executor(W);
  
    std::vector<Worker> workers(W);
    
    // TODO
    distances[source] = 0;
    workers[0].frontiers[FrontierType::THIS].push_back(source);
  
    //std::srand(std::time(nullptr)); // use current time as seed for random generator
    //const size_t alpha = std::rand() % 1 + 1;
    //printf("alpha = %zu\n", alpha);
    //for(size_t w=0; w<workers.size(); w++) {
    //  workers[w].frontiers[FrontierType::THIS].resize(std::rand() % 2, 0);
    //  printf("worker %2zu resize to %5zu frontiers\n", w, workers[w].frontiers[FrontierType::THIS].size());
    //}
  
  
  
    
    while(true) {
  
      // calculate the number of this frontiers
      size_t num_this_frontiers = num_frontiers(workers, FrontierType::THIS);
      
      // no more tasks to do
      if(num_this_frontiers == 0) {
        break;
      }
      
      // distribute this-frontiers among workers
      size_t chunk_size = std::max(alpha, (num_this_frontiers + W - 1) / W);
      size_t beg = 0;
      size_t end;
      size_t w = 0;
      //std::atomic<size_t> test = 0;
  
      //printf("chunk_size = %zu; num_this_frontiers = %zu\n", chunk_size, num_this_frontiers);
  
      while(beg < num_this_frontiers) {
  
        end = std::min(beg + chunk_size, num_this_frontiers);
  
        // has task
        auto task = [
          this,
          w,             // for next frontiers
          beg,
          end,
          &workers,
          distances
          //&test
        ] () mutable {
  
          auto& next_frontier = workers[w].frontiers[FrontierType::NEXT];
  
          next_frontier.clear();
  
          // workers[from].frontiers - workers[to].frontiers
          auto from = find_from(workers, beg, FrontierType::THIS);
          auto to   = find_to(workers, end, FrontierType::THIS);
          
          //printf("worker %2zu handle discrete domain [%zu, %zu) spanning [%zu, %zu]\n", w, beg, end, from, to);
  
          for(size_t i=from; i<=to; i++) {
            // fetch this-frontiers from worker i
            auto& this_frontier = workers[i].frontiers[FrontierType::THIS];
            auto cur_prefix_sum = workers[i].prefix_sum[FrontierType::THIS];
            auto pre_prefix_sum = (i ? workers[i-1].prefix_sum[FrontierType::THIS] : 0);
            auto f = std::max(beg, pre_prefix_sum) - (i ? pre_prefix_sum : 0);
            auto t = std::min(end, cur_prefix_sum) - (i ? pre_prefix_sum : 0);
            //printf("worker %2zu on frontiers %2zu: [%5zu, %5zu) of size %zu\n", w, i, f, t, t-f);
  
            //if(t > this_frontier.size() || f > this_frontier.size()) {
            //  throw std::runtime_error("failed 1");
            //} 
            
            // DEBUG
            //test += (t-f);
            //for(size_t j=f; j<t; j++) {
            //  this_frontier[j]++;
            //  test += this_frontier[j];
            //}
            
            // one step of bfs
            for(size_t j=f; j<t; j++) {
              auto u = this_frontier[j];
              for (eidType k = rowptr[u]; k < rowptr[u+1]; k++) {
                auto v = col[k];
                if(__sync_bool_compare_and_swap(&distances[v], MAX_DISTANCE, distances[u] + 1)) {
                  next_frontier.push_back(v);
                }
              }
            }
          }
        };
  
        // last task
        if(++w == W || (beg = end) >= num_this_frontiers) {
          task();
          break;
        }
        else {
          executor.silent_async(task);
        }
      }
  
      executor.wait_for_all();
        
      //if(test != num_this_frontiers) {
      //  printf("test = %zu\n", test.load());
      //  throw std::runtime_error("failed 2\n");
      //}
  
      // TODO: swap this and next fontiers
      for(size_t wid=0; wid < workers.size(); wid++) {
        //for(size_t i=0; i<workers[wid].frontiers[FrontierType::THIS].size(); i++) {
        //  if(workers[wid].frontiers[FrontierType::THIS][i] != 1) {
        //    throw std::runtime_error("failed 3\n");
        //  }
        //}   
        workers[wid].frontiers[FrontierType::THIS].clear();
        std::swap(workers[wid].frontiers[FrontierType::THIS], workers[wid].frontiers[FrontierType::NEXT]); 
      }
    }
  }

};  // end of Graph

int main() {
  
  const size_t N = 5;
  const size_t M = 12;
  std::array<eidType, N+1> row = {0, 2, 5, 7, 10, 12};
  std::array<vidType, M> col = {1, 4, 0, 2, 3, 1, 3, 1, 2, 4, 0, 3};
  std::vector<weight_type> distances(N, MAX_DISTANCE);
  
  Graph graph(row.data(), col.data(), N, M);
  graph.BFS(0, distances.data());

  //BFS(0, distances.data(), row.data(), col.data(), N);

  for(size_t i=0; i<distances.size(); i++) {
    printf("distances[%5zu] = ", i);
    std::cout << distances[i] << '\n';
  }

}









