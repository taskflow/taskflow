// 2019/02/20 - modified by Tsung-Wei Huang 
//   - added empty_subflow benchmarks
//   - added steady_subflow benchmarks 
//
// 2018/12/04 - modified by Tsung-Wei Huang
//   - replaced privatized executor with work stealing executor
//
// 2018/10/24 - modified by Tsung-Wei Huang
//   - Taskflow is templated at executor
//   - added graph-level comparison with different thread pools
//
// 2018/09/19 - created by Tsung-Wei Huang
//
// This program is used to benchmark the taskflow under different types 
// of workloads.

#include <taskflow/taskflow.hpp>
#include <chrono>
#include <random>
#include <climits>
#include <iomanip>

constexpr int WIDTH = 15;

// Procedure: benchmark
#define BENCHMARK(TITLE, F)                                                                \
std::cout                                                                                  \
<< std::setw(WIDTH) << TITLE << std::flush                                                 \
<< std::setw(WIDTH) << F<tf::BasicTaskflow<tf::SimpleExecutor>>() << std::flush            \
<< std::setw(WIDTH) << F<tf::BasicTaskflow<tf::ProactiveExecutor>>() << std::flush         \
<< std::setw(WIDTH) << F<tf::BasicTaskflow<tf::SpeculativeExecutor>>() << std::flush       \
<< std::setw(WIDTH) << F<tf::BasicTaskflow<tf::WorkStealingExecutor>>() << std::flush      \
<< std::setw(WIDTH) << F<tf::BasicTaskflow<tf::EigenWorkStealingExecutor>>() << std::flush \
<< std::endl;
  
// ============================================================================
// Dynamic Stem
// ============================================================================

// Function: dynamic_stem
template <typename T>
auto dynamic_stem() {
  
  auto beg = std::chrono::high_resolution_clock::now();
  {  
    const int L = 1024;

    std::atomic<size_t> sum {0};

    T tf;

    std::optional<tf::Task> prev;

    for(int l=0; l<L; ++l) {
      auto curr = tf.emplace([&, l] (auto& subflow) {
        sum.fetch_add(1, std::memory_order_relaxed);
        std::optional<tf::Task> p;
        for(int k=0; k<L; k++) {
          auto c = subflow.emplace([&] () {
            sum.fetch_add(1, std::memory_order_relaxed);
          });
          if(p) {
            p->precede(c);
          }
          p = c;
        }
        if(l & 1) {
          subflow.detach();
        }
      });

      if(prev) {
        prev->precede(curr);
      }
      prev = curr;
    }

    tf.wait_for_all();

    assert(sum == L*(L+1));
  }
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// ============================================================================
// Map-Reduce
// ============================================================================

// Function: map_reduce
template <typename T>
auto map_reduce() {
  
  auto beg = std::chrono::high_resolution_clock::now();
  {  
    const int num_batches = 65536;

    std::vector<int> C(1024, 10);
    std::atomic<size_t> sum {0};

    T tf;

    std::optional<tf::Task> prev;

    for(int i=0; i<num_batches; ++i) {
      
      auto [s, t] = tf.parallel_for(C.begin(), C.end(), [&] (int v) {
        sum.fetch_add(v, std::memory_order_relaxed);
      });
      
      if(prev) {
        prev->precede(s);
      }

      prev = t;
    }

    tf.wait_for_all();
 
    assert(sum == num_batches * C.size() * 10);
  }
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// ============================================================================
// Level Graph
// ============================================================================

// Function: level_graph
template <typename T>
auto level_graph() {
  
  const int num_levels = 2048;
  const int num_nodes_per_level = 1024;
  
  auto beg = std::chrono::high_resolution_clock::now();
  {
    std::atomic<size_t> sum {0};

    T tf;

    std::vector< std::vector<tf::Task> > tasks;

    tasks.resize(num_levels);
    for(int l=0; l<num_levels; ++l) {
      for(int i=0; i<num_nodes_per_level; ++i) {
        tasks[l].push_back(tf.emplace([&] () {
          sum.fetch_add(1, std::memory_order_relaxed);
        }));
      }
    }

    // connections for each level l to l+1
    for(int l=0; l<num_levels-1; ++l) {
      for(int i=0; i<num_nodes_per_level; ++i) {
        for(int j=0; j<num_nodes_per_level; j=j+i+1) {
          tasks[l][i].precede(tasks[l+1][j]);    
        }
      }
    }

    tf.wait_for_all();
 
    assert(sum == num_levels * num_nodes_per_level);
  }
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// ============================================================================
// Linear Graph
// ============================================================================

// Function: linear_graph
template <typename T>
auto linear_graph() {
  
  const int num_nodes = 1000000;
  
  auto beg = std::chrono::high_resolution_clock::now();
  {
    size_t sum {0};

    T tf;

    std::vector<tf::Task> tasks;

    for(int i=0; i<num_nodes; ++i) {
      tasks.push_back(tf.emplace([&] () { ++sum; }));
    }

    tf.linearize(tasks);
    tf.wait_for_all();
 
    assert(sum == num_nodes);
  }
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// ============================================================================
// Binary Tree
// ============================================================================

// Function: binary_tree
template <typename T>
auto binary_tree() {
  
  const int num_levels = 21;

  auto beg = std::chrono::high_resolution_clock::now();
  {  
    T tf;
    
    std::atomic<size_t> sum {0};
    std::function<void(int, tf::Task)> insert;
    
    insert = [&] (int l, tf::Task parent) {

      if(l < num_levels) {

        auto lc = tf.emplace([&] () {
          sum.fetch_add(1, std::memory_order_relaxed);
        });

        auto rc = tf.emplace([&] () {
          sum.fetch_add(1, std::memory_order_relaxed);
        });

        parent.precede(lc);
        parent.precede(rc);

        insert(l+1, lc);
        insert(l+1, rc);
      }
    };
    
    auto root = tf.emplace([&] () {
      sum.fetch_add(1, std::memory_order_relaxed);
    });

    insert(1, root);

    // synchronize until all tasks finish
    tf.wait_for_all();

    assert(sum == (1 << (num_levels)) - 1);
  }
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// ============================================================================
// Empty Jobs
// ============================================================================

// Function: empty_jobs
template <typename T>
auto empty_jobs() {
  
  const int num_tasks = 2000000;
  
  auto beg = std::chrono::high_resolution_clock::now();
  {
    T tf;

    for(size_t i=0; i<num_tasks; i++){
      tf.emplace([](){}); 
    }

    tf.wait_for_all();
  }
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// ============================================================================
// Atomic add
// ============================================================================

// Function: atomic_add
template <typename T>
auto atomic_add() {
  
  const int num_tasks = 1000000;

  auto beg = std::chrono::high_resolution_clock::now();
  {
    std::atomic<int> counter(0);
    T tf;
    for(size_t i=0; i<num_tasks; i++){
      tf.emplace([&counter](){ 
        counter.fetch_add(1, std::memory_order_relaxed);
      }); 
    }
    tf.wait_for_all();

    assert(counter == num_tasks);
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// ============================================================================
// Multiple Dispatches
// ============================================================================

// Function: multiple_dispatches
template <typename T>
auto multiple_dispatches() {

  auto create_graph = [] (T& tf, size_t N, std::atomic<int>& c) {
    for(size_t i=0; i<N; ++i) {
      auto [A, B, C, D] = tf.emplace(
        [&] () { c.fetch_add(1, std::memory_order_relaxed); },
        [&] () { c.fetch_add(1, std::memory_order_relaxed); },
        [&] () { c.fetch_add(1, std::memory_order_relaxed); },
        [&] () { c.fetch_add(1, std::memory_order_relaxed); }
      );
      A.precede(B);
      C.precede(D);
    }
  };
  
  auto beg = std::chrono::high_resolution_clock::now();
  
  {
    T tf;
    for(int p=0; p<1024; ++p) {
      std::atomic<int> counter(0);
      create_graph(tf, p, counter);
      tf.wait_for_all();
      assert(counter == p * 4);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// ============================================================================
// Nested subflow
// ============================================================================

// Function: empty_subflow
template <typename T>
auto empty_subflow() {

  std::function<void(tf::SubflowBuilder&, uint64_t)> grow;
  
  grow = [&grow] (tf::SubflowBuilder& subflow, uint64_t depth) {
    if(depth < 20) {
      subflow.emplace(
        [depth, &grow](tf::SubflowBuilder& subsubflow){ grow(subsubflow, depth+1); },
        [depth, &grow](tf::SubflowBuilder& subsubflow){ grow(subsubflow, depth+1); });
      subflow.detach();
    }
  };

  auto beg = std::chrono::high_resolution_clock::now();
  
  {
    T tf;
    tf.emplace([&] (tf::SubflowBuilder& subflow) { grow(subflow, 0); });
    tf.wait_for_all();
  }

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// Function: steady_subflow
template <typename T>
auto steady_subflow() {

  std::function<void(tf::SubflowBuilder&, uint64_t)> grow;
  
  grow = [&grow] (tf::SubflowBuilder& subflow, uint64_t depth) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if(depth < 3) {
      subflow.emplace(
        [depth, &grow](tf::SubflowBuilder& subsubflow){ grow(subsubflow, depth+1); },
        [depth, &grow](tf::SubflowBuilder& subsubflow){ grow(subsubflow, depth+1); });
      subflow.detach();
    }
  };

  auto beg = std::chrono::high_resolution_clock::now();
  
  {
    T tf;
    tf.emplace([&] (tf::SubflowBuilder& subflow) { grow(subflow, 0); });
    tf.wait_for_all();
  }

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// ----------------------------------------------------------------------------

// Function: main
int main(int argc, char* argv[]) {
  
  std::cout << std::setw(WIDTH) << "workload"
            << std::setw(WIDTH) << "tf+simple"
            << std::setw(WIDTH) << "tf+pro"
            << std::setw(WIDTH) << "tf+spec"
            << std::setw(WIDTH) << "tf+steal"
            << std::setw(WIDTH) << "tf+eigen"
            << std::endl;

  BENCHMARK("map-reduce", map_reduce);
  BENCHMARK("empty jobs", empty_jobs);
  BENCHMARK("atomic add", atomic_add);
  BENCHMARK("dispatches", multiple_dispatches);
  BENCHMARK("b-tree", binary_tree);
  BENCHMARK("linear", linear_graph);
  BENCHMARK("dag", level_graph);
  BENCHMARK("dynamic", dynamic_stem);
  BENCHMARK("empty-subflow", empty_subflow);
  BENCHMARK("steady-subflow", steady_subflow);

  
  return 0;
}



