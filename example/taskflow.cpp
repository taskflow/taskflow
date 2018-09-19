// 2018/09/18 - created by Tsung-Wei Huang
//
// This program is used to benchmark the taskflow under different types 
// of workloads.

#include <taskflow/taskflow.hpp>
#include <chrono>
#include <random>
#include <climits>
  
using tf_simple_t      = tf::BasicTaskflow<std::function, tf::SimpleThreadpool>;
using tf_proactive_t   = tf::BasicTaskflow<std::function, tf::ProactiveThreadpool>;
using tf_speculative_t = tf::BasicTaskflow<std::function, tf::SpeculativeThreadpool>;
using tf_privatized_t  = tf::BasicTaskflow<std::function, tf::PrivatizedThreadpool>;

// Procedure: benchmark
#define BENCHMARK(TITLE, F)                                             \
  std::cout << "========== " << TITLE << " ==========\n";               \
                                                                        \
  std::cout << "Taskflow [simple + std::func] elapsed time     : "      \
            << F<tf_simple_t>() << " ms\n";                             \
                                                                        \
  std::cout << "Taskflow [proactive + std::func] elapsed time  : "      \
            << F<tf_proactive_t>() << " ms\n";                          \
                                                                        \
  std::cout << "Taskflow [speculative + std::func] elapsed time: "      \
            << F<tf_speculative_t>() << " ms\n";                        \
                                                                        \
  std::cout << "Taskflow [privatized + std::func] elapsed time : "      \
            << F<tf_privatized_t>() << " ms\n";                         \

// ============================================================================
// Binary Tree
// ============================================================================

// Function: binary_tree
template <typename T>
auto binary_tree() {
  
  const int num_levels = 21;

  auto beg = std::chrono::high_resolution_clock::now();
  
  T tf;
  
  std::atomic<size_t> sum {0};
  std::function<void(int, typename T::TaskType)> insert;
  
  insert = [&] (int l, typename T::TaskType parent) {

    if(l < num_levels) {

      auto lc = tf.silent_emplace([&] () {
        sum.fetch_add(1, std::memory_order_relaxed);
      });

      auto rc = tf.silent_emplace([&] () {
        sum.fetch_add(1, std::memory_order_relaxed);
      });

      parent.precede(lc);
      parent.precede(rc);

      insert(l+1, lc);
      insert(l+1, rc);
    }
  };
  
  auto root = tf.silent_emplace([&] () {
    sum.fetch_add(1, std::memory_order_relaxed);
  });

  insert(1, root);

  // synchronize until all tasks finish
  tf.wait_for_all();

  assert(sum == (1 << (num_levels)) - 1);

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// ============================================================================
// Empty Jobs
// ============================================================================

// Function: empty_jobs
template <typename T>
auto empty_jobs() {
  
  const int num_tasks = 1000000;
  
  auto beg = std::chrono::high_resolution_clock::now();

  T tf;

  for(size_t i=0; i<num_tasks; i++){
    tf.silent_emplace([](){}); 
  }

  tf.wait_for_all();
  
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
  
  std::atomic<int> counter(0);
  auto beg = std::chrono::high_resolution_clock::now();
  
  T tf;
  for(size_t i=0; i<num_tasks; i++){
    tf.silent_emplace([&counter](){ 
      counter.fetch_add(1, std::memory_order_relaxed);
    }); 
  }
  tf.wait_for_all();

  assert(counter == num_tasks);
  
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// ----------------------------------------------------------------------------

// Function: main
int main(int argc, char* argv[]) {

  BENCHMARK("Empty Jobs", empty_jobs);
  BENCHMARK("Atomic Add", atomic_add);
  BENCHMARK("Binary Tree", binary_tree);
  
  return 0;
}



