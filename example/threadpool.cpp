// 2018/09/17 modified by Tsung-Wei Huang
//   - added binary_tree benchmark
//   - added modulo_insertions benchmark
//
// 2018/08/31 contributed by Guannan
//
// Examples to test different threadpool implementations:
//   - SimpleThreadpool
//   - ProactiveThreadpool

#include <taskflow/threadpool/threadpool.hpp>
#include <chrono>
#include <random>
#include <climits>

// ============================================================================
// Divide and conquer to solve max subarray sum problem
// https://www.geeksforgeeks.org/divide-and-conquer-maximum-sum-subarray/
// ============================================================================

constexpr auto tree_height = 20u;
constexpr auto total_nodes = 1u << tree_height;

void update_max(std::atomic<int>& max_val, const int value) {
  int old = max_val;
  while(old < value && !max_val.compare_exchange_weak(old, value));
}

int max_cross_sum(const std::vector<int>& vec, int l, int m, int r){  
  // Include elements on left of mid. 
  auto sum = 0; 
  auto left_sum = INT_MIN; 
  for (auto i = m; i >= l; i--){ 
    sum = sum + vec[i]; 
    if (sum > left_sum) 
      left_sum = sum; 
  } 

  // Include elements on right of mid 
  sum = 0; 
  auto right_sum = INT_MIN; 
  for (auto i = m+1; i <= r; i++) 
  { 
    sum = sum + vec[i]; 
    if (sum > right_sum) 
      right_sum = sum; 
  } 

  // Return sum of elements on left and right of mid 
  return left_sum + right_sum; 
} 

template<typename T>
void max_subsum(
  const std::vector<int>& vec, 
  int l, int r, 
  std::atomic<int>& max_num, 
  T& tp, 
  std::atomic<size_t>& counter, 
  std::promise<void>& promise
) { 
  // Base Case: Only one element 
  if (l == r) {
    update_max(max_num, vec[l]);  
    if(++counter == total_nodes*2-1){
      promise.set_value();
    }
    return ;
  }

  // Find middle point 
  int m = (l + r)/2; 

  tp.silent_async([&, l=l, m=m] () { 
    max_subsum(vec, l, m, max_num, tp, counter, promise); 
  });

  tp.silent_async([&, m=m, r=r] () { 
    max_subsum(vec, m+1, r, max_num, tp, counter, promise); 
  });

  update_max(max_num, max_cross_sum(vec, l, m, r));

  if(++counter == total_nodes*2-1){
    promise.set_value();
  }
} 

template<typename T>
auto subsum(const std::vector<int>& vec){

  std::atomic<int> result {INT_MIN};
  std::atomic<size_t> counter{0};
  std::promise<void> promise;
  auto future = promise.get_future();

  auto start = std::chrono::high_resolution_clock::now();
  T tp(std::thread::hardware_concurrency());
  max_subsum(vec, 0, total_nodes-1, result, tp, counter, promise);
  future.get();
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  return elapsed.count();
}

// Procedure: benchmark_divide_and_conquer
void benchmark_divide_and_conquer(){

  std::vector<int> vec(total_nodes, 0);
  auto gen = [] () {return ::rand()%(100) + 50;};
  std::generate(std::begin(vec), std::end(vec), gen);

  std::cout << "==== Divide and Conquer ====\n";
  
  std::cout << "Privatized threadpool elapsed time : " 
            << subsum<tf::PrivatizedThreadpool>(vec) << " ms\n";
  
  std::cout << "Speculative threadpool elapsed time: " 
            << subsum<tf::SpeculativeThreadpool>(vec) << " ms\n";

  std::cout << "Proactive threadpool elapsed time  : " 
            << subsum<tf::ProactiveThreadpool>(vec) << " ms\n";

  std::cout << "Simple threadpool elapsed time     : " 
            << subsum<tf::SimpleThreadpool>(vec) << " ms\n";
}

// ============================================================================
// Binary tree computation pattern
// ============================================================================

// Function: binary_tree
template <typename T>
auto binary_tree() {
  
  const int num_threads = std::thread::hardware_concurrency();
  const int num_levels  = 20;

  auto beg = std::chrono::high_resolution_clock::now();
  
  T threadpool(num_threads);
  
  std::atomic<size_t> sum {0};
  std::function<void(int)> insert;
  
  insert = [&] (int l) {
    sum.fetch_add(1, std::memory_order_relaxed);
    if(l < num_levels) {
      for(int i=0; i<2; ++i) {
        threadpool.silent_async([&insert, l] () {
          insert(l+1);
        });
      }
    }
  };
  
  insert(0);

  // synchronize until all tasks finish
  threadpool.wait_for_all();

  assert(sum == (1 << (num_levels + 1)) - 1);

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// Procedure: benchmark_binary_tree
void benchmark_binary_tree() {

  std::cout << "==== Binary Tree ====\n";
  
  std::cout << "Privatized threadpool elapsed time : " 
            << binary_tree<tf::PrivatizedThreadpool>() << " ms\n";

  std::cout << "Speculative threadpool elapsed time: " 
            << binary_tree<tf::SpeculativeThreadpool>() << " ms\n";

  std::cout << "Proactive threadpool elapsed time  : " 
            << binary_tree<tf::ProactiveThreadpool>() << " ms\n";

  std::cout << "Simple threadpool elapsed time     : " 
            << binary_tree<tf::SimpleThreadpool>() << " ms\n";
}

// ============================================================================
// Dynamic tasking through module operations
// ============================================================================

// Procedure: modulo_insertions
template <typename T>
auto modulo_insertions() {
  
  const int num_threads = std::thread::hardware_concurrency();
  const int num_tasks   = 1000000;

  auto beg = std::chrono::high_resolution_clock::now();
  
  T threadpool(num_threads);
  
  std::atomic<size_t> sum {0};
  std::function<void(int)> insert;
  
  insert = [&threadpool, &insert, &sum] (int i) {
    if(i % 8 != 0) {
      threadpool.silent_async([&insert, i] () {
        insert(i+1);
      });
    }
    else {
      sum.fetch_add(1, std::memory_order_relaxed);
    }
  };

  for(size_t i=0; i<num_tasks; i++){
    insert(i);
  }

  // synchronize until all tasks finish
  threadpool.wait_for_all();

  assert(sum == num_tasks);

  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// Procedure: benchmark_modulo_insertions
void benchmark_modulo_insertions() {

  std::cout << "==== Modulo Insertions ====\n";
  
  std::cout << "Privatized threadpool elapsed time : " 
            << modulo_insertions<tf::PrivatizedThreadpool>() << " ms\n";

  std::cout << "Speculative threadpool elapsed time: " 
            << modulo_insertions<tf::SpeculativeThreadpool>() << " ms\n";

  std::cout << "Proactive threadpool elapsed time  : " 
            << modulo_insertions<tf::ProactiveThreadpool>() << " ms\n";

  std::cout << "Simple threadpool elapsed time     : " 
            << modulo_insertions<tf::SimpleThreadpool>() << " ms\n";
}

// ============================================================================
// Dynamic tasking through linear insertions
// ============================================================================

// Procedure: linear_insertions
template <typename T>
auto linear_insertions() {

  const int num_threads = std::thread::hardware_concurrency();
  const int num_tasks   = 2000000;
  
  auto beg = std::chrono::high_resolution_clock::now();
  
  T threadpool(num_threads);

  std::atomic<size_t> sum {0};

  std::function<void(int)> insert;
  std::promise<int> promise;
  auto future = promise.get_future();
  
  insert = [&threadpool, &insert, &sum, &promise] (int i) {
    if(i > 0) {
      threadpool.silent_async([i=i-1, &insert] () {
        insert(i);
      });
    }
    else {
      if(auto s = ++sum; s == threadpool.num_workers()) {
        promise.set_value(1);
      }
    }
  };

  for(int i=0; i<num_threads; i++){
    insert(num_tasks / num_threads);
  }

  // synchronize until all tasks finish
  threadpool.wait_for_all();
  
  assert(future.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
  assert(future.get() == 1); 
  assert(sum == num_threads);
  
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// Procedure: benchmark_linear_insertions
void benchmark_linear_insertions() {

  std::cout << "==== Linear Insertions ====\n";
  
  std::cout << "Privatized threadpool elapsed time : " 
            << linear_insertions<tf::PrivatizedThreadpool>() << " ms\n";

  std::cout << "Speculative threadpool elapsed time: " 
            << linear_insertions<tf::SpeculativeThreadpool>() << " ms\n";

  std::cout << "Proactive threadpool elapsed time  : " 
            << linear_insertions<tf::ProactiveThreadpool>() << " ms\n";

  std::cout << "Simple threadpool elapsed time     : " 
            << linear_insertions<tf::SimpleThreadpool>() << " ms\n";
}

// ============================================================================
// Insertions with empty jobs
// ============================================================================

// Function: empty_jobs
template <typename T>
auto empty_jobs() {
  
  const int num_threads = std::thread::hardware_concurrency();
  const int num_tasks   = 1000000;
  
  auto beg = std::chrono::high_resolution_clock::now();

  T threadpool(num_threads);

  for(size_t i=0; i<num_tasks; i++){
    threadpool.silent_async([](){}); 
  }

  threadpool.wait_for_all();
  
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// Procedure: benchmark_empty_jobs
void benchmark_empty_jobs() {

  std::cout << "==== Empty Jobs ====\n";
  
  std::cout << "Privatized threadpool elapsed time : " 
            << empty_jobs<tf::PrivatizedThreadpool>() << " ms\n";
  
  std::cout << "Speculative threadpool elapsed time: " 
            << empty_jobs<tf::SpeculativeThreadpool>() << " ms\n";

  std::cout << "Proactive threadpool elapsed time  : " 
            << empty_jobs<tf::ProactiveThreadpool>() << " ms\n";

  std::cout << "Simple threadpool elapsed time     : " 
            << empty_jobs<tf::SimpleThreadpool>() << " ms\n";
}

// ============================================================================
// Insertions with simple summation jobs
// ============================================================================

// Function: atomic_add
template <typename T>
auto atomic_add() {
  
  const int num_threads = std::thread::hardware_concurrency();
  const int num_tasks   = 1000000;
  
  std::atomic<int> counter(0);
  auto beg = std::chrono::high_resolution_clock::now();
  
  T threadpool(num_threads);
  for(size_t i=0; i<num_tasks; i++){
    threadpool.silent_async([&counter](){ 
      counter.fetch_add(1, std::memory_order_relaxed);
    }); 
  }
  threadpool.shutdown();

  assert(counter == num_tasks);
  
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// Procedure: benchmark_atomic_add
void benchmark_atomic_add() {

  std::cout << "==== Atomic Add ====\n";
  
  std::cout << "Privatized threadpool elapsed time : " 
            << atomic_add<tf::PrivatizedThreadpool>() << " ms\n";
  
  std::cout << "Speculative threadpool elapsed time: " 
            << atomic_add<tf::SpeculativeThreadpool>() << " ms\n";

  std::cout << "Proactive threadpool elapsed time  : " 
            << atomic_add<tf::ProactiveThreadpool>() << " ms\n";

  std::cout << "Simple threadpool elapsed time     : " 
            << atomic_add<tf::SimpleThreadpool>() << " ms\n";
}

// ----------------------------------------------------------------------------

// Function: main
int main(int argc, char* argv[]) {

  benchmark_empty_jobs();
  benchmark_atomic_add();
  benchmark_linear_insertions();
  benchmark_modulo_insertions();
  benchmark_binary_tree();
  benchmark_divide_and_conquer();
  
  return 0;
}



