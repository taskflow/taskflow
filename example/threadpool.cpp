// 2018/12/08 modified by Tsung-Wei Huang
//   - refactored the output format 
//
// 2018/12/07 modified by Tsung-Wei Huang
//   - refactored the output format
//
// 2018/12/06 modified by Tsung-Wei Huang
//   - added nested insertions test
//
// 2018/12/04 modified by Tsung-Wei Huang
//   - replace privatized threadpool with work stealing threadpool
//   
// 2018/10/04 modified by Tsung-Wei Huang
//   - removed binary_tree
//   - removed modulo_insertions
//   - adopted to the new threadpool implementation
//
// 2018/09/19 modified by Tsung-Wei Huang
//   - added binary_tree benchmark
//   - added modulo_insertions benchmark
//   - refactored benchmark calls
//
// 2018/08/31 contributed by Guannan
//
// Examples to test different threadpool implementations:
//   - SimpleThreadpool
//   - ProactiveThreadpool
//   - SpeculativeThreadpool
//   - PrivatizedThreadpool

#include <taskflow/threadpool/threadpool.hpp>
#include <chrono>
#include <cmath>
#include <random>
#include <numeric>
#include <climits>
#include <iomanip>

constexpr int WIDTH = 12;

using Closure = std::function<void()>;

// Procedure: benchmark
#define BENCHMARK(TITLE, F)                                                     \
std::cout                                                                       \
  << std::setw(WIDTH) << TITLE << std::flush                                    \
  << std::setw(WIDTH) << F<tf::SimpleThreadpool<Closure>>() << std::flush       \
  << std::setw(WIDTH) << F<tf::ProactiveThreadpool<Closure>>() << std::flush    \
  << std::setw(WIDTH) << F<tf::SpeculativeThreadpool<Closure>>() << std::flush  \
  << std::setw(WIDTH) << F<tf::WorkStealingThreadpool<Closure>>() << std::flush \
  << std::endl;

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

  tp.emplace([&, l=l, m=m] () { 
    max_subsum(vec, l, m, max_num, tp, counter, promise); 
  });

  tp.emplace([&, m=m, r=r] () { 
    max_subsum(vec, m+1, r, max_num, tp, counter, promise); 
  });

  update_max(max_num, max_cross_sum(vec, l, m, r));

  if(++counter == total_nodes*2-1){
    promise.set_value();
  }
} 

template<typename T>
auto subsum(){
  
  std::vector<int> vec(total_nodes);
  std::iota(vec.begin(), vec.end(), -50);

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

  std::atomic<int> sum {0};

  std::function<void(int)> insert;
  std::promise<int> promise;
  auto future = promise.get_future();
  
  insert = [&threadpool, &insert, &sum, &promise] (int i) {
    if(i > 0) {
      threadpool.emplace([i=i-1, &insert] () {
        insert(i);
      });
    }
    else {
      if(size_t s = ++sum; s == threadpool.num_workers()) {
        promise.set_value(1);
      }
    }
  };

  for(int i=0; i<num_threads; i++){
    insert(num_tasks / num_threads);
  }

  // synchronize until all tasks finish
  assert(future.get() == 1);
  assert(sum == num_threads);
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// ============================================================================
// Insertions with atomic summation
// ============================================================================

// Function: atomic_add
template <typename T>
auto atomic_add() {
  
  const int num_threads = std::thread::hardware_concurrency();
  const int num_tasks   = 1000000;
  
  std::atomic<int> counter(0);
  auto beg = std::chrono::high_resolution_clock::now();

  std::promise<void> promise;
  auto future = promise.get_future();
  
  T threadpool(num_threads);
  for(size_t i=0; i<num_tasks; i++){
    threadpool.emplace([&](){ 
      if(counter.fetch_add(1, std::memory_order_relaxed) + 1 == num_tasks) {
        promise.set_value();
      }
    }); 
  }

  future.get();

  assert(counter == num_tasks);
  
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// ============================================================================
// skewed insertions
// ============================================================================

// Function: nested_insertions
template <typename T>
auto nested_insertions() {
  
  const int num_threads = std::thread::hardware_concurrency();
  const int num_tasks = 32;
  
  auto beg = std::chrono::high_resolution_clock::now();

  std::atomic<int64_t> counter(0);
  
  std::promise<void> promise;
  auto future = promise.get_future();

  auto increment = [&] () {
    int64_t sum = 0;
    for(int i=0; i<5; ++i) {
      sum = (sum + 1)*num_tasks;
    }
    if(++counter == sum) {
      promise.set_value();
    }
  };

  T threadpool(num_threads);

  threadpool.emplace([&] () {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    for(int i=0; i<num_tasks; ++i) {
      increment();
      threadpool.emplace([&] () {
        for(int i=0; i<num_tasks; ++i) {
          increment();
          threadpool.emplace([&] () {
            for(int i=0; i<num_tasks; ++i) {
              increment();
              threadpool.emplace([&] () {
                for(int i=0; i<num_tasks; ++i) {
                  increment();
                  threadpool.emplace([&] () {
                    for(int i=0; i<num_tasks; ++i) {
                      increment();
                    }
                  });
                }
              });
            }
          });
        }
      });
    }
  });

  future.get(); 

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// ============================================================================
// batch insertion
// ============================================================================

// Function: batch_insertions
template <typename T>
auto batch_insertions() {
  
  const int num_threads = std::thread::hardware_concurrency();
  const int num_batches = 512;
  const int num_tasks   = 512;
  
  auto beg = std::chrono::high_resolution_clock::now();

  std::atomic<int> counter(0);
  std::vector<std::function<void()>> tasks (num_tasks);
  
  std::promise<void> promise;
  auto future = promise.get_future();

  for(auto & task : tasks) {
    task = [&] () {
      if(++counter == 2*num_tasks*num_batches) {
        promise.set_value();
      }
    };
  }
  
  T threadpool(num_threads);

  // master to insert a batch
  for(size_t i=0; i<num_batches; ++i) {
    auto copy = tasks;
    threadpool.batch(std::move(copy));
  }

  for(size_t i=0; i<num_batches; i++){
    threadpool.emplace([&](){ 
      auto copy = tasks;
      threadpool.batch(std::move(copy));
    }); 
  }

  future.get();
  
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// Function: main
int main(int argc, char* argv[]) {
  
  std::cout << std::setw(WIDTH) << "workload"
            << std::setw(WIDTH) << "simple"
            << std::setw(WIDTH) << "pro"
            << std::setw(WIDTH) << "spec"
            << std::setw(WIDTH) << "steal"
            << std::endl;

  BENCHMARK("Atomic", atomic_add);
  BENCHMARK("Linear", linear_insertions);
  BENCHMARK("D&Q", subsum);
  BENCHMARK("Batch", batch_insertions);
  BENCHMARK("Nested", nested_insertions);
  
  return 0;
}



