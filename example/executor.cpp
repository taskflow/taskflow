// 2018/10/24 - created by Tsung-Wei Huang
//
// This program demonstrates how to share the executor among different
// taskflow objects to avoid over subcription of threads.

#include <taskflow/taskflow.hpp>
#include <chrono>
#include <random>
#include <climits>

// Parameters
size_t MAX_TASKFLOW       = 1;
const size_t MAX_COUNT    = 4096;
const size_t MAX_THREAD   = std::thread::hardware_concurrency();

// Procedure
void create_quadruples(tf::Taskflow& tf, std::atomic<int>& c) {
  for(size_t i=0; i<MAX_COUNT; ++i) {
    auto [A, B, C, D] = tf.silent_emplace(
      [&] () { c.fetch_add(1, std::memory_order_relaxed); },
      [&] () { c.fetch_add(1, std::memory_order_relaxed); },
      [&] () { c.fetch_add(1, std::memory_order_relaxed); },
      [&] () { c.fetch_add(1, std::memory_order_relaxed); }
    );
    A.precede(B);
    A.precede(C);
    C.precede(D);
    B.precede(D);
  }
}
  
// Function: unique_executor
auto unique_executor() {

  auto beg = std::chrono::high_resolution_clock::now();

  std::atomic<int> counter {0};
  std::list<tf::Taskflow> tfs;

  for(size_t i=0; i<MAX_TASKFLOW; ++i) {
    auto& tf = tfs.emplace_back(MAX_THREAD);
    create_quadruples(tf, counter);
    assert(tf.share_executor().use_count() == 2);
  }

  std::vector<std::shared_future<void>> futures;
  for(auto& tf : tfs) {
    futures.emplace_back(tf.dispatch());
  }

  for(auto& fu : futures) {
    fu.get();
  }
  
  assert(counter == MAX_TASKFLOW*MAX_COUNT*4);

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// Function: shared_executor
auto shared_executor() {

  auto beg = std::chrono::high_resolution_clock::now();

  std::atomic<int> counter {0};
  std::list<tf::Taskflow> tfs;

  auto executor = std::make_shared<tf::Taskflow::Executor>(MAX_THREAD);

  for(size_t i=0; i<MAX_TASKFLOW; ++i) {
    assert(executor.use_count() == i + 1);
    auto& tf = tfs.emplace_back(executor);
    create_quadruples(tf, counter);
  }

  std::vector<std::shared_future<void>> futures;
  for(auto& tf : tfs) {
    futures.emplace_back(tf.dispatch());
  }

  for(auto& fu : futures) {
    fu.get();
  }
  
  assert(counter == MAX_TASKFLOW*MAX_COUNT*4);

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// ----------------------------------------------------------------------------

// Function: main
int main(int argc, char* argv[]) {

  const size_t width = 12;

  std::cout << std::setw(width) << "# taskflows"
            << std::setw(width) << "shared (ms)"
            << std::setw(width) << "unique (ms)"
            << std::endl;

  for(MAX_TASKFLOW=1; MAX_TASKFLOW<=128; MAX_TASKFLOW *= 2) {
    
    auto s = shared_executor();
    auto u = unique_executor();

    std::cout << std::setw(width) << MAX_TASKFLOW
              << std::setw(width) << s
              << std::setw(width) << u 
              << std::endl;
  }



  
  return 0;
}



