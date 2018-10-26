// 2018/10/24 - created by Tsung-Wei Huang
//
// This program demonstrates how to share the executor among different
// taskflow objects to avoid over subcription of threads.

#include <taskflow/taskflow.hpp>
#include <chrono>
#include <random>
#include <climits>

// Parameters
size_t MAX_TASKFLOW;
const size_t MAX_COUNT    = 100;
const size_t MAX_THREAD   = std::thread::hardware_concurrency();

// do some useful work
void matrix_multiplication() {

  thread_local std::random_device r;
  thread_local std::default_random_engine eng(r());
  thread_local std::uniform_int_distribution<int> dist(1, 100);

  std::vector<std::vector<int>> a (MAX_COUNT);
  std::vector<std::vector<int>> b (MAX_COUNT);
  std::vector<std::vector<int>> c (MAX_COUNT);
  
  // initialize the matrix
  for(size_t i=0; i<MAX_COUNT; ++i) {
    a[i].resize(MAX_COUNT);
    b[i].resize(MAX_COUNT);
    c[i].resize(MAX_COUNT);
    for(int j=0; j<MAX_COUNT; ++j) {
      a[i][j] = dist(eng);
      b[i][j] = dist(eng);
      c[i][j] = 0;
    }
  }

  // matrix multiplication
  for(size_t i=0; i<MAX_COUNT; ++i) {
    for(size_t j=0; j<MAX_COUNT; ++j) {
      for(size_t k=0; k<MAX_COUNT; ++k) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }

}

// Procedure
void create_task_dependency_graph(tf::Taskflow& tf) {
  for(size_t i=0; i<MAX_COUNT; ++i) {
    auto [A, B, C, D] = tf.silent_emplace(
      [&] () { matrix_multiplication(); },
      [&] () { matrix_multiplication(); },
      [&] () { matrix_multiplication(); },
      [&] () { matrix_multiplication(); }
    );
    A.precede(B);
    A.precede(C);
    C.precede(D);
    B.precede(D);
  }
}
  
// Function: unique_executor
// Each taskflow object maintains a unique executor to demonstrate
// the overhead of thread over-subcription.
auto unique_executor() {

  auto beg = std::chrono::high_resolution_clock::now();

  std::list<tf::Taskflow> tfs;

  for(size_t i=0; i<MAX_TASKFLOW; ++i) {
    auto& tf = tfs.emplace_back(MAX_THREAD);
    create_task_dependency_graph(tf);
    assert(tf.share_executor().use_count() == 2);
  }

  std::vector<std::shared_future<void>> futures;
  for(auto& tf : tfs) {
    futures.emplace_back(tf.dispatch());
  }

  for(auto& fu : futures) {
    fu.get();
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// Function: shared_executor
// The program creates an executor and share it with multiple 
// taskflow objects. There is no over-subscription in this implementation.
auto shared_executor() {

  auto beg = std::chrono::high_resolution_clock::now();

  std::list<tf::Taskflow> tfs;

  auto executor = std::make_shared<tf::Taskflow::Executor>(MAX_THREAD);

  for(size_t i=0; i<MAX_TASKFLOW; ++i) {
    assert(executor.use_count() == i + 1);
    auto& tf = tfs.emplace_back(executor);
    create_task_dependency_graph(tf);
  }

  std::vector<std::shared_future<void>> futures;
  for(auto& tf : tfs) {
    futures.emplace_back(tf.dispatch());
  }

  for(auto& fu : futures) {
    fu.get();
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
}

// ----------------------------------------------------------------------------

// Function: main
int main(int argc, char* argv[]) {

  const size_t width = 12;

  std::cout << "====================================================\n"
            << "shared: all taskflow objects share the same executor\n"
            << "unique: each taskflow objects keep a unique executor\n"
            << "====================================================\n";

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



