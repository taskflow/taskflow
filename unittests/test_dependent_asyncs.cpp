#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

// ----------------------------------------------------------------------------
// embarrassing parallelism
// ----------------------------------------------------------------------------

void silent_dependent_async_embarrassing_parallelism(unsigned W) {

  tf::Executor executor(W);

  std::atomic<int> counter(0);

  int N = 100000;

  for (int i = 0; i < N; ++i) {
    executor.silent_dependent_async(
      std::to_string(i), [&](){
        counter.fetch_add(1, std::memory_order_relaxed);
      }
    );
  }

  executor.wait_for_all();

  REQUIRE(counter == N);
}

TEST_CASE("SilentDependentAsync.EmbarrassingParallelism.1thread" * doctest::timeout(300)) {
  silent_dependent_async_embarrassing_parallelism(1);
}

TEST_CASE("SilentDependentAsync.EmbarrassingParallelism.2threads" * doctest::timeout(300)) {
  silent_dependent_async_embarrassing_parallelism(2);
}

TEST_CASE("SilentDependentAsync.EmbarrassingParallelism.4threads" * doctest::timeout(300)) {
  silent_dependent_async_embarrassing_parallelism(4);
}

TEST_CASE("SilentDependentAsync.EmbarrassingParallelism.8threads" * doctest::timeout(300)) {
  silent_dependent_async_embarrassing_parallelism(8);
}

TEST_CASE("SilentDependentAsync.EmbarrassingParallelism.16threads" * doctest::timeout(300)) {
  silent_dependent_async_embarrassing_parallelism(16);
}

// ----------------------------------------------------------------------------
// Linear Chain
// ----------------------------------------------------------------------------

void silent_dependent_async_linear_chain(unsigned W) {

  tf::Executor executor(W);

  int N = 100000;
  std::vector<tf::CachelineAligned<int>> results(N);
  std::vector<tf::AsyncTask> tasks;

  for (int i = 0; i < N; ++i) {
    if (i == 0) {
      auto t = executor.silent_dependent_async(
        std::to_string(i), [&results, i](){
          results[i].data = i+1;
        }
      );
      tasks.push_back(t);
    }
    else {
      auto t = executor.silent_dependent_async(
        std::to_string(i), [&results, i](){
          results[i].data = results[i-1].data + i;
        }, tasks.begin(), tasks.end()
      );
      tasks.clear();
      tasks.push_back(t);
    }
  }

  executor.wait_for_all();

  REQUIRE(results[0].data == 1);

  for (int i = 1; i < N; ++i) {
    REQUIRE(results[i].data == results[i-1].data + i);
  }
}

TEST_CASE("SilentDependentAsync.LinearChain.1thread" * doctest::timeout(300)) {
  silent_dependent_async_linear_chain(1);
}

TEST_CASE("SilentDependentAsync.LinearChain.2threads" * doctest::timeout(300)) {
  silent_dependent_async_linear_chain(2);
}

TEST_CASE("SilentDependentAsync.LinearChain.4threads" * doctest::timeout(300)) {
  silent_dependent_async_linear_chain(4);
}

TEST_CASE("SilentDependentAsync.LinearChain.8threads" * doctest::timeout(300)) {
  silent_dependent_async_linear_chain(8);
}

TEST_CASE("SilentDependentAsync.LinearChain.16threads" * doctest::timeout(300)) {
  silent_dependent_async_linear_chain(16);
}

// ----------------------------------------------------------------------------
// Graph
// ----------------------------------------------------------------------------

// task dependence :
//
//    |--> 1   |--> 4
// 0 ----> 2 -----> 5
//    |--> 3   |--> 6
//
void silent_dependent_async_random_graph(unsigned W) {

  tf::Executor executor(W);

  int counts = 7;
  std::vector<int> results(counts);
  std::vector<tf::AsyncTask> tasks;
  std::vector<tf::AsyncTask> tasks1;

  for (int id = 0; id < 100; ++id) {
    auto t0 = executor.silent_dependent_async(
      "t0", [&](){
        results[0] = 100 + id;
      }
    );

    tasks.push_back(t0);

    auto t1 = executor.silent_dependent_async(
      "t1", [&](){
        results[1] = results[0] * 6 + id;
      }, tasks.begin(), tasks.end()
    );

    auto t2 = executor.silent_dependent_async(
      "t2", [&](){
        results[2] = results[0] - 200 + id;
      }, tasks.begin(), tasks.end()
    );

    auto t3 = executor.silent_dependent_async(
      "t3", [&](){
        results[3] = results[0] / 9 + id;
      }, tasks.begin(), tasks.end()
    );

    tasks1.push_back(t2);

    auto t4 = executor.silent_dependent_async(
      "t4", [&](){
        results[4] = results[2] + 66 + id;
      }, tasks1.begin(), tasks1.end()
    );

    auto t5 = executor.silent_dependent_async(
      "t5", [&](){
        results[5] = results[2] - 999 + id;
      }, tasks1.begin(), tasks1.end()
    );

    auto t6 = executor.silent_dependent_async(
      "t6", [&](){
        results[6] = results[2] * 9 / 13 + id;
      }, tasks1.begin(), tasks1.end()
    );

    executor.wait_for_all();

    for (int i = 0; i < counts; ++i) {
      switch (i) {
        case 0:
          REQUIRE(results[i] == 100 + id);
        break;

        case 1:
          REQUIRE(results[i] == results[0] * 6 + id);
        break;

        case 2:
          REQUIRE(results[i] == results[0] - 200 + id);
        break;

        case 3:
          REQUIRE(results[i] == results[0] / 9 + id);
        break;

        case 4:
          REQUIRE(results[i] == results[2] + 66 + id);
        break;

        case 5:
          REQUIRE(results[i] == results[2] - 999 + id);
        break;

        case 6:
          REQUIRE(results[i] == results[2] * 9 / 13 + id);
        break;
      }
    }

    results.clear();
    tasks.clear();
    tasks1.clear();
  }
}

TEST_CASE("SilentDependentAsync.RandomGraph.1thread" * doctest::timeout(300)) {
  silent_dependent_async_random_graph(1);
}

TEST_CASE("SilentDependentAsync.RandomGraph.2threads" * doctest::timeout(300)) {
  silent_dependent_async_random_graph(2);
}

TEST_CASE("SilentDependentAsync.RandomGraph.4threads" * doctest::timeout(300)) {
  silent_dependent_async_random_graph(4);
}

TEST_CASE("SilentDependentAsync.RandomGraph.8threads" * doctest::timeout(300)) {
  silent_dependent_async_random_graph(8);
}

TEST_CASE("SilentDependentAsync.RandomGraph.16threads" * doctest::timeout(300)) {
  silent_dependent_async_random_graph(16);
}

// ----------------------------------------------------------------------------
// Binary Tree
// ----------------------------------------------------------------------------

void binary_tree(unsigned W) {

  size_t L = 16;

  tf::Executor executor(W);
  
  std::vector<int> data(1<<L, 0);

  std::vector<tf::AsyncTask> tasks_p, tasks_c;
  std::array<tf::AsyncTask, 1> dep;
  size_t task_id = 1;
  
  // iterate all other tasks level by level
  for(size_t i=0; i<L; i++) {
    for(size_t n=0; n < (1<<i); n++) {
      if(task_id == 1) {
        tasks_c.push_back(
          executor.silent_dependent_async(
            std::to_string(n),
            [task_id, &data](){
              data[task_id] = 1;
            }
          )
        );
      }
      else {
        dep[0] = tasks_p[n/2];
        tasks_c.push_back(
          executor.silent_dependent_async(
            std::to_string(n),
            [task_id, &data](){
              data[task_id] = data[task_id/2] + 1;
            },
            dep.begin(), dep.end()
          )
        );
      }
      task_id++;
    }
    tasks_p = std::move(tasks_c);
  }

  executor.wait_for_all();
  
  task_id = 1;
  for(size_t i=0; i<L; i++) {
    for(size_t n=0; n<(1<<i); n++) {
      REQUIRE(data[task_id] == i + 1);
      //printf("data[%zu]=%d\n", task_id, data[task_id]);
      task_id++;
    }
  }
}

TEST_CASE("SilentDependentAsync.IterativeBinaryTree.1thread" * doctest::timeout(300)) {
  binary_tree(1);
}

TEST_CASE("SilentDependentAsync.IterativeBinaryTree.2threads" * doctest::timeout(300)) {
  binary_tree(2);
}

TEST_CASE("SilentDependentAsync.IterativeBinaryTree.3threads" * doctest::timeout(300)) {
  binary_tree(3);
}

TEST_CASE("SilentDependentAsync.IterativeBinaryTree.4threads" * doctest::timeout(300)) {
  binary_tree(4);
}

TEST_CASE("SilentDependentAsync.IterativeBinaryTree.8threads" * doctest::timeout(300)) {
  binary_tree(8);
}

TEST_CASE("SilentDependentAsync.IterativeBinaryTree.16threads" * doctest::timeout(300)) {
  binary_tree(16);
}








