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
// Simple Graph
// ----------------------------------------------------------------------------

// task dependence :
//
//    |--> 1   |--> 4
// 0 ----> 2 -----> 5
//    |--> 3   |--> 6
//
void silent_dependent_async_simple_graph(unsigned W) {

  tf::Executor executor(W);

  size_t count = 7;

  std::vector<int> results;
  std::vector<tf::AsyncTask> tasks;

  for (int id = 0; id < 100; ++id) {

    results.resize(count);

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

    auto t4 = executor.silent_dependent_async(
      "t4", [&](){
        results[4] = results[2] + 66 + id;
      }, t2
    );

    auto t5 = executor.silent_dependent_async(
      "t5", [&](){
        results[5] = results[2] - 999 + id;
      }, t2
    );

    auto t6 = executor.silent_dependent_async(
      "t6", [&](){
        results[6] = results[2] * 9 / 13 + id;
      }, t2
    );

    executor.wait_for_all();

    for (size_t i = 0; i < count; ++i) {
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
  }
}

TEST_CASE("SilentDependentAsync.SimpleGraph.1thread" * doctest::timeout(300)) {
  silent_dependent_async_simple_graph(1);
}

TEST_CASE("SilentDependentAsync.SimpleGraph.2threads" * doctest::timeout(300)) {
  silent_dependent_async_simple_graph(2);
}

TEST_CASE("SilentDependentAsync.SimpleGraph.4threads" * doctest::timeout(300)) {
  silent_dependent_async_simple_graph(4);
}

TEST_CASE("SilentDependentAsync.SimpleGraph.8threads" * doctest::timeout(300)) {
  silent_dependent_async_simple_graph(8);
}

TEST_CASE("SilentDependentAsync.SimpleGraph.16threads" * doctest::timeout(300)) {
  silent_dependent_async_simple_graph(16);
}

// ----------------------------------------------------------------------------
// Simple Graph 2
// ----------------------------------------------------------------------------

// --------------------------------------------------------------------
// Graph 2
// --------------------------------------------------------------------
//
// task dependence :
//        ----------------------------
//       |        |--> 3 --|          |
//       |        |         --> 7 --->|
//  0 ---|        |--> 4 --|          |
//       v        ^                   v
//        --> 2 --| ---------------------> 9
//       ^        v                   ^
//  1 ---|        |--> 5 --|          |
//       |        |         --> 8 --->|
//       |        |--> 6 --|          |
//       -----------------------------
//
void silent_dependent_async_simple_graph_2(unsigned W) {

  tf::Executor executor(W);

  size_t count = 10;
  std::vector<tf::CachelineAligned<int>> results(count);
  std::vector<tf::AsyncTask> tasks1;
  std::vector<tf::AsyncTask> tasks2;
  std::vector<tf::AsyncTask> tasks3;
  std::vector<tf::AsyncTask> tasks4;

  for (int id = 0; id < 100; ++id) {

    results.resize(count);

    auto t0 = executor.silent_dependent_async(
      "t0", [&](){
        results[0].data = 100 + id;
      }
    );

    auto t1 = executor.silent_dependent_async(
      "t1", [&](){
        results[1].data = 6 * id;
      }
    );
    
    auto t2 = executor.silent_dependent_async(
      "t2", [&](){
        results[2].data = results[0].data + results[1].data + id;
      }, t0, t1
    );

    tasks1.push_back(t2);

    auto t3 = executor.silent_dependent_async(
      "t3", [&](){
        results[3].data = results[2].data + id;
      }, tasks1.begin(), tasks1.end()
    );

    auto t4 = executor.silent_dependent_async(
      "t4", [&](){
        results[4].data = results[2].data + id;
      }, tasks1.begin(), tasks1.end()
    );

    auto t5 = executor.silent_dependent_async(
      "t5", [&](){
        results[5].data = results[2].data + id;
      }, tasks1.begin(), tasks1.end()
    );
    auto t6 = executor.silent_dependent_async(
      "t6", [&](){
        results[6].data = results[2].data + id;
      }, tasks1.begin(), tasks1.end()
    );

    tasks2.push_back(t3);
    tasks2.push_back(t4);
    tasks3.push_back(t5);
    tasks3.push_back(t6);

    auto t7 = executor.silent_dependent_async(
      "t7", [&](){
        results[7].data = results[3].data + results[4].data + id;
      }, tasks2.begin(), tasks2.end()
    );

    auto t8 = executor.silent_dependent_async(
      "t8", [&](){
        results[8].data = results[5].data + results[6].data + id;
      }, tasks3.begin(), tasks3.end()
    );
    
    tasks4.push_back(t0);
    tasks4.push_back(t1);
    tasks4.push_back(t2);
    tasks4.push_back(t7);
    tasks4.push_back(t8);

    auto t9 = executor.silent_dependent_async(
      "t9", [&](){
        results[9].data = results[0].data + results[1].data +  
          results[2].data + results[7].data + results[8].data + id;
      }, tasks4.begin(), tasks4.end()
    );

    executor.wait_for_all();
    for (size_t i = 0; i < count; ++i) {
      switch (i) {
        case 0:
          REQUIRE(results[i].data == 100 + id);
        break;

        case 1:
          REQUIRE(results[i].data == 6 * id);
        break;

        case 2:
          REQUIRE(results[i].data == results[0].data + results[1].data + id);
        break;

        case 3:
          REQUIRE(results[i].data == results[2].data + id);
        break;

        case 4:
          REQUIRE(results[i].data == results[2].data+ id);
        break;

        case 5:
          REQUIRE(results[i].data == results[2].data + id);
        break;

        case 6:
          REQUIRE(results[i].data == results[2].data + id);
        break;
        
        case 7:
          REQUIRE(results[i].data == results[3].data + results[4].data + id);
        break;

        case 8:
          REQUIRE(results[i].data == results[5].data + results[5].data + id);
        break;

        case 9:
          REQUIRE(results[i].data == results[0].data + results[1].data + 
            results[2].data + results[7].data + results[8].data + id);
        break;
      }
    }

    results.clear();
    tasks1.clear();
    tasks2.clear();
    tasks3.clear();
    tasks4.clear();
  }
}

TEST_CASE("SilentDependentAsync.SimpleGraph2.1thread" * doctest::timeout(300)) {
  silent_dependent_async_simple_graph_2(1);
}

TEST_CASE("SilentDependentAsync.SimpleGraph2.2threads" * doctest::timeout(300)) {
  silent_dependent_async_simple_graph_2(2);
}

TEST_CASE("SilentDependentAsync.SimpleGraph2.4threads" * doctest::timeout(300)) {
  silent_dependent_async_simple_graph_2(4);
}

TEST_CASE("SilentDependentAsync.SimpleGraph2.8threads" * doctest::timeout(300)) {
  silent_dependent_async_simple_graph_2(8);
}

TEST_CASE("SilentDependentAsync.SimpleGraph2.16threads" * doctest::timeout(300)) {
  silent_dependent_async_simple_graph_2(16);
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

// ----------------------------------------------------------------------------
// Linear Chain with Complete Dependencies (N choose 2)
// ----------------------------------------------------------------------------

void silent_dependent_async_complete_linear_chain(unsigned W) {

  tf::Executor executor0(W);
  tf::Executor executor1(W);

  int N = 1000;
  std::vector<tf::CachelineAligned<int>> results(2*N);
  std::vector<tf::AsyncTask> tasks;

  // executor 0
  for (int i = 0; i < N; ++i) {
    if (i == 0) {
      auto t = executor0.silent_dependent_async(
        std::to_string(i), [&results, i](){
          results[i].data = i+1;
        }
      );
      tasks.push_back(t);
    }
    else {
      auto t = executor0.silent_dependent_async(
        std::to_string(i), [&results, i](){
          results[i].data = results[i-1].data + i;
        }, tasks.begin(), tasks.end()
      );
      tasks.push_back(t);
    }
  }

  executor0.wait_for_all();

  REQUIRE(results[0].data == 1);
  for (int i = 1; i < N; ++i) {
    REQUIRE(results[i].data == results[i-1].data + i);
  }
  
  tasks.clear();
  
  // executor 1
  for (int i = 0; i < N; ++i) {
    if (i == 0) {
      auto t = executor1.silent_dependent_async(
        std::to_string(i+N), [&results, i, N](){
          results[i+N].data = results[i-1+N].data + i;
        }
      );
      tasks.push_back(t);
    }
    else {
      auto t = executor1.silent_dependent_async(
        std::to_string(i+N), [&results, i, N](){
          results[i+N].data = results[i-1+N].data + i;
        }, tasks.begin(), tasks.end()
      );
      tasks.push_back(t);
    }
  }

  executor1.wait_for_all();

  REQUIRE(results[0+N].data == results[0+N-1].data);

  for (int i = 1; i < N; ++i) {
    REQUIRE(results[i+N].data == results[i-1+N].data + i);
  }
}

TEST_CASE("SilentDependentAsync.CompleteLinearChain.1thread" * doctest::timeout(300)) {
  silent_dependent_async_complete_linear_chain(1);
}

TEST_CASE("SilentDependentAsync.CompleteLinearChain.2threads" * doctest::timeout(300)) {
  silent_dependent_async_complete_linear_chain(2);
}

TEST_CASE("SilentDependentAsync.CompleteLinearChain.4threads" * doctest::timeout(300)) {
  silent_dependent_async_complete_linear_chain(4);
}

TEST_CASE("SilentDependentAsync.CompleteLinearChain.8threads" * doctest::timeout(300)) {
  silent_dependent_async_complete_linear_chain(8);
}

TEST_CASE("SilentDependentAsync.CompleteLinearChain.16threads" * doctest::timeout(300)) {
  silent_dependent_async_complete_linear_chain(16);
}


