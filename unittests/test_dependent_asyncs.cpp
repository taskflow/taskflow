#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

// ----------------------------------------------------------------------------
// embarrassing parallelism
// ----------------------------------------------------------------------------

void embarrassing_parallelism(unsigned W) {

  tf::Executor executor(W);

  std::atomic<int> counter(0);

  int N = 100000;

  for (int i = 0; i < N/2; ++i) {
    executor.silent_dependent_async(
      std::to_string(i), [&](){
        counter.fetch_add(1, std::memory_order_relaxed);
      }
    );
  }
  
  for (int i = N/2; i < N; ++i) {
    executor.dependent_async(
      std::to_string(i), [&](){
        counter.fetch_add(1, std::memory_order_relaxed);
      }
    );
  }

  executor.wait_for_all();

  REQUIRE(counter == N);
}

TEST_CASE("DependentAsync.EmbarrassingParallelism.1thread" * doctest::timeout(300)) {
  embarrassing_parallelism(1);
}

TEST_CASE("DependentAsync.EmbarrassingParallelism.2threads" * doctest::timeout(300)) {
  embarrassing_parallelism(2);
}

TEST_CASE("DependentAsync.EmbarrassingParallelism.4threads" * doctest::timeout(300)) {
  embarrassing_parallelism(4);
}

TEST_CASE("DependentAsync.EmbarrassingParallelism.8threads" * doctest::timeout(300)) {
  embarrassing_parallelism(8);
}

TEST_CASE("DependentAsync.EmbarrassingParallelism.16threads" * doctest::timeout(300)) {
  embarrassing_parallelism(16);
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
void simple_graph(unsigned W) {

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
  simple_graph(1);
}

TEST_CASE("SilentDependentAsync.SimpleGraph.2threads" * doctest::timeout(300)) {
  simple_graph(2);
}

TEST_CASE("SilentDependentAsync.SimpleGraph.4threads" * doctest::timeout(300)) {
  simple_graph(4);
}

TEST_CASE("SilentDependentAsync.SimpleGraph.8threads" * doctest::timeout(300)) {
  simple_graph(8);
}

TEST_CASE("SilentDependentAsync.SimpleGraph.16threads" * doctest::timeout(300)) {
  simple_graph(16);
}

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
void simple_graph_2(unsigned W) {

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

    auto [t3, fu3] = executor.dependent_async(
      "t3", [&](){
        results[3].data = results[2].data + id;
        return results[3].data;
      }, tasks1.begin(), tasks1.end()
    );

    auto t4 = executor.silent_dependent_async(
      "t4", [&](){
        results[4].data = results[2].data + id;
      }, tasks1.begin(), tasks1.end()
    );

    auto [t5, fu5] = executor.dependent_async(
      "t5", [&](){
        results[5].data = results[2].data + id;
        return results[5].data;
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

    auto [t7, fu7] = executor.dependent_async(
      "t7", [&](){
        results[7].data = results[3].data + results[4].data + id;
        return results[7].data;
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

    auto [t9, fu9] = executor.dependent_async(
      "t9", [&](){
        results[9].data = results[0].data + results[1].data +  
          results[2].data + results[7].data + results[8].data + id;
        return results[9].data;
      }, tasks4.begin(), tasks4.end()
    );
    

    REQUIRE(fu9.get() == results[9].data);
    
    REQUIRE(fu3.wait_for(std::chrono::microseconds(1)) == std::future_status::ready);
    REQUIRE(fu3.get() == results[3].data);
    
    REQUIRE(fu5.wait_for(std::chrono::microseconds(1)) == std::future_status::ready);
    REQUIRE(fu5.get() == results[5].data);
    
    REQUIRE(fu7.wait_for(std::chrono::microseconds(1)) == std::future_status::ready);
    REQUIRE(fu7.get() == results[7].data);

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

TEST_CASE("DependentAsync.SimpleGraph2.1thread" * doctest::timeout(300)) {
  simple_graph_2(1);
}

TEST_CASE("DependentAsync.SimpleGraph2.2threads" * doctest::timeout(300)) {
  simple_graph_2(2);
}

TEST_CASE("DependentAsync.SimpleGraph2.4threads" * doctest::timeout(300)) {
  simple_graph_2(4);
}

TEST_CASE("DependentAsync.SimpleGraph2.8threads" * doctest::timeout(300)) {
  simple_graph_2(8);
}

TEST_CASE("DependentAsync.SimpleGraph2.16threads" * doctest::timeout(300)) {
  simple_graph_2(16);
}

// -------------------------------------------------------------------------------------
// Complex Graph
// -------------------------------------------------------------------------------------
//
// task graph
//                 ---> 101 ----
//                 |     .     |
//      ---> 1 ---->     .     ---> 10101 ---
//     |     .     |     .     |      .     |
//     |     .     ---> 200 ----      .     |
//     |     .           .            .     |
// 0 -->     .           .            .      ---> 10201
//     |     .           .            .     |
//     |     .     ---> 10001 --      .     |
//     |     .     |     .     |      .     |
//      ---> 100 -->     .     ---> 10200 ---
//                 |     .     |
//                 ---> 10100 --
//
// level 0 : task 0 has 100 output edges pointing to task 1 to task 100
// level 1 : task 1 has 100 output edges pointing to task 101 to task 200
//           task 2 has 100 output edges pointing to task 201 to task 300
//           task 100 has 100 output edges pointing to task 10001 to task 10100
// level 2 : task 101 to task 200 has the same output edge pointing to task 10101
//           task 201 to task 300 has the same output edge pointing to task 10102
//           task 10001 to task 10100 has the same output edge pointing to from task 10200
// level 3 : task 10101 to task 10200 has the same output edge pointing to task 10201

auto make_complex_graph(tf::Executor& executor, int r) {
  
  int count = 10202;
  std::vector<tf::CachelineAligned<int>> results(count);
  std::vector<tf::AsyncTask> tasks_level_1;
  std::vector<tf::AsyncTask> tasks_level_2;
  std::vector<tf::AsyncTask> tasks_level_3;
    
  // define task 0
  auto task0 = executor.silent_dependent_async(
    "0", [&results, r](){
      results[0].data = 100 + r;
    }
  );

  // define task 1 to task 100
  // and push them in the vector tasks_level_1
  for (int i = 1; i <= 100; ++i) {
    tasks_level_1.push_back(
      executor.silent_dependent_async(
        std::to_string(i), [&results, i, r](){
          results[i].data = results[0].data + i + r;
        },
        task0
      )
    );
  }

  // define task 101 to task 10100
  // and push them in the vector tasks_level_2
  for (int i = 101; i <= 10100; ++i) {
    tasks_level_2.push_back(
      executor.silent_dependent_async(
        std::to_string(i), [&results, i, r](){
          results[i].data = results[(i-1)/100].data + i + r;
        },
        std::next(tasks_level_1.begin(), (i-1)/100-1), 
        std::next(tasks_level_1.begin(), (i-1)/100)
      )
    );
  }

  // define task 10101 to task 10200
  // and push them in the vector tasks_level_3
  for (int i = 10101; i <= 10200; ++i) {
    tasks_level_3.push_back(
      executor.silent_dependent_async(
        std::to_string(i), [&results, i, r](){
          int value = 0;
          int beg = i-10101;
          beg = (beg+1)*100+1;
          for (int j = beg; j < beg+100; ++j) {
            value += results[j].data;
          }
          results[i].data = value + i + r;
        },
        std::next(tasks_level_2.begin(),(i-10101)*100), 
        std::next(tasks_level_2.begin(), (i-10101)*100+100)
      )
    );
  }

  // define task 10201
  executor.dependent_async(
    "10201", [&results, r](){
      int value = 0;
      for (int i = 10101; i <= 10200; ++i) {
        value += results[i].data;
      }
      results[10201].data = value + 10201 + r;
      return results[10201].data;
    },
    tasks_level_3.begin(), tasks_level_3.end()
  ).second.get();
  
  // verify the result  
  for (int i = 0; i < 10202; ++i) {
    if (i == 0) {
      REQUIRE(results[i].data == 100 + r);
    }

    else if (i >= 1 && i <= 100) {
      REQUIRE(results[i].data == results[0].data + i + r);
    }

    else if (i >= 101 && i <= 10100) {
      REQUIRE(results[i].data == results[(i-1)/100].data + i + r);
    }

    else if (i >= 10101 && i <= 10200) {
      int value = 0;
      int beg = i-10101;
      beg = (beg+1)*100+1;
      for (int j = beg; j < beg+100; ++j) {
        value += results[j].data;
      }
      REQUIRE(results[i].data == value + i + r);
    }

    else if (i == 10201) {
      int value = 0;
      for (int j = 10101; j <= 10200; ++j) {
        value += results[j].data;
      }
      REQUIRE(results[i].data == value + r + 10201);
    }
  }
}

void complex_graph(unsigned W) {
  tf::Executor executor(W);
  for (int r = 0; r < 10; ++r) {
    make_complex_graph(executor, r);
  }
}

TEST_CASE("DependentAsync.ComplexGraph.1thread" * doctest::timeout(300)) {
  complex_graph(1);
}

TEST_CASE("DependentAsync.ComplexGraph.2threads" * doctest::timeout(300)) {
  complex_graph(2);
}

TEST_CASE("DependentAsync.ComplexGraph.4threads" * doctest::timeout(300)) {
  complex_graph(4);
}

TEST_CASE("DependentAsync.ComplexGraph.8threads" * doctest::timeout(300)) {
  complex_graph(8);
}

TEST_CASE("DependentAsync.ComplexGraph.16threads" * doctest::timeout(300)) {
  complex_graph(16);
}

// ----------------------------------------------------------------------------
// Complex Worker From Worker
// ----------------------------------------------------------------------------

// since make_complex_graph blocks so W must be at least one larger than R
void complex_graph_from_worker(unsigned W, int R) {
  tf::Executor executor(W);
  tf::Taskflow taskflow;
  for(int r=0; r<R; r++) {
    taskflow.emplace([&executor, r](){
      make_complex_graph(executor, r);
    });
  }
  executor.run(taskflow).wait();
}

TEST_CASE("DependentAsync.ComplexGraphFromWorker.2threads" * doctest::timeout(300)) {
  complex_graph_from_worker(2, 1);
}

TEST_CASE("DependentAsync.ComplexGraphFromWorker.4threads" * doctest::timeout(300)) {
  complex_graph_from_worker(4, 3);
}

TEST_CASE("DependentAsync.ComplexGraphFromWorker.8threads" * doctest::timeout(300)) {
  complex_graph_from_worker(8, 7);
}

TEST_CASE("DependentAsync.ComplexGraphFromWorker.16threads" * doctest::timeout(300)) {
  complex_graph_from_worker(16, 10);
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
          executor.dependent_async(
            std::to_string(n),
            [task_id, &data](){
              data[task_id] = data[task_id/2] + 1;
            },
            dep.begin(), dep.end()
          ).first
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

TEST_CASE("DependentAsync.BinaryTree.1thread" * doctest::timeout(300)) {
  binary_tree(1);
}

TEST_CASE("DependentAsync.BinaryTree.2threads" * doctest::timeout(300)) {
  binary_tree(2);
}

TEST_CASE("DependentAsync.BinaryTree.3threads" * doctest::timeout(300)) {
  binary_tree(3);
}

TEST_CASE("DependentAsync.BinaryTree.4threads" * doctest::timeout(300)) {
  binary_tree(4);
}

TEST_CASE("DependentAsync.BinaryTree.8threads" * doctest::timeout(300)) {
  binary_tree(8);
}

TEST_CASE("DependentAsync.BinaryTree.16threads" * doctest::timeout(300)) {
  binary_tree(16);
}

// ----------------------------------------------------------------------------
// Linear Chain with Complete Dependencies (N choose 2)
// ----------------------------------------------------------------------------

void complete_linear_chain(unsigned W) {

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

TEST_CASE("DependentAsync.CompleteLinearChain.1thread" * doctest::timeout(300)) {
  complete_linear_chain(1);
}

TEST_CASE("DependentAsync.CompleteLinearChain.2threads" * doctest::timeout(300)) {
  complete_linear_chain(2);
}

TEST_CASE("DependentAsync.CompleteLinearChain.4threads" * doctest::timeout(300)) {
  complete_linear_chain(4);
}

TEST_CASE("DependentAsync.CompleteLinearChain.8threads" * doctest::timeout(300)) {
  complete_linear_chain(8);
}

TEST_CASE("DependentAsync.CompleteLinearChain.16threads" * doctest::timeout(300)) {
  complete_linear_chain(16);
}

// ----------------------------------------------------------------------------
// Parallel Graph Construction
// ----------------------------------------------------------------------------

// multiple workers to construct a pascal diagram simultaneously
//  0 1 2 3  
//  |/|/| /
//  4 5 6
//  |/| /
//  7 8
//  |/
//  9

void parallel_graph_construction(unsigned W) {

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  int L = 500;
  int id = 0;
  std::vector<tf::Task> tasks((1+L)*L/2);
  std::vector<int> data((1+L)*L/2, -1);
  
  std::vector<tf::AsyncTask> async_tasks((1+L)*L/2);

  for(int l=L; l>=1; l--) {
    for(int i=0; i<l; i++) {
      
      int pr = id - l;
      int pl = id - l - 1;

      tasks[id] = taskflow.emplace([&, pr, pl, id](){
        
        if(pl >= 0 && pr >= 0) {
          REQUIRE(async_tasks[pl].empty() == false);
          REQUIRE(async_tasks[pr].empty() == false);
          async_tasks[id] = executor.silent_dependent_async([&, pr, pl, id](){
            REQUIRE(data[pr] == pr);
            REQUIRE(data[pl] == pl);
            data[id] = id;
          }, async_tasks[pl], async_tasks[pr]);
        }
        else {
          async_tasks[id] = executor.silent_dependent_async([&, id](){
            data[id] = id;
          });
        }
         
      }).name(std::to_string(id));

      if(pr >= 0) {
        tasks[id].succeed(tasks[pr]);
      }

      if(pl >= 0) {
        tasks[id].succeed(tasks[pl]);
      }
      ++id;
    }
  }

  executor.run(taskflow);
  executor.wait_for_all();
}

TEST_CASE("DependentAsync.ParallelGraphConstruction.1thread" * doctest::timeout(300)) {
  parallel_graph_construction(1);
}

TEST_CASE("DependentAsync.ParallelGraphConstruction.2threads" * doctest::timeout(300)) {
  parallel_graph_construction(2);
}

TEST_CASE("DependentAsync.ParallelGraphConstruction.4threads" * doctest::timeout(300)) {
  parallel_graph_construction(4);
}

TEST_CASE("DependentAsync.ParallelGraphConstruction.8threads" * doctest::timeout(300)) {
  parallel_graph_construction(8);
}

TEST_CASE("DependentAsync.ParallelGraphConstruction.16threads" * doctest::timeout(300)) {
  parallel_graph_construction(16);
}




