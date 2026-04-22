#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

#include <atomic>
#include <vector>
#include <mutex>
#include <numeric>

// ============================================================================
// Correctness tests for prioritized_run().
//
// Each test adapts a pattern from the existing test suite (test_basics,
// test_control_flow, test_subflows, test_runtimes, test_exceptions) and
// runs it through prioritized_run() instead of run().
//
// This file is compiled twice:
//   1. test_prioritized_run          -- default (no exploit sweep)
//   2. test_prioritized_run_enforced -- with -DTF_ENFORCE_PRIORITY_EXPLOIT
// ============================================================================

// ----------------------------------------------------------------------------
// EmbarrassinglyParallel (from test_basics.cpp)
// N independent tasks, all become ready at once.
// ----------------------------------------------------------------------------

void embarrassingly_parallel(unsigned W) {
  tf::Executor executor(W);
  tf::Taskflow taskflow;

  const int N = 1024;
  std::atomic<int> counter{0};

  for(int i = 0; i < N; i++) {
    taskflow.emplace([&]() { counter++; })
      .priority(static_cast<tf::TaskPriority>(i % 3));
  }

  executor.prioritized_run(taskflow).wait();
  REQUIRE(counter.load() == N);
}

TEST_CASE("PrioritizedRun.EmbarrassinglyParallel.1thread" * doctest::timeout(300)) {
  embarrassingly_parallel(1);
}
TEST_CASE("PrioritizedRun.EmbarrassinglyParallel.2threads" * doctest::timeout(300)) {
  embarrassingly_parallel(2);
}
TEST_CASE("PrioritizedRun.EmbarrassinglyParallel.4threads" * doctest::timeout(300)) {
  embarrassingly_parallel(4);
}

// ----------------------------------------------------------------------------
// LinearChain (from test_basics.cpp)
// Sequential chain: must execute in dependency order.
// ----------------------------------------------------------------------------

void linear_chain(unsigned W) {
  tf::Executor executor(W);
  tf::Taskflow taskflow;

  const int N = 100;
  std::atomic<int> counter{0};
  std::vector<tf::Task> tasks;

  for(int i = 0; i < N; i++) {
    tasks.emplace_back(
      taskflow.emplace([&counter, i]() {
        REQUIRE(counter.load() == i);
        counter++;
      }).priority(static_cast<tf::TaskPriority>(i % 3))
    );
    if(i > 0) tasks[i-1].precede(tasks[i]);
  }

  executor.prioritized_run(taskflow).wait();
  REQUIRE(counter.load() == N);
}

TEST_CASE("PrioritizedRun.LinearChain.1thread" * doctest::timeout(300)) {
  linear_chain(1);
}
TEST_CASE("PrioritizedRun.LinearChain.2threads" * doctest::timeout(300)) {
  linear_chain(2);
}
TEST_CASE("PrioritizedRun.LinearChain.4threads" * doctest::timeout(300)) {
  linear_chain(4);
}

// ----------------------------------------------------------------------------
// Diamond (from test_basics.cpp)
//     A
//    / \
//   B   C
//    \ /
//     D
// ----------------------------------------------------------------------------

void diamond(unsigned W) {
  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::atomic<int> counter{0};

  auto A = taskflow.emplace([&]() { counter++; }).priority(tf::TaskPriority::HIGH);
  auto B = taskflow.emplace([&]() { counter++; }).priority(tf::TaskPriority::NORMAL);
  auto C = taskflow.emplace([&]() { counter++; }).priority(tf::TaskPriority::LOW);
  auto D = taskflow.emplace([&]() { counter++; }).priority(tf::TaskPriority::HIGH);

  A.precede(B, C);
  B.precede(D);
  C.precede(D);

  executor.prioritized_run(taskflow).wait();
  REQUIRE(counter.load() == 4);

  // Run again to verify reusability
  counter = 0;
  executor.prioritized_run(taskflow).wait();
  REQUIRE(counter.load() == 4);
}

TEST_CASE("PrioritizedRun.Diamond.1thread" * doctest::timeout(300)) {
  diamond(1);
}
TEST_CASE("PrioritizedRun.Diamond.2threads" * doctest::timeout(300)) {
  diamond(2);
}
TEST_CASE("PrioritizedRun.Diamond.4threads" * doctest::timeout(300)) {
  diamond(4);
}

// ----------------------------------------------------------------------------
// LoopCond (from test_control_flow.cpp)
// Condition task loops back to itself 100 times.
// A -> B --(0)--> B (loop)
//        \-(1)--> C (exit)
// ----------------------------------------------------------------------------

void loop_cond(unsigned W) {
  tf::Executor executor(W);
  tf::Taskflow taskflow;

  int counter = -1;
  int state = 0;

  auto A = taskflow.emplace([&]() { counter = 0; })
    .priority(tf::TaskPriority::HIGH);

  auto B = taskflow.emplace([&]() mutable {
    REQUIRE((++counter % 100) == (++state % 100));
    return counter < 100 ? 0 : 1;
  }).priority(tf::TaskPriority::NORMAL);

  auto C = taskflow.emplace([&]() {
    REQUIRE(counter == 100);
    counter = 0;
  }).priority(tf::TaskPriority::LOW);

  A.precede(B);
  B.precede(B, C);

  executor.prioritized_run(taskflow).wait();
  REQUIRE(counter == 0);
  REQUIRE(state == 100);
}

TEST_CASE("PrioritizedRun.LoopCond.1thread" * doctest::timeout(300)) {
  loop_cond(1);
}
TEST_CASE("PrioritizedRun.LoopCond.2threads" * doctest::timeout(300)) {
  loop_cond(2);
}
TEST_CASE("PrioritizedRun.LoopCond.4threads" * doctest::timeout(300)) {
  loop_cond(4);
}

// ----------------------------------------------------------------------------
// CyclicCond (from test_control_flow.cpp)
// Branch condition cycles through 1000 unique branches, each visited once.
// S -> A -> Branch -> [1000 branch tasks] -> A (loop back) or T (exit)
// ----------------------------------------------------------------------------

void cyclic_cond(unsigned W) {
  tf::Executor executor(W);
  tf::Taskflow taskflow;

  int num_iterations = 0;
  const int total_iteration = 1000;
  auto S = taskflow.emplace([](){}).priority(tf::TaskPriority::HIGH);
  auto A = taskflow.emplace([&](){ num_iterations++; })
    .priority(tf::TaskPriority::NORMAL);
  S.precede(A);

  int sel = 0;
  bool pass_T = false;
  std::vector<bool> pass(total_iteration, false);
  auto T = taskflow.emplace([&](){
    REQUIRE(num_iterations == total_iteration);
    pass_T = true;
  }).priority(tf::TaskPriority::LOW);

  auto branch = taskflow.emplace([&](){ return sel++; })
    .priority(tf::TaskPriority::NORMAL);
  A.precede(branch);

  for(int i = 0; i < total_iteration; i++) {
    auto t = taskflow.emplace([&, i](){
      if(num_iterations < total_iteration) {
        REQUIRE(!pass[i]);
        pass[i] = true;
        return 0;
      }
      REQUIRE(!pass[i]);
      pass[i] = true;
      return 1;
    }).priority(static_cast<tf::TaskPriority>(i % 3));
    branch.precede(t);
    t.precede(A);
    t.precede(T);
  }

  executor.prioritized_run(taskflow).wait();

  REQUIRE(pass_T);
  for(int i = 0; i < total_iteration; i++) {
    REQUIRE(pass[i]);
  }
}

TEST_CASE("PrioritizedRun.CyclicCond.1thread" * doctest::timeout(300)) {
  cyclic_cond(1);
}
TEST_CASE("PrioritizedRun.CyclicCond.2threads" * doctest::timeout(300)) {
  cyclic_cond(2);
}
TEST_CASE("PrioritizedRun.CyclicCond.4threads" * doctest::timeout(300)) {
  cyclic_cond(4);
}

// ----------------------------------------------------------------------------
// JoinedSubflow (from test_subflows.cpp)
// Nested subflows with join synchronization.
// ----------------------------------------------------------------------------

void joined_subflow(unsigned W) {
  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::atomic<int> counter{0};

  auto sf1 = taskflow.emplace([&](tf::Subflow& fb) {
    counter++;
    fb.join();
  }).priority(tf::TaskPriority::HIGH);

  auto sf2 = taskflow.emplace([&](tf::Subflow& fb) {
    counter++;
    fb.emplace([&](tf::Subflow& fb2) {
      counter++;
      fb2.emplace([&](tf::Subflow& fb3) {
        counter++;
        fb3.join();
      }).priority(tf::TaskPriority::LOW);
    }).priority(tf::TaskPriority::NORMAL);
  }).priority(tf::TaskPriority::NORMAL);

  auto sf3 = taskflow.emplace([&](tf::Subflow& fb) {
    counter++;
    auto s1 = fb.emplace([&]() { counter++; }).priority(tf::TaskPriority::HIGH);
    auto s2 = fb.emplace([&]() { counter++; }).priority(tf::TaskPriority::LOW);
    s1.precede(s2);
  }).priority(tf::TaskPriority::LOW);

  auto finish = taskflow.emplace([&]() { counter++; })
    .priority(tf::TaskPriority::HIGH);

  sf1.precede(sf2);
  sf2.precede(sf3);
  sf3.precede(finish);

  executor.prioritized_run(taskflow).wait();
  REQUIRE(counter.load() == 8);
}

TEST_CASE("PrioritizedRun.JoinedSubflow.1thread" * doctest::timeout(300)) {
  joined_subflow(1);
}
TEST_CASE("PrioritizedRun.JoinedSubflow.2threads" * doctest::timeout(300)) {
  joined_subflow(2);
}
TEST_CASE("PrioritizedRun.JoinedSubflow.4threads" * doctest::timeout(300)) {
  joined_subflow(4);
}

// ----------------------------------------------------------------------------
// NestedSubflow (from test_subflows.cpp)
// Subflow spawns children that themselves spawn subflows, 3 levels deep.
// ----------------------------------------------------------------------------

void nested_subflow(unsigned W) {
  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::atomic<int> counter{0};
  const int CHILDREN = 4;

  taskflow.emplace([&](tf::Subflow& sf1) {
    for(int i = 0; i < CHILDREN; i++) {
      sf1.emplace([&](tf::Subflow& sf2) {
        counter++;
        for(int j = 0; j < CHILDREN; j++) {
          sf2.emplace([&]() { counter++; })
            .priority(static_cast<tf::TaskPriority>(j % 3));
        }
      }).priority(static_cast<tf::TaskPriority>(i % 3));
    }
  }).priority(tf::TaskPriority::HIGH);

  executor.prioritized_run(taskflow).wait();
  // 4 level-2 subflows + 4*4=16 level-3 tasks = 20
  REQUIRE(counter.load() == CHILDREN + CHILDREN * CHILDREN);
}

TEST_CASE("PrioritizedRun.NestedSubflow.1thread" * doctest::timeout(300)) {
  nested_subflow(1);
}
TEST_CASE("PrioritizedRun.NestedSubflow.2threads" * doctest::timeout(300)) {
  nested_subflow(2);
}
TEST_CASE("PrioritizedRun.NestedSubflow.4threads" * doctest::timeout(300)) {
  nested_subflow(4);
}

// ----------------------------------------------------------------------------
// RuntimeCorun (from test_runtimes.cpp)
// Runtime task runs external graphs via executor.corun().
// ----------------------------------------------------------------------------

void runtime_corun(unsigned W) {
  tf::Executor executor(W);
  tf::Taskflow taskflow;

  const int N = 50;
  std::vector<int> results(N, 0);
  std::vector<tf::Taskflow> graphs(N);

  for(int i = 0; i < N; i++) {
    auto& g = graphs[i];
    auto A = g.emplace([&, i]() { results[i]++; });
    auto B = g.emplace([&, i]() { results[i]++; });
    auto C = g.emplace([&, i]() { results[i]++; });
    A.precede(B);
    B.precede(C);

    taskflow.emplace([&executor, &graph=graphs[i]](tf::Runtime&) {
      executor.corun(graph);
    }).priority(static_cast<tf::TaskPriority>(i % 3));
  }

  executor.prioritized_run(taskflow).wait();

  for(int i = 0; i < N; i++) {
    REQUIRE(results[i] == 3);
  }
}

// TEST_CASE("PrioritizedRun.RuntimeCorun.1thread" * doctest::timeout(300)) {
//   runtime_corun(1);
// }
// TEST_CASE("PrioritizedRun.RuntimeCorun.2threads" * doctest::timeout(300)) {
//   runtime_corun(2);
// }
// TEST_CASE("PrioritizedRun.RuntimeCorun.4threads" * doctest::timeout(300)) {
//   runtime_corun(4);
// }

// ----------------------------------------------------------------------------
// ExceptionPropagation (from test_exceptions.cpp)
// Exception in a chain must still propagate correctly through prioritized_run.
// ----------------------------------------------------------------------------

void exception_propagation(unsigned W) {
  tf::Taskflow taskflow;
  tf::Executor executor(W);

  std::atomic<size_t> counter{0};

  auto A = taskflow.emplace([&]() { counter++; }).priority(tf::TaskPriority::HIGH);
  auto B = taskflow.emplace([&]() { counter++; }).priority(tf::TaskPriority::NORMAL);
  auto C = taskflow.emplace([&]() { counter++; }).priority(tf::TaskPriority::LOW);
  auto D = taskflow.emplace([&]() { counter++; }).priority(tf::TaskPriority::HIGH);

  A.precede(B);
  B.precede(C);
  C.precede(D);

  // Normal run should succeed
  REQUIRE_NOTHROW(executor.prioritized_run(taskflow).get());
  REQUIRE(counter == 4);

  // Inject exception at C
  counter = 0;
  C.work([]() { throw std::runtime_error("priority_error"); });

  REQUIRE_THROWS_WITH_AS(
    executor.prioritized_run(taskflow).get(),
    "priority_error",
    std::runtime_error
  );
  // A and B ran before C threw
  REQUIRE(counter == 2);
}

TEST_CASE("PrioritizedRun.ExceptionPropagation.1thread" * doctest::timeout(300)) {
  exception_propagation(1);
}
TEST_CASE("PrioritizedRun.ExceptionPropagation.2threads" * doctest::timeout(300)) {
  exception_propagation(2);
}
TEST_CASE("PrioritizedRun.ExceptionPropagation.4threads" * doctest::timeout(300)) {
  exception_propagation(4);
}

// ----------------------------------------------------------------------------
// MultipleRuns (verify prioritized_run reusability)
// Same taskflow run multiple times in sequence.
// ----------------------------------------------------------------------------

void multiple_runs(unsigned W) {
  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::atomic<int> counter{0};

  auto start = taskflow.emplace([](){}).priority(tf::TaskPriority::HIGH);
  for(int i = 0; i < 20; i++) {
    auto t = taskflow.emplace([&]() { counter++; })
      .priority(static_cast<tf::TaskPriority>(i % 3));
    start.precede(t);
  }

  for(int run = 0; run < 10; run++) {
    executor.prioritized_run(taskflow).wait();
  }

  REQUIRE(counter.load() == 200);
}

TEST_CASE("PrioritizedRun.MultipleRuns.1thread" * doctest::timeout(300)) {
  multiple_runs(1);
}
TEST_CASE("PrioritizedRun.MultipleRuns.2threads" * doctest::timeout(300)) {
  multiple_runs(2);
}
TEST_CASE("PrioritizedRun.MultipleRuns.4threads" * doctest::timeout(300)) {
  multiple_runs(4);
}

// ----------------------------------------------------------------------------
// ConcurrentTopologies
// Two prioritized taskflows running concurrently on the same executor.
// ----------------------------------------------------------------------------

void concurrent_topologies(unsigned W) {
  tf::Executor executor(W);

  tf::Taskflow tf_a, tf_b;
  std::atomic<int> counter_a{0}, counter_b{0};

  for(int i = 0; i < 50; i++) {
    tf_a.emplace([&]() { counter_a++; })
      .priority(static_cast<tf::TaskPriority>(i % 3));
    tf_b.emplace([&]() { counter_b++; })
      .priority(static_cast<tf::TaskPriority>((i + 1) % 3));
  }

  auto f1 = executor.prioritized_run(tf_a);
  auto f2 = executor.prioritized_run(tf_b);
  f1.wait();
  f2.wait();

  REQUIRE(counter_a.load() == 50);
  REQUIRE(counter_b.load() == 50);
}

TEST_CASE("PrioritizedRun.ConcurrentTopologies.1thread" * doctest::timeout(300)) {
  concurrent_topologies(1);
}
TEST_CASE("PrioritizedRun.ConcurrentTopologies.2threads" * doctest::timeout(300)) {
  concurrent_topologies(2);
}
TEST_CASE("PrioritizedRun.ConcurrentTopologies.4threads" * doctest::timeout(300)) {
  concurrent_topologies(4);
}

// ----------------------------------------------------------------------------
// MixedRunAndPrioritizedRun
// Regular run() and prioritized_run() concurrently on the same executor.
// ----------------------------------------------------------------------------

void mixed_run(unsigned W) {
  tf::Executor executor(W);

  tf::Taskflow regular_tf, prio_tf;
  std::atomic<int> regular_counter{0}, prio_counter{0};

  for(int i = 0; i < 50; i++) {
    regular_tf.emplace([&]() { regular_counter++; });
    prio_tf.emplace([&]() { prio_counter++; })
      .priority(static_cast<tf::TaskPriority>(i % 3));
  }

  auto f1 = executor.run(regular_tf);
  auto f2 = executor.prioritized_run(prio_tf);
  f1.wait();
  f2.wait();

  REQUIRE(regular_counter.load() == 50);
  REQUIRE(prio_counter.load() == 50);
}

TEST_CASE("PrioritizedRun.MixedRunAndPrioritizedRun.1thread" * doctest::timeout(300)) {
  mixed_run(1);
}
TEST_CASE("PrioritizedRun.MixedRunAndPrioritizedRun.2threads" * doctest::timeout(300)) {
  mixed_run(2);
}
TEST_CASE("PrioritizedRun.MixedRunAndPrioritizedRun.4threads" * doctest::timeout(300)) {
  mixed_run(4);
}
