#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taskflow/taskflow.hpp>

#include <array>
#include <atomic>

// ============================================================================
// This test PASSES only when compiled with -DTF_ENFORCE_PRIORITY_EXPLOIT.
// Without the macro, it FAILS.
//
// Mechanism under test:
//   In _exploit_task(), after each invoke the worker picks its next task via:
//     - Non-enforce: _prio_pop_task()    -- only checks OWN prio_wsq queues
//     - Enforce:     _prio_exploit_task() -- sweeps ALL workers' queues globally
//
//   When Worker B holds LOW tasks locally while Worker A has HIGH tasks,
//   only the enforce sweep lets Worker B discover and steal them.
//
// Graph structure (2 workers):
//
//   LOW fan-out:  L_src --> L0, L1, ..., L13   (15 LOW tasks total)
//   HIGH tree:    H0 --> {H1, H2}              (binary tree, depth 4 = 15 tasks) only one initial source H0
//                 H1 --> {H3, H4}
//                 H2 --> {H5, H6}  ...
//
// Why 500K busy-wait?
//   At -O3, each task takes ~289us. With 15 tasks per worker, total is ~4.3ms.
//   Thread scheduling jitter (~100us) is negligible, ensuring both workers
//   overlap concurrently and the scheduling policy (not startup timing)
//   determines execution order.
//
// Expected behavior:
//   Without enforce: Worker A drains own HIGH, Worker B drains own LOW.
//                    They interleave concurrently. avg_high ~ avg_low. Gap ~ 0.
//   With enforce:    Worker B steals HIGH from Worker A before touching LOW.
//                    Both workers execute HIGH first, then LOW.
//                    avg_high << avg_low. Gap ~ 11.
// ============================================================================

static constexpr int BUSYWAIT = 500000;  // ~289us per task at -O3

TEST_CASE("GraphPriority.HighChainBeforeLowChain") {

  constexpr int NUM_LOW_LEAVES = 14;
  constexpr int NUM_LOW = NUM_LOW_LEAVES + 1;    // +1 for L_src = 15
  constexpr int HIGH_TREE_DEPTH = 4;
  constexpr int NUM_HIGH = (1 << HIGH_TREE_DEPTH) - 1;  // 15
  constexpr int TOTAL = NUM_LOW + NUM_HIGH;              // 30
  constexpr int NUM_ITERS = 10;

  double total_gap = 0;

  for(int iter = 0; iter < NUM_ITERS; iter++) {
    tf::Executor executor(2);
    tf::Taskflow taskflow;

    std::atomic<int> counter{0};
    std::array<int, TOTAL> priorities{};  // 0 = HIGH, 2 = LOW

    // ---- LOW fan-out: L_src -> {L0, L1, ..., L13} ----
    auto low_source = taskflow.emplace([&]() {
      volatile int x = 0;
      for(int j = 0; j < BUSYWAIT; j++) x += j;
      int pos = counter.fetch_add(1, std::memory_order_acq_rel);
      priorities[pos] = 2;
    });
    low_source.priority(tf::TaskPriority::LOW);
    low_source.name("L_src");

    for(int i = 0; i < NUM_LOW_LEAVES; i++) {
      auto leaf = taskflow.emplace([&]() {
        volatile int x = 0;
        for(int j = 0; j < BUSYWAIT; j++) x += j;
        int pos = counter.fetch_add(1, std::memory_order_acq_rel);
        priorities[pos] = 2;
      });
      leaf.priority(tf::TaskPriority::LOW);
      leaf.name("L" + std::to_string(i));
      low_source.precede(leaf);
    }

    // ---- HIGH binary tree (depth 4, 15 tasks) ----
    std::vector<tf::Task> high_tasks;
    for(int i = 0; i < NUM_HIGH; i++) {
      auto t = taskflow.emplace([&]() {
        volatile int x = 0;
        for(int j = 0; j < BUSYWAIT; j++) x += j;
        int pos = counter.fetch_add(1, std::memory_order_acq_rel);
        priorities[pos] = 0;
      });
      t.priority(tf::TaskPriority::HIGH);
      t.name("H" + std::to_string(i));
      high_tasks.push_back(t);
    }
    // Wire binary tree: node i -> children 2i+1, 2i+2
    for(int i = 0; i < NUM_HIGH; i++) {
      int left = 2 * i + 1;
      int right = 2 * i + 2;
      if(left < NUM_HIGH)  high_tasks[i].precede(high_tasks[left]);
      if(right < NUM_HIGH) high_tasks[i].precede(high_tasks[right]);
    }

    executor.prioritized_run(taskflow).wait();

    REQUIRE(counter.load() == TOTAL);

    // Compute average execution position of HIGH vs LOW tasks
    double sum_high = 0, sum_low = 0;
    int count_high = 0, count_low = 0;
    for(int i = 0; i < TOTAL; i++) {
      if(priorities[i] == 0) { sum_high += i; count_high++; }
      else                   { sum_low  += i; count_low++;  }
    }

    REQUIRE(count_high == NUM_HIGH);
    REQUIRE(count_low == NUM_LOW);

    double avg_high = sum_high / count_high;
    double avg_low  = sum_low  / count_low;
    double gap = avg_low - avg_high;
    total_gap += gap;
  }

  double avg_gap = total_gap / NUM_ITERS;

  // With enforce: avg_gap ~ 10-11 (HIGH consistently executes before LOW)
  // Without enforce: avg_gap ~ 0 (interleaved, each worker drains own queue)
  REQUIRE(avg_gap >= 5.0);
}
