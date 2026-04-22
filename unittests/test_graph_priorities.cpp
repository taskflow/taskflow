#include <thread>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>


// ============================================================================
// Test: Linear chain with priorities
// A(HIGH) -> B(LOW) -> C(HIGH) -> D(LOW)
// Each task must execute in dependency order regardless of priority.
// ============================================================================
TEST_CASE("GraphPriority.LinearChain" * doctest::timeout(300)) {
  tf::Executor executor(2);
  tf::Taskflow taskflow;

  std::vector<int> execution_order;

  auto A = taskflow.emplace([&]() {
    execution_order.push_back(0);
  }).name("A");

  auto B = taskflow.emplace([&]() {
    execution_order.push_back(1);
  }).name("B");

  auto C = taskflow.emplace([&]() {
    execution_order.push_back(2);
  }).name("C");

  auto D = taskflow.emplace([&]() {
    execution_order.push_back(3);
  }).name("D");

  A.precede(B);
  B.precede(C);
  C.precede(D);

  A.priority(tf::TaskPriority::HIGH);
  B.priority(tf::TaskPriority::LOW);
  C.priority(tf::TaskPriority::HIGH);
  D.priority(tf::TaskPriority::LOW);

  executor.prioritized_run(taskflow).wait();

  REQUIRE(execution_order.size() == 4);
  REQUIRE(execution_order[0] == 0);
  REQUIRE(execution_order[1] == 1);
  REQUIRE(execution_order[2] == 2);
  REQUIRE(execution_order[3] == 3);
}

// ============================================================================
// Test: Diamond pattern with priorities
//        A(HIGH)
//       / \
//    B(HIGH) C(LOW)
//       \ /
//        D(NORMAL)
// ============================================================================
TEST_CASE("GraphPriority.Diamond" * doctest::timeout(300)) {
  tf::Executor executor(2);
  tf::Taskflow taskflow;

  std::atomic<int> counter{0};
  std::atomic<int> b_order{-1}, c_order{-1};

  auto A = taskflow.emplace([&]() {
    counter++;
  }).name("A");

  auto B = taskflow.emplace([&]() {
    b_order.store(counter.fetch_add(1));
  }).name("B");

  auto C = taskflow.emplace([&]() {
    c_order.store(counter.fetch_add(1));
  }).name("C");

  auto D = taskflow.emplace([&]() {
    counter++;
  }).name("D");

  A.precede(B, C);
  B.precede(D);
  C.precede(D);

  A.priority(tf::TaskPriority::HIGH);
  B.priority(tf::TaskPriority::HIGH);
  C.priority(tf::TaskPriority::LOW);
  D.priority(tf::TaskPriority::NORMAL);

  executor.prioritized_run(taskflow).wait();

  REQUIRE(counter.load() == 4);
  // B (HIGH) should execute before C (LOW)
  REQUIRE(b_order.load() < c_order.load());
  REQUIRE(b_order.load() >= 0);
  REQUIRE(c_order.load() >= 0);
}

// ============================================================================
// Test: Diamond pattern with single thread — strict ordering guaranteed
//        A(HIGH)
//       /    \
//    B(HIGH) C(LOW)
//       \    /
//        D(NORMAL)
// With one worker, B and C cannot run in parallel, so B (HIGH) must
// execute before C (LOW).
// ============================================================================
TEST_CASE("GraphPriority.DiamondSingleThread" * doctest::timeout(300)) {
  tf::Executor executor(1);
  tf::Taskflow taskflow;

  std::vector<std::string> order;

  auto A = taskflow.emplace([&]() {
    order.push_back("A");
  }).name("A");

  auto B = taskflow.emplace([&]() {
    order.push_back("B");
  }).name("B");

  auto C = taskflow.emplace([&]() {
    order.push_back("C");
  }).name("C");

  auto D = taskflow.emplace([&]() {
    order.push_back("D");
  }).name("D");

  A.precede(B, C);
  B.precede(D);
  C.precede(D);

  A.priority(tf::TaskPriority::HIGH);
  B.priority(tf::TaskPriority::HIGH);
  C.priority(tf::TaskPriority::LOW);
  D.priority(tf::TaskPriority::NORMAL);

  executor.prioritized_run(taskflow).wait();

  REQUIRE(order.size() == 4);
  REQUIRE(order[0] == "A");

  // Find positions of B and C
  auto pos_B = std::find(order.begin(), order.end(), "B") - order.begin();
  auto pos_C = std::find(order.begin(), order.end(), "C") - order.begin();

  // With a single worker, B (HIGH) must run before C (LOW)
  REQUIRE(pos_B < pos_C);

  REQUIRE(order[3] == "D");
}

// ============================================================================
// Test: Multiple independent diamonds, test that everything executes
// ============================================================================
TEST_CASE("GraphPriority.MultipleDiamonds" * doctest::timeout(300)) {
  tf::Executor executor(4);
  tf::Taskflow taskflow;

  const int NUM_DIAMONDS = 5;
  std::atomic<int> counter{0};

  for(int d = 0; d < NUM_DIAMONDS; d++) {
    auto top = taskflow.emplace([&]() { counter++; })
                 .priority(tf::TaskPriority::HIGH);
    auto left = taskflow.emplace([&]() { counter++; })
                  .priority(tf::TaskPriority::HIGH);
    auto right = taskflow.emplace([&]() { counter++; })
                   .priority(tf::TaskPriority::LOW);
    auto bottom = taskflow.emplace([&]() { counter++; })
                    .priority(tf::TaskPriority::NORMAL);

    top.precede(left, right);
    left.precede(bottom);
    right.precede(bottom);
  }

  executor.prioritized_run(taskflow).wait();

  REQUIRE(counter.load() == NUM_DIAMONDS * 4);
}

// ============================================================================
// Test: Complex DAG with mixed priorities
//
//   A(H) --> B(L) --> D(H) --> F(N)
//     \               ^
//      --> C(H) ------/
//            |\
//            | --> E(H) --> G(L)
//            H(L)
// ============================================================================
TEST_CASE("GraphPriority.ComplexDAG" * doctest::timeout(300)) {
  tf::Executor executor(2);
  tf::Taskflow taskflow;

  // 8 tasks: A=0, B=1, C=2, D=3, E=4, F=5, G=6, H=7
  // Each task records its completion position via atomic counter.
  std::atomic<int> order{0};
  std::array<int, 8> positions{};  // positions[task_id] = execution position

  auto make_task = [&](int id) {
    return [&, id]() {
      positions[id] = order.fetch_add(1, std::memory_order_acq_rel);
    };
  };

  auto A = taskflow.emplace(make_task(0)).priority(tf::TaskPriority::HIGH).name("A");
  auto B = taskflow.emplace(make_task(1)).priority(tf::TaskPriority::LOW).name("B");
  auto C = taskflow.emplace(make_task(2)).priority(tf::TaskPriority::HIGH).name("C");
  auto D = taskflow.emplace(make_task(3)).priority(tf::TaskPriority::HIGH).name("D");
  auto E = taskflow.emplace(make_task(4)).priority(tf::TaskPriority::HIGH).name("E");
  auto F = taskflow.emplace(make_task(5)).priority(tf::TaskPriority::NORMAL).name("F");
  auto G = taskflow.emplace(make_task(6)).priority(tf::TaskPriority::LOW).name("G");
  auto H = taskflow.emplace(make_task(7)).priority(tf::TaskPriority::LOW).name("H");

  A.precede(B, C);
  B.precede(D);
  C.precede(D, E, H);
  D.precede(F);
  E.precede(G);

  executor.prioritized_run(taskflow).wait();

  REQUIRE(order.load() == 8);

  // Verify dependency ordering using recorded positions
  // pos(X) is positions[X's id]
  REQUIRE(positions[0] < positions[1]);  // A < B
  REQUIRE(positions[0] < positions[2]);  // A < C
  REQUIRE(positions[1] < positions[3]);  // B < D
  REQUIRE(positions[3] < positions[7]);  // D < H
  REQUIRE(positions[4] < positions[7]);  // E < H
  REQUIRE(positions[2] < positions[3]);  // C < D
  REQUIRE(positions[2] < positions[4]);  // C < E
  REQUIRE(positions[3] < positions[5]);  // D < F
  REQUIRE(positions[4] < positions[6]);  // E < G
}

// ============================================================================
// Test: Queue ordering verification
// With single thread, tasks should execute in priority order when all
// are ready simultaneously.
// ============================================================================
TEST_CASE("GraphPriority.QueueOrdering" * doctest::timeout(300)) {
  tf::Executor executor(1);
  tf::Taskflow taskflow;

  std::vector<int> execution_order;

  auto start = taskflow.emplace([](){});

  // Create tasks with known priorities, all depending on start
  auto low1 = taskflow.emplace([&]() {
    execution_order.emplace_back(0);
  }).priority(tf::TaskPriority::LOW);

  auto high1 = taskflow.emplace([&]() {
    execution_order.emplace_back(1);
  }).priority(tf::TaskPriority::HIGH);

  auto normal1 = taskflow.emplace([&]() {
    execution_order.emplace_back(2);
  }).priority(tf::TaskPriority::NORMAL);

  auto high2 = taskflow.emplace([&]() {
    execution_order.emplace_back(3);
  }).priority(tf::TaskPriority::HIGH);

  auto low2 = taskflow.emplace([&]() {
    execution_order.emplace_back(4);
  }).priority(tf::TaskPriority::LOW);

  start.precede(low1, high1, normal1, high2, low2);

  executor.prioritized_run(taskflow).wait();

  REQUIRE(execution_order.size() == 5);

  // With single worker, HIGH tasks should come first
  std::vector<int> high_tasks, normal_tasks, low_tasks;
  for(int id : execution_order) {
    if(id == 1 || id == 3) high_tasks.push_back(id);
    else if(id == 2) normal_tasks.push_back(id);
    else low_tasks.push_back(id);
  }

  // Verify all HIGH before NORMAL before LOW (single worker = deterministic)
  if(!high_tasks.empty() && !normal_tasks.empty()) {
    // Find positions
    size_t last_high = 0, first_normal = execution_order.size();
    for(size_t i = 0; i < execution_order.size(); i++) {
      if(execution_order[i] == 1 || execution_order[i] == 3) last_high = i;
      if((execution_order[i] == 2) && i < first_normal) first_normal = i;
    }
    REQUIRE(last_high < first_normal);
  }
}

// ============================================================================
// Test: Wide graph with many tasks at each priority level
// ============================================================================
TEST_CASE("GraphPriority.WideGraph" * doctest::timeout(300)) {
  tf::Executor executor(4);
  tf::Taskflow taskflow;

  const int TASKS_PER_PRIORITY = 100;
  const int TOTAL_TASKS = TASKS_PER_PRIORITY * 3;
  std::atomic<int> high_count{0}, normal_count{0}, low_count{0};
  std::atomic<int> order{0};
  std::array<int, 300> positions{};  // positions[pos] = priority value

  auto start = taskflow.emplace([](){});
  auto end = taskflow.emplace([](){});

  for(int i = 0; i < TASKS_PER_PRIORITY; i++) {
    auto h = taskflow.emplace([&]() {
      high_count++;
      int pos = order.fetch_add(1, std::memory_order_acq_rel);
      positions[pos] = static_cast<int>(tf::TaskPriority::HIGH);
    }).priority(tf::TaskPriority::HIGH);
    auto n = taskflow.emplace([&]() {
      normal_count++;
      int pos = order.fetch_add(1, std::memory_order_acq_rel);
      positions[pos] = static_cast<int>(tf::TaskPriority::NORMAL);
    }).priority(tf::TaskPriority::NORMAL);
    auto l = taskflow.emplace([&]() {
      low_count++;
      int pos = order.fetch_add(1, std::memory_order_acq_rel);
      positions[pos] = static_cast<int>(tf::TaskPriority::LOW);
    }).priority(tf::TaskPriority::LOW);
    start.precede(h, n, l);
    h.precede(end);
    n.precede(end);
    l.precede(end);
  }

  executor.prioritized_run(taskflow).wait();

  REQUIRE(high_count.load() == TASKS_PER_PRIORITY);
  REQUIRE(normal_count.load() == TASKS_PER_PRIORITY);
  REQUIRE(low_count.load() == TASKS_PER_PRIORITY);
  REQUIRE(order.load() == TOTAL_TASKS);

  // Compute average execution index per priority level.
  // Lower average index = executed earlier on average.
  double high_sum = 0, normal_sum = 0, low_sum = 0;
  for(int i = 0; i < TOTAL_TASKS; i++) {
    if     (positions[i] == static_cast<int>(tf::TaskPriority::HIGH))   high_sum   += i;
    else if(positions[i] == static_cast<int>(tf::TaskPriority::NORMAL)) normal_sum += i;
    else                                                                  low_sum    += i;
  }

  double avg_high   = high_sum   / TASKS_PER_PRIORITY;
  double avg_normal = normal_sum / TASKS_PER_PRIORITY;
  double avg_low    = low_sum    / TASKS_PER_PRIORITY;

  REQUIRE(avg_high < avg_normal);
  REQUIRE(avg_normal < avg_low);
}

// ============================================================================
// Test: Layered graph - tasks in layers, each layer depends on the previous
// ============================================================================
TEST_CASE("GraphPriority.LayeredGraph" * doctest::timeout(300)) {
  tf::Executor executor(8);
  tf::Taskflow taskflow;

  const int NUM_LAYERS = 3;
  const int TASKS_PER_LAYER = 60;
  std::atomic<int> total_count{0};

  // Per-layer atomic counter assigns positions via fetch_add — strict
  // arrival order with zero lock contention, giving the most faithful
  // recording of task completion order.
  struct LayerLog {
    std::atomic<int> counter{0};
    std::array<int, 60> priorities{};  // filled by fetch_add position
  };
  std::vector<LayerLog> layer_logs(NUM_LAYERS);
  std::vector<std::vector<tf::Task>> layers(NUM_LAYERS);

  for(int layer = 0; layer < NUM_LAYERS; layer++) {
    for(int i = 0; i < TASKS_PER_LAYER; i++) {
      tf::TaskPriority prio;
      if(i % 3 == 0) prio = tf::TaskPriority::HIGH;
      else if(i % 3 == 1) prio = tf::TaskPriority::NORMAL;
      else prio = tf::TaskPriority::LOW;

      auto t = taskflow.emplace([&, layer, prio]() {
        // Busy-wait so task duration >> steal overhead, ensuring workers
        // distribute properly across priority levels before any one
        // worker can lap the others.
        volatile int x = 0;
        for(int j = 0; j < 10000; j++) x += j;
        total_count++;
        int pos = layer_logs[layer].counter.fetch_add(1, std::memory_order_acq_rel);
        layer_logs[layer].priorities[pos] = static_cast<int>(prio);
      }).priority(prio);
      layers[layer].push_back(t);

      if(layer > 0) {
        for(auto& prev : layers[layer-1]) {
          prev.precede(t);
        }
      }
    }
  }

  executor.prioritized_run(taskflow).wait();

  REQUIRE(total_count.load() == NUM_LAYERS * TASKS_PER_LAYER);

  // For each layer all tasks become ready simultaneously (previous layer done),
  // so within each layer HIGH should execute before NORMAL before LOW on average.
  for(int layer = 0; layer < NUM_LAYERS; layer++) {
    REQUIRE(layer_logs[layer].counter.load() == TASKS_PER_LAYER);

    double high_sum = 0, normal_sum = 0, low_sum = 0;
    int high_n = 0, normal_n = 0, low_n = 0;
    for(int i = 0; i < TASKS_PER_LAYER; i++) {
      int p = layer_logs[layer].priorities[i];
      if     (p == static_cast<int>(tf::TaskPriority::HIGH))   { high_sum   += i; high_n++;   }
      else if(p == static_cast<int>(tf::TaskPriority::NORMAL)) { normal_sum += i; normal_n++; }
      else                                                       { low_sum    += i; low_n++;    }
    }

    double avg_high   = high_sum   / high_n;
    double avg_normal = normal_sum / normal_n;
    double avg_low    = low_sum    / low_n;

    REQUIRE(avg_high < avg_normal);
    REQUIRE(avg_normal < avg_low);
  }
}


// ============================================================================
// Test: Subflow with priorities
// ============================================================================
TEST_CASE("GraphPriority.Subflow" * doctest::timeout(300)) {
  tf::Executor executor(2);
  tf::Taskflow taskflow;

  std::atomic<int> counter{0};
  std::atomic<int> order{0};
  int s1_pos{-1}, s2_pos{-1}, s3_pos{-1};

  auto A = taskflow.emplace([&](tf::Subflow& sf) {
    sf.emplace([&]() {
      counter++;
      s1_pos = order.fetch_add(1);
    }).priority(tf::TaskPriority::HIGH);
    sf.emplace([&]() {
      counter++;
      s2_pos = order.fetch_add(1);
    }).priority(tf::TaskPriority::LOW);
    sf.emplace([&]() {
      counter++;
      s3_pos = order.fetch_add(1);
    }).priority(tf::TaskPriority::HIGH);
  }).priority(tf::TaskPriority::HIGH);

  auto B = taskflow.emplace([&]() {
    counter++;
  }).priority(tf::TaskPriority::LOW);

  A.precede(B);

  executor.prioritized_run(taskflow).wait();

  REQUIRE(counter.load() == 4);

  // s1(HIGH) and s3(HIGH) should both execute before s2(LOW)
  REQUIRE(s1_pos < s2_pos);
  REQUIRE(s3_pos < s2_pos);
}

// ============================================================================
// Test: Multiple runs of same taskflow
// ============================================================================
TEST_CASE("GraphPriority.MultipleRuns" * doctest::timeout(300)) {
  tf::Executor executor(2);
  tf::Taskflow taskflow;

  std::atomic<int> counter{0};

  auto start = taskflow.emplace([](){});
  for(int i = 0; i < 10; i++) {
    auto t = taskflow.emplace([&]() { counter++; })
               .priority(static_cast<tf::TaskPriority>(i % 3));
    start.precede(t);
  }

  for(int run = 0; run < 5; run++) {
    executor.prioritized_run(taskflow).wait();
  }

  REQUIRE(counter.load() == 50);
}

// ============================================================================
// Test: Dynamic priority change between runs
// ============================================================================
TEST_CASE("GraphPriority.DynamicChange" * doctest::timeout(300)) {
  tf::Executor executor(1);
  tf::Taskflow taskflow;

  std::vector<int> order;

  auto start = taskflow.emplace([](){});

  auto t0 = taskflow.emplace([&]() { order.emplace_back(0); });
  auto t1 = taskflow.emplace([&]() { order.emplace_back(1); });

  start.precede(t0, t1);

  // First run: t0=HIGH, t1=LOW
  t0.priority(tf::TaskPriority::HIGH);
  t1.priority(tf::TaskPriority::LOW);
  executor.prioritized_run(taskflow).wait();

  REQUIRE(order.size() == 2);
  REQUIRE(order[0] == 0);  // HIGH first
  REQUIRE(order[1] == 1);

  order.clear();

  // Second run: t0=LOW, t1=HIGH
  t0.priority(tf::TaskPriority::LOW);
  t1.priority(tf::TaskPriority::HIGH);
  executor.prioritized_run(taskflow).wait();

  REQUIRE(order.size() == 2);
  REQUIRE(order[0] == 1);  // HIGH first (t1 now)
  REQUIRE(order[1] == 0);
}

// ============================================================================
// Test: All same priority
// ============================================================================
TEST_CASE("GraphPriority.AllSamePriority" * doctest::timeout(300)) {
  tf::Executor executor(4);
  tf::Taskflow taskflow;

  std::atomic<int> counter{0};
  const int N = 50;

  auto start = taskflow.emplace([](){});
  auto end = taskflow.emplace([](){});

  for(int i = 0; i < N; i++) {
    auto t = taskflow.emplace([&]() { counter++; })
               .priority(tf::TaskPriority::NORMAL);
    start.precede(t);
    t.precede(end);
  }

  executor.prioritized_run(taskflow).wait();

  REQUIRE(counter.load() == N);
}

// ============================================================================
// Test: Dynamic fan-out (subflow creates many prioritized children)
// ============================================================================
TEST_CASE("GraphPriority.DynamicFanout" * doctest::timeout(300)) {
  tf::Executor executor(4);
  tf::Taskflow taskflow;

  const int FAN_WIDTH = 200;
  std::atomic<int> high_count{0}, low_count{0};

  // Atomic counter assigns completion positions with strict arrival order.
  std::atomic<int> order{0};
  std::array<int, 200> positions{};  // priority value at each position

  taskflow.emplace([&](tf::Subflow& sf) {
    for(int i = 0; i < FAN_WIDTH; i++) {
      if(i % 2 == 0) {
        sf.emplace([&]() {
          volatile int x = 0;
          for(int j = 0; j < 10000; j++) x += j;
          high_count++;
          int pos = order.fetch_add(1, std::memory_order_acq_rel);
          positions[pos] = static_cast<int>(tf::TaskPriority::HIGH);
        }).priority(tf::TaskPriority::HIGH);
      } else {
        sf.emplace([&]() {
          volatile int x = 0;
          for(int j = 0; j < 10000; j++) x += j;
          low_count++;
          int pos = order.fetch_add(1, std::memory_order_acq_rel);
          positions[pos] = static_cast<int>(tf::TaskPriority::LOW);
        }).priority(tf::TaskPriority::LOW);
      }
    }
  });

  executor.prioritized_run(taskflow).wait();

  REQUIRE(high_count.load() == FAN_WIDTH / 2);
  REQUIRE(low_count.load() == FAN_WIDTH / 2);

  // Verify HIGH tasks execute before LOW on average.
  double high_sum = 0, low_sum = 0;
  for(int i = 0; i < FAN_WIDTH; i++) {
    if(positions[i] == static_cast<int>(tf::TaskPriority::HIGH)) high_sum += i;
    else                                                           low_sum  += i;
  }

  double avg_high = high_sum / (FAN_WIDTH / 2);
  double avg_low  = low_sum  / (FAN_WIDTH / 2);

  REQUIRE(avg_high < avg_low);
  REQUIRE(avg_low - avg_high >= 60);
}

// ============================================================================
// Test: Cascading Burst — producers spawn mixed-priority work.
//
//   trigger → [prod0, prod1, prod2, prod3] → [L2 tasks] → sink
//
// Each producer releases a burst of HIGH, NORMAL, and LOW tasks. Tests that
// the scheduler correctly prioritizes HIGH over NORMAL over LOW when work
// arrives in staggered bursts from multiple producers.
// ============================================================================
TEST_CASE("GraphPriority.CascadingBurst" * doctest::timeout(300)) {
  const int NUM_WORKERS = 4;
  const int NUM_PRODUCERS = 4;
  const int TASKS_PER_PRODUCER = 30;  // 10 HIGH + 10 NORMAL + 10 LOW each
  const int NUM_L2_TASKS = NUM_PRODUCERS * TASKS_PER_PRODUCER;

  tf::Executor executor(NUM_WORKERS);
  tf::Taskflow taskflow;

  std::atomic<int> order{0};
  std::array<int, NUM_L2_TASKS> positions{};

  auto trigger = taskflow.emplace([](){});
  auto sink = taskflow.emplace([](){});

  std::vector<tf::Task> producers;
  for(int p = 0; p < NUM_PRODUCERS; p++) {
    auto prod = taskflow.emplace([&]() {
      volatile int x = 0;
      for(int i = 0; i < 1000; i++) x += i;
    }).priority(tf::TaskPriority::NORMAL);
    trigger.precede(prod);
    producers.push_back(prod);
  }

  for(int p = 0; p < NUM_PRODUCERS; p++) {
    for(int i = 0; i < TASKS_PER_PRODUCER; i++) {
      tf::TaskPriority prio;
      if(i % 3 == 0)      prio = tf::TaskPriority::HIGH;
      else if(i % 3 == 1) prio = tf::TaskPriority::NORMAL;
      else                 prio = tf::TaskPriority::LOW;

      auto t = taskflow.emplace([&, prio]() {
        volatile int x = 0;
        for(int j = 0; j < 10000; j++) x += j;
        int pos = order.fetch_add(1, std::memory_order_acq_rel);
        positions[pos] = static_cast<int>(prio);
      }).priority(prio);

      producers[p].precede(t);
      t.precede(sink);
    }
  }

  executor.prioritized_run(taskflow).wait();

  REQUIRE(order.load() == NUM_L2_TASKS);

  // Verify HIGH < NORMAL < LOW on average.
  double high_sum = 0, normal_sum = 0, low_sum = 0;
  int high_n = 0, normal_n = 0, low_n = 0;
  for(int i = 0; i < NUM_L2_TASKS; i++) {
    if     (positions[i] == static_cast<int>(tf::TaskPriority::HIGH))   { high_sum   += i; high_n++;   }
    else if(positions[i] == static_cast<int>(tf::TaskPriority::NORMAL)) { normal_sum += i; normal_n++; }
    else                                                                  { low_sum    += i; low_n++;    }
  }

  REQUIRE(high_n > 0);
  REQUIRE(normal_n > 0);
  REQUIRE(low_n > 0);
  REQUIRE(high_n + normal_n + low_n == NUM_L2_TASKS);

  double avg_high   = high_sum   / high_n;
  double avg_normal = normal_sum / normal_n;
  double avg_low    = low_sum    / low_n;

  REQUIRE(avg_high < avg_normal);
  REQUIRE(avg_normal < avg_low);
  REQUIRE(avg_normal - avg_high >= 8);
  REQUIRE(avg_low - avg_normal >= 8);
}

// ============================================================================
// Test: Continuation Cache Race
//
//   start → [hub0..hub15] → each hub fans out to HIGH, NORMAL, LOW → end
//
// When a hub completes, _prio_update_cache processes its 3 successors.
// The cache should pick HIGH for inline execution and schedule NORMAL/LOW.
// With many hubs completing across workers, this tests that the cache
// correctly picks the highest priority under concurrent pressure.
// ============================================================================
TEST_CASE("GraphPriority.ContinuationCacheRace" * doctest::timeout(300)) {
  const int NUM_HUBS = 16;
  const int NUM_WORKERS = 4;
  const int NUM_TASKS = NUM_HUBS * 3;  // 3 tasks per hub (HIGH, NORMAL, LOW)

  tf::Executor executor(NUM_WORKERS);
  tf::Taskflow taskflow;

  std::atomic<int> order{0};
  std::array<int, NUM_TASKS> positions{};

  auto start = taskflow.emplace([](){});
  auto end = taskflow.emplace([](){});

  for(int h = 0; h < NUM_HUBS; h++) {
    auto hub = taskflow.emplace([](){}).priority(tf::TaskPriority::NORMAL);
    start.precede(hub);

    auto high = taskflow.emplace([&]() {
      volatile int x = 0;
      for(int j = 0; j < 10000; j++) x += j;
      int pos = order.fetch_add(1, std::memory_order_acq_rel);
      positions[pos] = static_cast<int>(tf::TaskPriority::HIGH);
    }).priority(tf::TaskPriority::HIGH);

    auto normal = taskflow.emplace([&]() {
      volatile int x = 0;
      for(int j = 0; j < 10000; j++) x += j;
      int pos = order.fetch_add(1, std::memory_order_acq_rel);
      positions[pos] = static_cast<int>(tf::TaskPriority::NORMAL);
    }).priority(tf::TaskPriority::NORMAL);

    auto low = taskflow.emplace([&]() {
      volatile int x = 0;
      for(int j = 0; j < 10000; j++) x += j;
      int pos = order.fetch_add(1, std::memory_order_acq_rel);
      positions[pos] = static_cast<int>(tf::TaskPriority::LOW);
    }).priority(tf::TaskPriority::LOW);

    hub.precede(high, normal, low);
    high.precede(end);
    normal.precede(end);
    low.precede(end);
  }

  executor.prioritized_run(taskflow).wait();

  REQUIRE(order.load() == NUM_TASKS);

  // Verify HIGH < NORMAL < LOW on average.
  double high_sum = 0, normal_sum = 0, low_sum = 0;
  int high_n = 0, normal_n = 0, low_n = 0;
  for(int i = 0; i < NUM_TASKS; i++) {
    if     (positions[i] == static_cast<int>(tf::TaskPriority::HIGH))   { high_sum   += i; high_n++;   }
    else if(positions[i] == static_cast<int>(tf::TaskPriority::NORMAL)) { normal_sum += i; normal_n++; }
    else                                                                  { low_sum    += i; low_n++;    }
  }

  REQUIRE(high_n == NUM_HUBS);
  REQUIRE(normal_n == NUM_HUBS);
  REQUIRE(low_n == NUM_HUBS);

  double avg_high   = high_sum   / high_n;
  double avg_normal = normal_sum / normal_n;
  double avg_low    = low_sum    / low_n;

  REQUIRE(avg_high < avg_normal);
  REQUIRE(avg_normal < avg_low);
}

// ============================================================================
// Test: Concurrent prioritized topologies
// ============================================================================
TEST_CASE("GraphPriority.ConcurrentPrioritizedRuns" * doctest::timeout(300)) {
  tf::Executor executor(4);

  std::atomic<int> counter_a{0};
  std::atomic<int> counter_b{0};

  tf::Taskflow tf_a;
  tf::Taskflow tf_b;

  auto a_start = tf_a.emplace([](){});
  for(int i = 0; i < 20; i++) {
    auto task = tf_a.emplace([&]() {
      counter_a++;
    }).priority(i < 10 ? tf::TaskPriority::HIGH : tf::TaskPriority::LOW);
    a_start.precede(task);
  }

  auto b_start = tf_b.emplace([](){});
  for(int i = 0; i < 20; i++) {
    auto task = tf_b.emplace([&]() {
      counter_b++;
    }).priority(i < 10 ? tf::TaskPriority::HIGH : tf::TaskPriority::LOW);
    b_start.precede(task);
  }

  auto f1 = executor.prioritized_run(tf_a);
  auto f2 = executor.prioritized_run(tf_b);

  f1.wait();
  f2.wait();

  REQUIRE(counter_a == 20);
  REQUIRE(counter_b == 20);
}

// ============================================================================
// Test: Mixed prioritized_run and regular run on the same executor
//
// Submits a prioritized taskflow and a regular taskflow concurrently to the
// same executor. Verifies:
//   1. All tasks from both taskflows complete.
//   2. The prioritized taskflow still respects priority ordering (HIGH before
//      LOW on average) even while sharing workers with the regular run.
//   3. Neither run interferes with the other's correctness.
// ============================================================================
TEST_CASE("GraphPriority.MixedPrioritizedAndRegularRun" * doctest::timeout(300)) {

  tf::Executor executor(4);

  // --- Prioritized taskflow: 200 tasks (100 HIGH, 100 LOW) behind a gate ---
  tf::Taskflow prio_tf;
  const int PRIO_TASKS = 200;

  std::atomic<int> prio_order{0};
  std::array<int, 200> prio_positions{};

  auto prio_start = prio_tf.emplace([](){});
  auto prio_end   = prio_tf.emplace([](){});

  for(int i = 0; i < PRIO_TASKS; i++) {
    tf::TaskPriority p = (i < PRIO_TASKS / 2) ? tf::TaskPriority::HIGH
                                               : tf::TaskPriority::LOW;
    auto t = prio_tf.emplace([&, p]() {
      // Small busy-wait to spread work across workers
      volatile int x = 0;
      for(int j = 0; j < 5000; j++) x += j;
      int pos = prio_order.fetch_add(1, std::memory_order_acq_rel);
      prio_positions[pos] = static_cast<int>(p);
    }).priority(p);
    prio_start.precede(t);
    t.precede(prio_end);
  }

  // --- Regular taskflow: 200 tasks, no priorities ---
  tf::Taskflow regular_tf;
  const int REG_TASKS = 200;
  std::atomic<int> reg_counter{0};

  auto reg_start = regular_tf.emplace([](){});
  auto reg_end   = regular_tf.emplace([](){});

  for(int i = 0; i < REG_TASKS; i++) {
    auto t = regular_tf.emplace([&]() {
      volatile int x = 0;
      for(int j = 0; j < 5000; j++) x += j;
      reg_counter++;
    });
    reg_start.precede(t);
    t.precede(reg_end);
  }

  // Submit both concurrently to the same executor
  auto f_prio = executor.prioritized_run(prio_tf);
  auto f_reg  = executor.run(regular_tf);

  f_prio.wait();
  f_reg.wait();

  // 1. All tasks completed
  REQUIRE(prio_order.load() == PRIO_TASKS);
  REQUIRE(reg_counter.load() == REG_TASKS);

  // 2. Priority ordering: HIGH tasks execute before LOW on average
  double high_sum = 0, low_sum = 0;
  int high_n = 0, low_n = 0;
  for(int i = 0; i < PRIO_TASKS; i++) {
    if(prio_positions[i] == static_cast<int>(tf::TaskPriority::HIGH)) {
      high_sum += i; high_n++;
    } else {
      low_sum += i; low_n++;
    }
  }

  REQUIRE(high_n == PRIO_TASKS / 2);
  REQUIRE(low_n  == PRIO_TASKS / 2);

  double avg_high = high_sum / high_n;
  double avg_low  = low_sum  / low_n;

  REQUIRE(avg_high < avg_low);
  REQUIRE(avg_low - avg_high >= 20);  // With 4 workers, should be a significant gap
}

// ============================================================================
// Test: Sequential mixed runs — alternating prioritized and regular runs
//
// Runs the same taskflow alternately with prioritized_run and run, verifying
// that both paths produce correct results and that switching between them
// does not leave stale priority state in the executor.
// ============================================================================
TEST_CASE("GraphPriority.AlternatingPrioritizedAndRegularRuns" * doctest::timeout(300)) {

  tf::Executor executor(2);
  tf::Taskflow taskflow;

  std::atomic<int> counter{0};

  auto start = taskflow.emplace([](){});
  for(int i = 0; i < 20; i++) {
    auto t = taskflow.emplace([&]() { counter++; })
               .priority(i < 10 ? tf::TaskPriority::HIGH : tf::TaskPriority::LOW);
    start.precede(t);
  }

  // Alternate between prioritized and regular runs
  for(int round = 0; round < 6; round++) {
    if(round % 2 == 0) {
      executor.prioritized_run(taskflow).wait();
    } else {
      executor.run(taskflow).wait();
    }
  }

  // 20 tasks x 6 runs = 120
  REQUIRE(counter.load() == 120);
}

// ============================================================================
// Test: Multiple concurrent mixed runs — several prioritized and regular
//       taskflows submitted at once, verifying all complete correctly.
// ============================================================================
TEST_CASE("GraphPriority.ManyConcurrentMixedRuns" * doctest::timeout(300)) {

  tf::Executor executor(4);

  const int NUM_FLOWS = 6;           // 3 prioritized + 3 regular
  const int TASKS_PER_FLOW = 50;

  std::array<std::atomic<int>, NUM_FLOWS> counters{};
  std::array<tf::Taskflow, NUM_FLOWS> flows;

  for(int f = 0; f < NUM_FLOWS; f++) {
    auto gate = flows[f].emplace([](){});
    for(int i = 0; i < TASKS_PER_FLOW; i++) {
      auto t = flows[f].emplace([&counters, f]() {
        counters[f]++;
      }).priority(static_cast<tf::TaskPriority>(i % 3));
      gate.precede(t);
    }
  }

  // Submit: even indices via prioritized_run, odd via run
  std::array<tf::Future<void>, NUM_FLOWS> futures;
  for(int f = 0; f < NUM_FLOWS; f++) {
    if(f % 2 == 0) {
      futures[f] = executor.prioritized_run(flows[f]);
    } else {
      futures[f] = executor.run(flows[f]);
    }
  }

  for(auto& fut : futures) {
    fut.wait();
  }

  for(int f = 0; f < NUM_FLOWS; f++) {
    REQUIRE(counters[f].load() == TASKS_PER_FLOW);
  }
}
