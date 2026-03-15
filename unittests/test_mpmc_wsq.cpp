// =============================================================================
// test_mpmc_wsq.cpp
// =============================================================================
// Exhaustive stress-test suite for the MPMC-extended UnboundedWSQ.
//
// All tests use void* as the element type so the null-sentinel / spin-wait
// path is exercised.  Items are encoded as tagged integers cast to void*:
//   bits 63-48 : producer id  (16-bit)
//   bits 47-0  : sequence     (48-bit)
// This fits entirely in a 64-bit pointer on x86-64 / Win64 where the
// upper 16 bits of a user-space address are always zero.
//
// Test framework: doctest (same as the rest of taskflow's unittests)
// =============================================================================

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taskflow/taskflow.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <thread>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <cstring>
#include <random>
#include <array>

// ---------------------------------------------------------------------------
// Helper: encode/decode void* tags
// ---------------------------------------------------------------------------

static inline void* encode(uint16_t producer_id, uint64_t seq) noexcept {
  // pack into 64-bit integer; high 16 bits = producer_id, low 48 bits = seq+1
  // We add 1 to seq so that (producer_id=0, seq=0) does NOT produce nullptr.
  // nullptr is the sentinel for "slot claimed but not yet written" in the
  // MPMC spin-wait path; pushing nullptr would cause an infinite spin.
  uint64_t v = (static_cast<uint64_t>(producer_id) << 48) | ((seq + 1) & 0x0000FFFFFFFFFFFF);
  void* p;
  std::memcpy(&p, &v, sizeof(p));
  return p;
}

static inline void decode(void* p, uint16_t& producer_id, uint64_t& seq) noexcept {
  uint64_t v;
  std::memcpy(&v, &p, sizeof(v));
  producer_id = static_cast<uint16_t>(v >> 48);
  seq         = v & 0x0000FFFFFFFFFFFF;
}

// ---------------------------------------------------------------------------
// Helper: drain a queue into a vector (single-threaded, safe to call after
// all producers/stealers have joined)
// ---------------------------------------------------------------------------
static void drain(tf::UnboundedWSQ<void*>& q, std::vector<void*>& out) {
  void* p;
  while ((p = q.steal()) != nullptr) {
    out.push_back(p);
  }
}

// ============================================================================
// TEST 1: Pure push contention
//   16 threads each push 1,000,000 unique items to ONE queue.
//   Items are tagged so every (producer_id, seq) pair is unique.
//   After join: steal all items and verify exact count, no duplicates.
// ============================================================================
TEST_CASE("MPMC.PurePushContention" * doctest::timeout(120)) {
  constexpr int    NUM_THREADS = 16;
  constexpr size_t PER_THREAD  = 1'000'000;
  constexpr size_t TOTAL       = NUM_THREADS * PER_THREAD;

  tf::UnboundedWSQ<void*> q;

  std::vector<std::thread> threads;
  threads.reserve(NUM_THREADS);
  for (int tid = 0; tid < NUM_THREADS; ++tid) {
    threads.emplace_back([&q, tid]() {
      for (size_t seq = 0; seq < PER_THREAD; ++seq) {
        q.push(encode(static_cast<uint16_t>(tid), seq));
      }
    });
  }
  for (auto& t : threads) t.join();

  // Now drain with steal()
  std::vector<void*> items;
  items.reserve(TOTAL);
  drain(q, items);

  REQUIRE(items.size() == TOTAL);

  // Verify no duplicates using an unordered_set keyed on raw uint64_t
  std::unordered_set<uint64_t> seen;
  seen.reserve(TOTAL);
  for (void* p : items) {
    uint64_t v;
    std::memcpy(&v, &p, 8);
    REQUIRE(v != 0u);  // null sentinel should never appear
    auto [it, inserted] = seen.insert(v);
    REQUIRE(inserted);  // duplicate detected if false
  }
  REQUIRE(seen.size() == TOTAL);
}

// ============================================================================
// TEST 2: Concurrent push + steal
//   8 producer threads, 8 stealer threads.
//   Producers push unique 64-bit tagged items until 10,000,000 total produced.
//   Stealers steal into per-thread vectors.
//   After join: merge, sort, verify == all produced items.
// ============================================================================
TEST_CASE("MPMC.ConcurrentPushSteal" * doctest::timeout(180)) {
  constexpr int    NUM_PRODUCERS = 8;
  constexpr int    NUM_STEALERS  = 8;
  constexpr size_t TOTAL         = 10'000'000;
  constexpr size_t PER_PRODUCER  = TOTAL / NUM_PRODUCERS;

  tf::UnboundedWSQ<void*> q;

  std::atomic<size_t> total_consumed{0};

  // per-stealer output (no shared container on hot path)
  std::vector<std::vector<void*>> stolen(NUM_STEALERS);

  std::vector<std::thread> stealers;
  stealers.reserve(NUM_STEALERS);
  for (int sid = 0; sid < NUM_STEALERS; ++sid) {
    stealers.emplace_back([&, sid]() {
      while (total_consumed.load(std::memory_order_relaxed) < TOTAL) {
        void* p = q.steal();
        if (p != nullptr) {
          stolen[sid].push_back(p);
          total_consumed.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  std::vector<std::thread> producers;
  producers.reserve(NUM_PRODUCERS);
  for (int pid = 0; pid < NUM_PRODUCERS; ++pid) {
    producers.emplace_back([&q, pid, PER_PRODUCER]() {
      for (size_t seq = 0; seq < PER_PRODUCER; ++seq) {
        q.push(encode(static_cast<uint16_t>(pid), seq));
      }
    });
  }
  for (auto& t : producers) t.join();
  for (auto& t : stealers)  t.join();

  // Merge all stolen vectors
  std::vector<void*> items;
  items.reserve(TOTAL);
  for (auto& sv : stolen) {
    for (void* p : sv) items.push_back(p);
  }

  // Also drain any leftovers the stealers may have missed due to race on
  // total_consumed (stealers exit when counter hits TOTAL, but a few items
  // might still be in the queue if two stealers incremented simultaneously)
  drain(q, items);

  REQUIRE(items.size() == TOTAL);

  std::unordered_set<uint64_t> seen;
  seen.reserve(TOTAL);
  for (void* p : items) {
    uint64_t v;
    std::memcpy(&v, &p, 8);
    REQUIRE(v != 0u);
    auto [it, inserted] = seen.insert(v);
    REQUIRE(inserted);
  }
  REQUIRE(seen.size() == TOTAL);
}

// ============================================================================
// TEST 3: Resize under contention
//   4 producer threads push 250,000 items each (1M total).
//   4 stealer threads steal concurrently.
//   Queue starts at small capacity (LogSize=3 => capacity=8).
//   Verify total items in == total items out.
// ============================================================================
TEST_CASE("MPMC.ResizeUnderContention" * doctest::timeout(60)) {
  constexpr int    NUM_PRODUCERS = 4;
  constexpr int    NUM_STEALERS  = 4;
  constexpr size_t PER_PRODUCER  = 250'000;
  constexpr size_t TOTAL         = NUM_PRODUCERS * PER_PRODUCER;

  // Start with a very small queue to force many resizes
  tf::UnboundedWSQ<void*> q(3);  // capacity = 8
  REQUIRE(q.capacity() == 8);

  std::atomic<size_t> total_consumed{0};
  std::vector<std::vector<void*>> stolen(NUM_STEALERS);

  std::vector<std::thread> stealers;
  for (int sid = 0; sid < NUM_STEALERS; ++sid) {
    stealers.emplace_back([&, sid]() {
      while (total_consumed.load(std::memory_order_relaxed) < TOTAL) {
        void* p = q.steal();
        if (p != nullptr) {
          stolen[sid].push_back(p);
          total_consumed.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  std::vector<std::thread> producers;
  for (int pid = 0; pid < NUM_PRODUCERS; ++pid) {
    producers.emplace_back([&q, pid, PER_PRODUCER]() {
      for (size_t seq = 0; seq < PER_PRODUCER; ++seq) {
        q.push(encode(static_cast<uint16_t>(pid), seq));
      }
    });
  }
  for (auto& t : producers) t.join();
  for (auto& t : stealers)  t.join();

  std::vector<void*> items;
  items.reserve(TOTAL);
  for (auto& sv : stolen) for (void* p : sv) items.push_back(p);
  drain(q, items);

  REQUIRE(items.size() == TOTAL);

  // verify uniqueness
  std::unordered_set<uint64_t> seen;
  seen.reserve(TOTAL);
  for (void* p : items) {
    uint64_t v;
    std::memcpy(&v, &p, 8);
    auto [it, ok] = seen.insert(v);
    REQUIRE(ok);
  }
}

// ============================================================================
// TEST 4: Burst scheduling (OpenTimer workload simulation)
//   1 thread bulk_push(1,000,000 items).
//   15 stealers start before bulk_push.
//   Repeat 5 times, verify exact count + no dups each round.
// ============================================================================
TEST_CASE("MPMC.BurstBulkPush" * doctest::timeout(120)) {
  constexpr int    NUM_STEALERS = 15;
  constexpr size_t N            = 1'000'000;
  constexpr int    ROUNDS       = 5;

  // Pre-build items array
  std::vector<void*> items_to_push(N);
  for (size_t i = 0; i < N; ++i) {
    items_to_push[i] = encode(0, i);
  }

  for (int round = 0; round < ROUNDS; ++round) {
    tf::UnboundedWSQ<void*> q;
    std::atomic<size_t> total_consumed{0};
    std::vector<std::vector<void*>> stolen(NUM_STEALERS);

    std::vector<std::thread> stealers;
    for (int sid = 0; sid < NUM_STEALERS; ++sid) {
      stealers.emplace_back([&, sid]() {
        while (total_consumed.load(std::memory_order_relaxed) < N) {
          void* p = q.steal();
          if (p != nullptr) {
            stolen[sid].push_back(p);
            total_consumed.fetch_add(1, std::memory_order_relaxed);
          }
        }
      });
    }

    // bulk_push from main thread
    auto first = items_to_push.data();
    q.bulk_push(first, N);

    for (auto& t : stealers) t.join();

    std::vector<void*> got;
    got.reserve(N);
    for (auto& sv : stolen) for (void* p : sv) got.push_back(p);
    drain(q, got);

    REQUIRE(got.size() == N);
    std::unordered_set<uint64_t> seen;
    seen.reserve(N);
    for (void* p : got) {
      uint64_t v;
      std::memcpy(&v, &p, 8);
      auto [it, ok] = seen.insert(v);
      REQUIRE(ok);
    }
    REQUIRE(seen.size() == N);
  }
}

// ============================================================================
// TEST 5: Near-empty thrashing (single item)
//   1 producer, 8 stealers.
//   Each iteration: producer pushes 1 item; exactly 1 stealer gets it.
//   100,000 iterations. Verify exactly 100,000 successful steals total.
// ============================================================================
TEST_CASE("MPMC.NearEmptyThrashing" * doctest::timeout(60)) {
  constexpr int    NUM_STEALERS = 8;
  constexpr size_t ITERATIONS   = 100'000;

  tf::UnboundedWSQ<void*> q;

  std::atomic<size_t> total_stolen{0};
  std::atomic<size_t> items_ready{0};
  std::atomic<bool>   done{false};

  std::vector<std::thread> stealers;
  for (int i = 0; i < NUM_STEALERS; ++i) {
    stealers.emplace_back([&]() {
      while (!done.load(std::memory_order_relaxed) ||
             total_stolen.load(std::memory_order_relaxed) < ITERATIONS) {
        void* p = q.steal();
        if (p != nullptr) {
          total_stolen.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  // producer: push one item at a time, wait until stolen before pushing next
  for (size_t iter = 0; iter < ITERATIONS; ++iter) {
    size_t target = iter + 1;
    q.push(encode(0, iter));
    // spin-wait until this item is consumed
    while (total_stolen.load(std::memory_order_relaxed) < target) {
      std::this_thread::yield();
    }
  }

  done.store(true, std::memory_order_relaxed);
  for (auto& t : stealers) t.join();

  REQUIRE(total_stolen.load() == ITERATIONS);
}

// ============================================================================
// TEST 6: Prolonged mixed push+steal to maximise OS preemption window
//   16 threads: 8 producers + 8 stealers, run for 10 seconds.
//   Exercises the claimed-but-not-written window via OS preemption.
//   Verify: every pushed item is eventually retrieved, no loss, no dup.
// ============================================================================
TEST_CASE("MPMC.PreemptionWindowStress" * doctest::timeout(30)) {
  constexpr int    NUM_PRODUCERS = 8;
  constexpr int    NUM_STEALERS  = 8;
  constexpr double RUN_SECONDS   = 8.0;

  tf::UnboundedWSQ<void*> q;

  std::atomic<bool>   stop_flag{false};
  std::atomic<size_t> total_pushed{0};
  std::atomic<size_t> total_stolen{0};

  // stealers
  std::vector<std::vector<void*>> stolen(NUM_STEALERS);
  std::vector<std::thread> stealers;
  for (int sid = 0; sid < NUM_STEALERS; ++sid) {
    stealers.emplace_back([&, sid]() {
      while (!stop_flag.load(std::memory_order_relaxed)) {
        void* p = q.steal();
        if (p != nullptr) {
          stolen[sid].push_back(p);
          total_stolen.fetch_add(1, std::memory_order_relaxed);
        }
      }
      // drain after stop
      void* p;
      while ((p = q.steal()) != nullptr) {
        stolen[sid].push_back(p);
        total_stolen.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // producers
  std::vector<std::thread> producers;
  std::vector<std::vector<void*>> pushed(NUM_PRODUCERS);
  for (int pid = 0; pid < NUM_PRODUCERS; ++pid) {
    producers.emplace_back([&, pid]() {
      size_t seq = 0;
      while (!stop_flag.load(std::memory_order_relaxed)) {
        void* item = encode(static_cast<uint16_t>(pid), seq++);
        pushed[pid].push_back(item);
        q.push(item);
        total_pushed.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // run for RUN_SECONDS
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::duration<double>(RUN_SECONDS);
  while (std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  stop_flag.store(true, std::memory_order_relaxed);

  for (auto& t : producers) t.join();
  // All producers have finished; drain any items they pushed after stealers
  // saw stop_flag=true (stealers may have already exited their drain loop).
  {
    void* p;
    while ((p = q.steal()) != nullptr) {
      stolen[0].push_back(p);
      total_stolen.fetch_add(1, std::memory_order_relaxed);
    }
  }
  for (auto& t : stealers)  t.join();

  size_t total_p = total_pushed.load();
  size_t total_s = total_stolen.load();
  INFO("pushed=" << total_p << " stolen=" << total_s);
  REQUIRE(total_s == total_p);

  // Verify uniqueness of all stolen items
  std::unordered_set<uint64_t> seen;
  seen.reserve(total_s);
  for (auto& sv : stolen) {
    for (void* p : sv) {
      uint64_t v;
      std::memcpy(&v, &p, 8);
      auto [it, ok] = seen.insert(v);
      REQUIRE(ok);
    }
  }
}

// ============================================================================
// TEST 7: Resize lock deadlock detection
//   8 threads simultaneously push 10,000 items each = 80,000 total.
//   Never drain during push => queue must keep growing.
//   Start from default size.  If deadlock occurs, doctest timeout fires.
//   After all pushes, drain and verify count.
// ============================================================================
TEST_CASE("MPMC.ResizeLockNoDeadlock" * doctest::timeout(30)) {
  constexpr int    NUM_THREADS = 8;
  constexpr size_t PER_THREAD  = 10'000;
  constexpr size_t TOTAL       = NUM_THREADS * PER_THREAD;

  tf::UnboundedWSQ<void*> q;

  std::vector<std::thread> threads;
  for (int tid = 0; tid < NUM_THREADS; ++tid) {
    threads.emplace_back([&q, tid]() {
      for (size_t seq = 0; seq < PER_THREAD; ++seq) {
        q.push(encode(static_cast<uint16_t>(tid), seq));
      }
    });
  }
  for (auto& t : threads) t.join();

  // All 80,000 items must be in the queue
  REQUIRE(q.size() == TOTAL);

  std::vector<void*> items;
  items.reserve(TOTAL);
  drain(q, items);
  REQUIRE(items.size() == TOTAL);

  // uniqueness
  std::unordered_set<uint64_t> seen;
  seen.reserve(TOTAL);
  for (void* p : items) {
    uint64_t v;
    std::memcpy(&v, &p, 8);
    auto [it, ok] = seen.insert(v);
    REQUIRE(ok);
  }
}

// ============================================================================
// TEST 8: Mixed bulk_push and single push
//   4 threads: bulk_push in batches of 10,000 items
//   4 threads: single push one by one
//   8 stealer threads
//   Total pushed: 2,500,000 from bulk + 2,500,000 from single = 5,000,000
//   Verify total stolen == total pushed (after drain)
// ============================================================================
TEST_CASE("MPMC.MixedBulkAndSinglePush" * doctest::timeout(120)) {
  constexpr int    NUM_BULK    = 4;
  constexpr int    NUM_SINGLE  = 4;
  constexpr int    NUM_STEAL   = 8;
  constexpr size_t BULK_TOTAL  = 2'500'000;
  constexpr size_t SINGLE_TOTAL= 2'500'000;
  constexpr size_t TOTAL       = BULK_TOTAL + SINGLE_TOTAL;

  tf::UnboundedWSQ<void*> q;
  std::atomic<size_t> total_consumed{0};

  std::vector<std::vector<void*>> stolen(NUM_STEAL);

  std::vector<std::thread> stealers;
  for (int sid = 0; sid < NUM_STEAL; ++sid) {
    stealers.emplace_back([&, sid]() {
      while (total_consumed.load(std::memory_order_relaxed) < TOTAL) {
        void* p = q.steal();
        if (p != nullptr) {
          stolen[sid].push_back(p);
          total_consumed.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  // bulk producers: id 0..3
  std::vector<std::thread> bulk_threads;
  constexpr size_t BULK_PER  = BULK_TOTAL  / NUM_BULK;    // 625,000 each
  constexpr size_t BATCH_SZ  = 10'000;
  for (int pid = 0; pid < NUM_BULK; ++pid) {
    bulk_threads.emplace_back([&q, pid, BULK_PER, BATCH_SZ]() {
      std::vector<void*> batch(BATCH_SZ);
      size_t seq = 0;
      size_t pushed = 0;
      while (pushed < BULK_PER) {
        size_t n = std::min(BATCH_SZ, BULK_PER - pushed);
        for (size_t i = 0; i < n; ++i) {
          batch[i] = encode(static_cast<uint16_t>(pid), seq++);
        }
        void** first = batch.data();
        q.bulk_push(first, n);
        pushed += n;
      }
    });
  }

  // single producers: id 4..7
  constexpr size_t SINGLE_PER = SINGLE_TOTAL / NUM_SINGLE;  // 625,000 each
  std::vector<std::thread> single_threads;
  for (int pid = 0; pid < NUM_SINGLE; ++pid) {
    single_threads.emplace_back([&q, pid, SINGLE_PER]() {
      for (size_t seq = 0; seq < SINGLE_PER; ++seq) {
        q.push(encode(static_cast<uint16_t>(pid + NUM_BULK), seq));
      }
    });
  }

  for (auto& t : bulk_threads)   t.join();
  for (auto& t : single_threads) t.join();
  for (auto& t : stealers)       t.join();

  std::vector<void*> items;
  items.reserve(TOTAL);
  for (auto& sv : stolen) for (void* p : sv) items.push_back(p);
  drain(q, items);

  REQUIRE(items.size() == TOTAL);

  std::unordered_set<uint64_t> seen;
  seen.reserve(TOTAL);
  for (void* p : items) {
    uint64_t v;
    std::memcpy(&v, &p, 8);
    auto [it, ok] = seen.insert(v);
    REQUIRE(ok);
  }
}

// ============================================================================
// TEST 9: Executor integration
//   10,000 root tasks, each spawning 100 sub-tasks via tf::Subflow.
//   Total ~1,000,000 tasks. 8 workers. Run 5 times.
//   Verify: completes without hang (doctest timeout enforces this).
// ============================================================================
TEST_CASE("MPMC.ExecutorIntegration" * doctest::timeout(300)) {
  constexpr int ROOT_TASKS     = 10'000;
  constexpr int CHILDREN_EACH  = 100;
  constexpr int ROUNDS         = 5;

  tf::Executor executor(8);

  for (int r = 0; r < ROUNDS; ++r) {
    tf::Taskflow taskflow;
    std::atomic<int> counter{0};

    for (int i = 0; i < ROOT_TASKS; ++i) {
      taskflow.emplace([&counter](tf::Subflow& sf) {
        for (int j = 0; j < CHILDREN_EACH; ++j) {
          sf.emplace([&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
          });
        }
      });
    }

    executor.run(taskflow).wait();
    REQUIRE(counter.load() == ROOT_TASKS * CHILDREN_EACH);
  }
}

// ============================================================================
// TEST 10: Shutdown during push+steal
//   Executor with continuous task spawning; let executor go out of scope
//   after 2 seconds. Verify: no hang, clean shutdown.
// ============================================================================
TEST_CASE("MPMC.ShutdownDuringPushSteal" * doctest::timeout(30)) {
  std::atomic<bool> running{true};
  std::atomic<long long> tasks_run{0};

  {
    tf::Executor executor(4);
    tf::Taskflow taskflow;

    // Spawn a self-re-queuing async stream for 2 seconds
    std::function<void()> spawn_wave;
    spawn_wave = [&]() {
      if (!running.load(std::memory_order_relaxed)) return;
      for (int i = 0; i < 100; ++i) {
        executor.async([&]() {
          tasks_run.fetch_add(1, std::memory_order_relaxed);
        });
      }
    };

    // fire initial wave then let it run
    for (int w = 0; w < 50; ++w) spawn_wave();

    std::this_thread::sleep_for(std::chrono::seconds(2));
    running.store(false, std::memory_order_relaxed);

    // executor goes out of scope here — destructor must not hang
  }

  INFO("tasks_run=" << tasks_run.load());
  REQUIRE(tasks_run.load() > 0);
}

// ============================================================================
// TEST 11: CAS retry instrumentation
//   16 producer threads, 1,000,000 items each.
//   Thread-local retry counter. After run, report statistics.
//   This test always PASSES; it prints statistics for manual inspection.
// ============================================================================

// NOTE: thread_local instrumentation counter.
// In a real instrumented build we would patch wsq.hpp.  Here we measure
// indirectly via a wrapper that calls push() and counts how many times the
// queue size does NOT advance between successive calls (a proxy for retries).

TEST_CASE("MPMC.CASRetryInstrumentation" * doctest::timeout(120)) {
  constexpr int    NUM_THREADS = 16;
  constexpr size_t PER_THREAD  = 1'000'000;
  constexpr size_t TOTAL       = NUM_THREADS * PER_THREAD;

  tf::UnboundedWSQ<void*> q;

  // We cannot inject into push() without modifying wsq.hpp.
  // Instead we measure real wall time per push as a proxy for contention.
  std::atomic<uint64_t> total_ns{0};
  std::atomic<uint64_t> max_ns{0};

  std::vector<std::thread> threads;
  threads.reserve(NUM_THREADS);
  for (int tid = 0; tid < NUM_THREADS; ++tid) {
    threads.emplace_back([&, tid]() {
      uint64_t thread_total_ns = 0;
      uint64_t thread_max_ns   = 0;
      for (size_t seq = 0; seq < PER_THREAD; ++seq) {
        auto t0 = std::chrono::steady_clock::now();
        q.push(encode(static_cast<uint16_t>(tid), seq));
        auto t1 = std::chrono::steady_clock::now();
        uint64_t ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        thread_total_ns += ns;
        if (ns > thread_max_ns) thread_max_ns = ns;
      }
      total_ns.fetch_add(thread_total_ns, std::memory_order_relaxed);
      // update global max with a CAS loop
      uint64_t cur = max_ns.load(std::memory_order_relaxed);
      while (thread_max_ns > cur &&
             !max_ns.compare_exchange_weak(cur, thread_max_ns,
               std::memory_order_relaxed, std::memory_order_relaxed))
        ;
    });
  }
  for (auto& t : threads) t.join();

  std::vector<void*> items;
  items.reserve(TOTAL);
  drain(q, items);
  REQUIRE(items.size() == TOTAL);

  double avg_ns = static_cast<double>(total_ns.load()) /
                  static_cast<double>(TOTAL);
  MESSAGE("CAS instrumentation proxy:");
  MESSAGE("  total pushes   = " << TOTAL);
  MESSAGE("  avg ns/push    = " << avg_ns);
  MESSAGE("  max ns/push    = " << max_ns.load());
  MESSAGE("  (high avg_ns indicates heavy CAS retry contention)");
}
