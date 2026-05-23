#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/utility/object_pool.hpp>

#include <atomic>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// TestObj
//
// Non-trivially constructible and destructible object used across all tests.
// Tracks constructor arguments and counts destructor invocations via an
// external atomic so tests can verify exact lifecycle counts.
// ============================================================================

struct TestObj {

  int               value     {0};
  std::string       label;
  std::atomic<int>* dtor_count {nullptr};

  TestObj() = default;

  TestObj(int v, std::string l, std::atomic<int>* cnt = nullptr)
    : value(v), label(std::move(l)), dtor_count(cnt) {}

  ~TestObj() {
    if(dtor_count) {
      dtor_count->fetch_add(1, std::memory_order_relaxed);
    }
  }

  TestObj(const TestObj&) = delete;
  TestObj& operator=(const TestObj&) = delete;
};

// ============================================================================
// op_animate_recycle
//
// Each of W threads animates N objects with distinct values and labels,
// verifies that constructor arguments are forwarded correctly, then recycles
// all objects. Tests basic correctness of animate/recycle under concurrency.
// ============================================================================

void op_animate_recycle(unsigned W) {

  tf::ObjectPool<TestObj> pool;

  const int N = 1024;

  std::vector<std::thread> threads;
  threads.reserve(W);

  for(unsigned w = 0; w < W; w++) {
    threads.emplace_back([&pool, w]() {
      std::vector<TestObj*> objs;
      objs.reserve(N);

      for(int i = 0; i < N; i++) {
        int         val   = static_cast<int>(w) * N + i;
        std::string label = "obj-" + std::to_string(val);
        TestObj* obj = pool.animate(val, label);
        REQUIRE(obj != nullptr);
        REQUIRE(obj->value == val);
        REQUIRE(obj->label == label);
        objs.push_back(obj);
      }

      for(auto* obj : objs) {
        pool.recycle(obj);
      }
    });
  }

  for(auto& t : threads) t.join();
}

// ============================================================================
// op_destructor
//
// Verifies that recycle invokes the destructor of T exactly once per object.
// W threads each animate N objects sharing a single atomic dtor_count.
// After all threads finish, the counter must equal W * N.
// ============================================================================

void op_destructor(unsigned W) {

  tf::ObjectPool<TestObj> pool;

  const int        N = 2048;
  std::atomic<int> dtor_count{0};

  std::vector<std::thread> threads;
  threads.reserve(W);

  for(unsigned w = 0; w < W; w++) {
    threads.emplace_back([&pool, &dtor_count]() {
      std::vector<TestObj*> objs;
      objs.reserve(N);

      for(int i = 0; i < N; i++) {
        objs.push_back(pool.animate(i, "x", &dtor_count));
      }

      for(auto* obj : objs) {
        pool.recycle(obj);
      }
    });
  }

  for(auto& t : threads) t.join();

  REQUIRE(dtor_count.load() == static_cast<int>(W) * N);
}

// ============================================================================
// op_nullptr
//
// Verifies that recycle(nullptr) is a safe no-op. W threads each call
// recycle(nullptr) repeatedly; no crash or undefined behavior should occur.
// ============================================================================

void op_nullptr(unsigned W) {

  tf::ObjectPool<TestObj> pool;

  const int N = 1024;

  std::vector<std::thread> threads;
  threads.reserve(W);

  for(unsigned w = 0; w < W; w++) {
    threads.emplace_back([&pool]() {
      for(int i = 0; i < N; i++) {
        pool.recycle(nullptr);
      }
    });
  }

  for(auto& t : threads) t.join();
}

// ============================================================================
// op_cross_thread
//
// Verifies correctness when animate and recycle occur on different threads,
// which is the common pattern in Taskflow: the calling thread creates a node
// and a worker thread destroys it after execution.
//
// The main thread animates W*N objects into a shared queue. W worker threads
// dequeue and recycle them. dtor_count must equal W*N after all workers join.
// ============================================================================

void op_cross_thread(unsigned W) {

  tf::ObjectPool<TestObj> pool;

  const int        N     = 1024;
  const int        total = static_cast<int>(W) * N;
  std::atomic<int> dtor_count{0};

  // main thread animates all objects upfront
  std::vector<TestObj*> shared;
  shared.reserve(total);

  for(int i = 0; i < total; i++) {
    shared.push_back(pool.animate(i, "cross", &dtor_count));
  }

  // worker threads recycle a disjoint slice each
  std::vector<std::thread> threads;
  threads.reserve(W);

  for(unsigned w = 0; w < W; w++) {
    threads.emplace_back([&shared, &pool, w]() {
      int begin = static_cast<int>(w) * N;
      int end   = begin + N;
      for(int i = begin; i < end; i++) {
        pool.recycle(shared[i]);
      }
    });
  }

  for(auto& t : threads) t.join();

  REQUIRE(dtor_count.load() == total);
}

// ============================================================================
// op_release
//
// Verifies that release() correctly resets the pool so it can be reused.
// Animates N objects, recycles all, calls release(), then animates N objects
// again. Total dtor_count must equal 2*N (both batches fully destructed).
// ============================================================================

void op_release(unsigned W) {

  tf::ObjectPool<TestObj> pool;

  const int        N = 2048;
  std::atomic<int> dtor_count{0};

  // --- phase 1: animate, recycle, release ---
  {
    std::vector<std::thread> threads;
    threads.reserve(W);

    for(unsigned w = 0; w < W; w++) {
      threads.emplace_back([&pool, &dtor_count]() {
        std::vector<TestObj*> objs;
        objs.reserve(N);
        for(int i = 0; i < N; i++) {
          objs.push_back(pool.animate(i, "phase1", &dtor_count));
        }
        for(auto* obj : objs) {
          pool.recycle(obj);
        }
      });
    }

    for(auto& t : threads) t.join();
  }

  REQUIRE(dtor_count.load() == static_cast<int>(W) * N);

  pool.release();

  // --- phase 2: pool must be fully functional after release ---
  {
    std::vector<std::thread> threads;
    threads.reserve(W);

    for(unsigned w = 0; w < W; w++) {
      threads.emplace_back([&pool, &dtor_count]() {
        std::vector<TestObj*> objs;
        objs.reserve(N);
        for(int i = 0; i < N; i++) {
          TestObj* obj = pool.animate(i, "phase2", &dtor_count);
          REQUIRE(obj != nullptr);
          REQUIRE(obj->value == i);
          objs.push_back(obj);
        }
        for(auto* obj : objs) {
          pool.recycle(obj);
        }
      });
    }

    for(auto& t : threads) t.join();
  }

  REQUIRE(dtor_count.load() == 2 * static_cast<int>(W) * N);
}

// ============================================================================
// op_stress
//
// High-frequency concurrent stress test. W threads each run a tight loop of
// animate-verify-recycle for a large number of iterations. Exercises the
// lock-free free stack under sustained contention and verifies that every
// constructed object is properly destructed.
// ============================================================================

void op_stress(unsigned W) {

  tf::ObjectPool<TestObj> pool;

  const int        N = 65536;
  std::atomic<int> dtor_count{0};

  std::vector<std::thread> threads;
  threads.reserve(W);

  for(unsigned w = 0; w < W; w++) {
    threads.emplace_back([&pool, &dtor_count, w]() {
      for(int i = 0; i < N; i++) {
        int      val = static_cast<int>(w) * N + i;
        TestObj* obj = pool.animate(val, "stress", &dtor_count);
        REQUIRE(obj != nullptr);
        REQUIRE(obj->value == val);
        pool.recycle(obj);
      }
    });
  }

  for(auto& t : threads) t.join();

  REQUIRE(dtor_count.load() == static_cast<int>(W) * N);
}

// ============================================================================
// TEST CASES: animate and recycle with constructor forwarding
// ============================================================================

TEST_CASE("ObjectPool.animate_recycle.1thread"  * doctest::timeout(300)) { op_animate_recycle(1);  }
TEST_CASE("ObjectPool.animate_recycle.2threads" * doctest::timeout(300)) { op_animate_recycle(2);  }
TEST_CASE("ObjectPool.animate_recycle.4threads" * doctest::timeout(300)) { op_animate_recycle(4);  }
TEST_CASE("ObjectPool.animate_recycle.8threads" * doctest::timeout(300)) { op_animate_recycle(8);  }

// ============================================================================
// TEST CASES: destructor called exactly once per recycle
// ============================================================================

TEST_CASE("ObjectPool.destructor.1thread"  * doctest::timeout(300)) { op_destructor(1); }
TEST_CASE("ObjectPool.destructor.2threads" * doctest::timeout(300)) { op_destructor(2); }
TEST_CASE("ObjectPool.destructor.4threads" * doctest::timeout(300)) { op_destructor(4); }
TEST_CASE("ObjectPool.destructor.8threads" * doctest::timeout(300)) { op_destructor(8); }

// ============================================================================
// TEST CASES: recycle(nullptr) is a safe no-op
// ============================================================================

TEST_CASE("ObjectPool.nullptr.1thread"  * doctest::timeout(300)) { op_nullptr(1); }
TEST_CASE("ObjectPool.nullptr.2threads" * doctest::timeout(300)) { op_nullptr(2); }
TEST_CASE("ObjectPool.nullptr.4threads" * doctest::timeout(300)) { op_nullptr(4); }
TEST_CASE("ObjectPool.nullptr.8threads" * doctest::timeout(300)) { op_nullptr(8); }

// ============================================================================
// TEST CASES: cross-thread animate/recycle
// ============================================================================

TEST_CASE("ObjectPool.cross_thread.1thread"  * doctest::timeout(300)) { op_cross_thread(1); }
TEST_CASE("ObjectPool.cross_thread.2threads" * doctest::timeout(300)) { op_cross_thread(2); }
TEST_CASE("ObjectPool.cross_thread.4threads" * doctest::timeout(300)) { op_cross_thread(4); }
TEST_CASE("ObjectPool.cross_thread.8threads" * doctest::timeout(300)) { op_cross_thread(8); }

// ============================================================================
// TEST CASES: release resets pool for reuse
// ============================================================================

TEST_CASE("ObjectPool.release.1thread"  * doctest::timeout(300)) { op_release(1); }
TEST_CASE("ObjectPool.release.2threads" * doctest::timeout(300)) { op_release(2); }
TEST_CASE("ObjectPool.release.4threads" * doctest::timeout(300)) { op_release(4); }
TEST_CASE("ObjectPool.release.8threads" * doctest::timeout(300)) { op_release(8); }

// ============================================================================
// TEST CASES: high-frequency stress under sustained concurrency
// ============================================================================

TEST_CASE("ObjectPool.stress.1thread"  * doctest::timeout(300)) { op_stress(1); }
TEST_CASE("ObjectPool.stress.2threads" * doctest::timeout(300)) { op_stress(2); }
TEST_CASE("ObjectPool.stress.4threads" * doctest::timeout(300)) { op_stress(4); }
TEST_CASE("ObjectPool.stress.8threads" * doctest::timeout(300)) { op_stress(8); }
