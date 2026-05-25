#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/utility/object_pool.hpp>


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
// op_animate_recycle<H>
//
// Each of W threads animates N objects with distinct values and labels,
// verifies that constructor arguments are forwarded correctly, then recycles
// all objects. Tests basic correctness of animate/recycle under concurrency.
// ============================================================================

template <typename H>
void op_animate_recycle(unsigned W) {

  tf::ObjectPool<TestObj, H, 5> pool;

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
// op_destructor<H>
//
// Verifies that recycle invokes the destructor of T exactly once per object.
// W threads each animate N objects sharing a single atomic dtor_count.
// After all threads finish, the counter must equal W * N.
// ============================================================================

template <typename H>
void op_destructor(unsigned W) {

  tf::ObjectPool<TestObj, H, 5> pool;

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
// op_nullptr<H>
//
// Verifies that recycle(nullptr) is a safe no-op. W threads each call
// recycle(nullptr) repeatedly; no crash or undefined behavior should occur.
// ============================================================================

template <typename H>
void op_nullptr(unsigned W) {

  tf::ObjectPool<TestObj, H, 5> pool;

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
// op_cross_thread<H>
//
// Verifies correctness when animate and recycle occur on different threads,
// which is the common pattern in Taskflow: the calling thread creates a node
// and a worker thread destroys it after execution.
//
// The main thread animates W*N objects into a shared queue. W worker threads
// dequeue and recycle them. dtor_count must equal W*N after all workers join.
// ============================================================================

template <typename H>
void op_cross_thread(unsigned W) {

  tf::ObjectPool<TestObj, H, 5> pool;

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
// op_release<H>
//
// Verifies that release() correctly resets the pool so it can be reused.
// Animates N objects, recycles all, calls release(), then animates N objects
// again. Total dtor_count must equal 2*N (both batches fully destructed).
// ============================================================================

template <typename H>
void op_release(unsigned W) {

  tf::ObjectPool<TestObj, H, 5> pool;

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
// op_stress<H>
//
// High-frequency concurrent stress test. W threads each run a tight loop of
// animate-verify-recycle for a large number of iterations. Exercises the
// lock-free free stack under sustained contention and verifies that every
// constructed object is properly destructed.
// ============================================================================

template <typename H>
void op_stress(unsigned W) {

  tf::ObjectPool<TestObj, H, 5> pool;

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
// TEST CASES — TaggedHead128
// ============================================================================

TEST_CASE("ObjectPool.H128.animate_recycle.1thread" * doctest::timeout(300)) {
  op_animate_recycle<tf::TaggedHead128>(1);
}
TEST_CASE("ObjectPool.H128.animate_recycle.2threads" * doctest::timeout(300)) {
  op_animate_recycle<tf::TaggedHead128>(2);
}
TEST_CASE("ObjectPool.H128.animate_recycle.4threads" * doctest::timeout(300)) {
  op_animate_recycle<tf::TaggedHead128>(4);
}
TEST_CASE("ObjectPool.H128.animate_recycle.8threads" * doctest::timeout(300)) {
  op_animate_recycle<tf::TaggedHead128>(8);
}

TEST_CASE("ObjectPool.H128.destructor.1thread" * doctest::timeout(300)) {
  op_destructor<tf::TaggedHead128>(1);
}
TEST_CASE("ObjectPool.H128.destructor.2threads" * doctest::timeout(300)) {
  op_destructor<tf::TaggedHead128>(2);
}
TEST_CASE("ObjectPool.H128.destructor.4threads" * doctest::timeout(300)) {
  op_destructor<tf::TaggedHead128>(4);
}
TEST_CASE("ObjectPool.H128.destructor.8threads" * doctest::timeout(300)) {
  op_destructor<tf::TaggedHead128>(8);
}

TEST_CASE("ObjectPool.H128.nullptr.1thread" * doctest::timeout(300)) {
  op_nullptr<tf::TaggedHead128>(1);
}
TEST_CASE("ObjectPool.H128.nullptr.2threads" * doctest::timeout(300)) {
  op_nullptr<tf::TaggedHead128>(2);
}
TEST_CASE("ObjectPool.H128.nullptr.4threads" * doctest::timeout(300)) {
  op_nullptr<tf::TaggedHead128>(4);
}
TEST_CASE("ObjectPool.H128.nullptr.8threads" * doctest::timeout(300)) {
  op_nullptr<tf::TaggedHead128>(8);
}

TEST_CASE("ObjectPool.H128.cross_thread.1thread" * doctest::timeout(300)) {
  op_cross_thread<tf::TaggedHead128>(1);
}
TEST_CASE("ObjectPool.H128.cross_thread.2threads" * doctest::timeout(300)) {
  op_cross_thread<tf::TaggedHead128>(2);
}
TEST_CASE("ObjectPool.H128.cross_thread.4threads" * doctest::timeout(300)) {
  op_cross_thread<tf::TaggedHead128>(4);
}
TEST_CASE("ObjectPool.H128.cross_thread.8threads" * doctest::timeout(300)) {
  op_cross_thread<tf::TaggedHead128>(8);
}

TEST_CASE("ObjectPool.H128.release.1thread" * doctest::timeout(300)) {
  op_release<tf::TaggedHead128>(1);
}
TEST_CASE("ObjectPool.H128.release.2threads" * doctest::timeout(300)) {
  op_release<tf::TaggedHead128>(2);
}
TEST_CASE("ObjectPool.H128.release.4threads" * doctest::timeout(300)) {
  op_release<tf::TaggedHead128>(4);
}
TEST_CASE("ObjectPool.H128.release.8threads" * doctest::timeout(300)) {
  op_release<tf::TaggedHead128>(8);
}

TEST_CASE("ObjectPool.H128.stress.1thread" * doctest::timeout(300)) {
  op_stress<tf::TaggedHead128>(1);
}
TEST_CASE("ObjectPool.H128.stress.2threads" * doctest::timeout(300)) {
  op_stress<tf::TaggedHead128>(2);
}
TEST_CASE("ObjectPool.H128.stress.4threads" * doctest::timeout(300)) {
  op_stress<tf::TaggedHead128>(4);
}
TEST_CASE("ObjectPool.H128.stress.8threads" * doctest::timeout(300)) {
  op_stress<tf::TaggedHead128>(8);
}

// ============================================================================
// TEST CASES — TaggedHead64<> (default PtrBits = TF_POINTER_BITS)
// ============================================================================

TEST_CASE("ObjectPool.H64.animate_recycle.1thread" * doctest::timeout(300)) {
  op_animate_recycle<tf::TaggedHead64<>>(1);
}
TEST_CASE("ObjectPool.H64.animate_recycle.2threads" * doctest::timeout(300)) {
  op_animate_recycle<tf::TaggedHead64<>>(2);
}
TEST_CASE("ObjectPool.H64.animate_recycle.4threads" * doctest::timeout(300)) {
  op_animate_recycle<tf::TaggedHead64<>>(4);
}
TEST_CASE("ObjectPool.H64.animate_recycle.8threads" * doctest::timeout(300)) {
  op_animate_recycle<tf::TaggedHead64<>>(8);
}

TEST_CASE("ObjectPool.H64.destructor.1thread" * doctest::timeout(300)) {
  op_destructor<tf::TaggedHead64<>>(1);
}
TEST_CASE("ObjectPool.H64.destructor.2threads" * doctest::timeout(300)) {
  op_destructor<tf::TaggedHead64<>>(2);
}
TEST_CASE("ObjectPool.H64.destructor.4threads" * doctest::timeout(300)) {
  op_destructor<tf::TaggedHead64<>>(4);
}
TEST_CASE("ObjectPool.H64.destructor.8threads" * doctest::timeout(300)) {
  op_destructor<tf::TaggedHead64<>>(8);
}

TEST_CASE("ObjectPool.H64.nullptr.1thread" * doctest::timeout(300)) {
  op_nullptr<tf::TaggedHead64<>>(1);
}
TEST_CASE("ObjectPool.H64.nullptr.2threads" * doctest::timeout(300)) {
  op_nullptr<tf::TaggedHead64<>>(2);
}
TEST_CASE("ObjectPool.H64.nullptr.4threads" * doctest::timeout(300)) {
  op_nullptr<tf::TaggedHead64<>>(4);
}
TEST_CASE("ObjectPool.H64.nullptr.8threads" * doctest::timeout(300)) {
  op_nullptr<tf::TaggedHead64<>>(8);
}

TEST_CASE("ObjectPool.H64.cross_thread.1thread" * doctest::timeout(300)) {
  op_cross_thread<tf::TaggedHead64<>>(1);
}
TEST_CASE("ObjectPool.H64.cross_thread.2threads" * doctest::timeout(300)) {
  op_cross_thread<tf::TaggedHead64<>>(2);
}
TEST_CASE("ObjectPool.H64.cross_thread.4threads" * doctest::timeout(300)) {
  op_cross_thread<tf::TaggedHead64<>>(4);
}
TEST_CASE("ObjectPool.H64.cross_thread.8threads" * doctest::timeout(300)) {
  op_cross_thread<tf::TaggedHead64<>>(8);
}

TEST_CASE("ObjectPool.H64.release.1thread" * doctest::timeout(300)) {
  op_release<tf::TaggedHead64<>>(1);
}
TEST_CASE("ObjectPool.H64.release.2threads" * doctest::timeout(300)) {
  op_release<tf::TaggedHead64<>>(2);
}
TEST_CASE("ObjectPool.H64.release.4threads" * doctest::timeout(300)) {
  op_release<tf::TaggedHead64<>>(4);
}
TEST_CASE("ObjectPool.H64.release.8threads" * doctest::timeout(300)) {
  op_release<tf::TaggedHead64<>>(8);
}

TEST_CASE("ObjectPool.H64.stress.1thread" * doctest::timeout(300)) {
  op_stress<tf::TaggedHead64<>>(1);
}
TEST_CASE("ObjectPool.H64.stress.2threads" * doctest::timeout(300)) {
  op_stress<tf::TaggedHead64<>>(2);
}
TEST_CASE("ObjectPool.H64.stress.4threads" * doctest::timeout(300)) {
  op_stress<tf::TaggedHead64<>>(4);
}
TEST_CASE("ObjectPool.H64.stress.8threads" * doctest::timeout(300)) {
  op_stress<tf::TaggedHead64<>>(8);
}
