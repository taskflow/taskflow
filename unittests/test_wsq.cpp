#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>


// ============================================================================
// BoundedWSQ Tests
// ============================================================================

// Procedure: bounded_wsq_owner
// Tests basic owner-thread push/pop/steal operations on BoundedWSQ.
// Runs across multiple iterations to stress the wrap-around behavior of the
// circular buffer.
template<size_t LogSize>
void bounded_wsq_owner() {

  tf::BoundedWSQ<void*, LogSize> queue;

  constexpr size_t N = (1 << LogSize);

  std::vector<void*> data(1);  // dummy space to avoid nullptr when calling .data

  for(size_t k=0; k<LogSize*10; k++) {

    data.clear();

    REQUIRE(queue.empty() == true);

    for(size_t i=0; i<N; i++) {
      auto ptr = data.data() + i;
      REQUIRE(queue.try_push(ptr) == true);
      data.push_back(ptr);
    }
    REQUIRE(queue.try_push(nullptr) == false);

    for(size_t i=0; i<N; i++) {
      auto item = queue.pop();
      REQUIRE(item != nullptr);
      REQUIRE(item == data[N-i-1]);
    }

    for(size_t i=0; i<N; i++) {
      REQUIRE(queue.pop() == nullptr);
    }
  }

  // test steal (plain, two-state)
  size_t dummy1, dummy2;

  REQUIRE(queue.try_push(&dummy1) == true);
  REQUIRE(queue.try_push(&dummy2) == true);
  REQUIRE(queue.steal() == &dummy1);
  REQUIRE(queue.steal() == &dummy2);
  REQUIRE(queue.steal() == nullptr);
}

TEST_CASE("BoundedWSQ.Owner.LogSize=2" * doctest::timeout(300)) {
  bounded_wsq_owner<2>();
}

TEST_CASE("BoundedWSQ.Owner.LogSize=3" * doctest::timeout(300)) {
  bounded_wsq_owner<3>();
}

TEST_CASE("BoundedWSQ.Owner.LogSize=4" * doctest::timeout(300)) {
  bounded_wsq_owner<4>();
}

TEST_CASE("BoundedWSQ.Owner.LogSize=5" * doctest::timeout(300)) {
  bounded_wsq_owner<5>();
}

TEST_CASE("BoundedWSQ.Owner.LogSize=6" * doctest::timeout(300)) {
  bounded_wsq_owner<6>();
}

TEST_CASE("BoundedWSQ.Owner.LogSize=7" * doctest::timeout(300)) {
  bounded_wsq_owner<7>();
}

TEST_CASE("BoundedWSQ.Owner.LogSize=8" * doctest::timeout(300)) {
  bounded_wsq_owner<8>();
}

TEST_CASE("BoundedWSQ.Owner.LogSize=9" * doctest::timeout(300)) {
  bounded_wsq_owner<9>();
}

TEST_CASE("BoundedWSQ.Owner.LogSize=10" * doctest::timeout(300)) {
  bounded_wsq_owner<10>();
}

// ----------------------------------------------------------------------------
// BoundedWSQ steal_with_feedback three-state sentinel test
//
// steal_with_feedback() returns one of three values:
//   - a valid pointer   : successfully stolen item (CAS succeeded)
//   - contended_value() : queue non-empty but CAS lost to another thief
//   - empty_value()     : queue was genuinely empty
//
// In a single-threaded test we can only observe STOLEN and EMPTY since there
// are no competing thieves to cause a CAS loss. The contended case is verified
// in the multi-consumer stress test below.
// ----------------------------------------------------------------------------

template<size_t LogSize>
void bounded_wsq_feedback() {

  tf::BoundedWSQ<void*, LogSize> queue;

  const auto EMPTY     = tf::BoundedWSQ<void*, LogSize>::empty_value();
  const auto CONTENDED = tf::BoundedWSQ<void*, LogSize>::contended_value();

  // sentinel sanity checks
  REQUIRE(EMPTY     == nullptr);  // empty_value() must be nullptr for pointer types
  REQUIRE(CONTENDED != nullptr);  // contended_value() must be non-null for pointer types
  REQUIRE(CONTENDED != EMPTY);    // contended_value() and empty_value() must be distinct
  // 0x1 is safe because alignof(void) >= 1 and real pointers are always >= 8
  REQUIRE(reinterpret_cast<uintptr_t>(CONTENDED) == uintptr_t{1});

  size_t dummy1, dummy2, dummy3;

  // --- empty queue: should return empty_value(), never contended_value() ---
  REQUIRE(queue.steal_with_feedback() == EMPTY);
  REQUIRE(queue.steal_with_feedback() == EMPTY);

  // --- push two items and steal them; should return valid pointers ---
  REQUIRE(queue.try_push(&dummy1) == true);
  REQUIRE(queue.try_push(&dummy2) == true);

  // steal is FIFO: dummy1 was pushed first, comes out first
  void* r1 = queue.steal_with_feedback();
  REQUIRE(r1 != EMPTY);
  REQUIRE(r1 != CONTENDED);
  REQUIRE(r1 == &dummy1);

  void* r2 = queue.steal_with_feedback();
  REQUIRE(r2 != EMPTY);
  REQUIRE(r2 != CONTENDED);
  REQUIRE(r2 == &dummy2);

  // queue now empty again
  REQUIRE(queue.steal_with_feedback() == EMPTY);
  REQUIRE(queue.steal_with_feedback() == EMPTY);

  // --- interleave push and steal_with_feedback ---
  REQUIRE(queue.try_push(&dummy1) == true);
  REQUIRE(queue.try_push(&dummy2) == true);
  REQUIRE(queue.try_push(&dummy3) == true);

  void* r3 = queue.steal_with_feedback();
  REQUIRE(r3 == &dummy1);  // FIFO steal order

  // steal remaining two
  REQUIRE(queue.steal_with_feedback() == &dummy2);
  REQUIRE(queue.steal_with_feedback() == &dummy3);

  // empty again
  REQUIRE(queue.steal_with_feedback() == EMPTY);

  // --- plain steal() still works alongside steal_with_feedback() ---
  REQUIRE(queue.try_push(&dummy1) == true);
  REQUIRE(queue.steal() == &dummy1);
  REQUIRE(queue.steal_with_feedback() == EMPTY);
}

TEST_CASE("BoundedWSQ.Feedback.LogSize=2" * doctest::timeout(300)) {
  bounded_wsq_feedback<2>();
}

TEST_CASE("BoundedWSQ.Feedback.LogSize=4" * doctest::timeout(300)) {
  bounded_wsq_feedback<4>();
}

TEST_CASE("BoundedWSQ.Feedback.LogSize=8" * doctest::timeout(300)) {
  bounded_wsq_feedback<8>();
}

// ----------------------------------------------------------------------------
// BoundedWSQ steal_with_feedback contention stress test
//
// Verifies that under concurrent thieves, steal_with_feedback() never returns
// a value other than a valid pointer, empty_value(), or contended_value().
// Also verifies that all N items are eventually consumed exactly once across
// all thieves (correctness under contention).
// ----------------------------------------------------------------------------

void bounded_wsq_feedback_n_consumers(size_t M) {

  tf::BoundedWSQ<void*> queue;

  const auto EMPTY     = tf::BoundedWSQ<void*>::empty_value();
  const auto CONTENDED = tf::BoundedWSQ<void*>::contended_value();

  std::vector<void*> gold;
  std::atomic<size_t> consumed;

  // 1, 4, 13, 40, 121, 364, 1093, 3280, 9841, 29524, 88573
  for(size_t N=1; N<=88573; N=N*3+1) {

    REQUIRE(queue.empty());

    gold.resize(N);
    consumed = 0;

    for(size_t i=0; i<N; ++i) {
      gold[i] = gold.data() + i;
    }

    // thieves use steal_with_feedback; contended_value() means retry same victim
    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> stolens(M);

    for(size_t i=0; i<M; ++i) {
      threads.emplace_back([&, i](){
        while(consumed != N) {
          auto ptr = queue.steal_with_feedback();
          if(ptr == EMPTY || ptr == CONTENDED) {
            continue;  // empty or lost CAS: try again
          }
          // valid stolen item
          stolens[i].push_back(ptr);
          consumed.fetch_add(1, std::memory_order_relaxed);
        }
        // after all consumed, queue must appear empty to steal_with_feedback
        auto final = queue.steal_with_feedback();
        REQUIRE((final == EMPTY || final == CONTENDED));
      });
    }

    // owner pushes all items
    for(size_t i=0; i<N; ++i) {
      while(queue.try_push(gold[i]) == false);
    }

    // owner also pops
    std::vector<void*> items;
    while(consumed != N) {
      auto ptr = queue.pop();
      if(ptr != nullptr) {
        items.push_back(ptr);
        consumed.fetch_add(1, std::memory_order_relaxed);
      }
    }

    REQUIRE(queue.steal_with_feedback() == EMPTY);
    REQUIRE(queue.pop() == nullptr);
    REQUIRE(queue.empty());

    for(auto& thread : threads) thread.join();

    for(size_t i=0; i<M; ++i) {
      for(auto s : stolens[i]) items.push_back(s);
    }

    std::sort(items.begin(), items.end());
    std::sort(gold.begin(), gold.end());
    REQUIRE(items.size() == N);
    REQUIRE(items == gold);
  }
}

TEST_CASE("BoundedWSQ.Feedback.1Consumer" * doctest::timeout(300)) {
  bounded_wsq_feedback_n_consumers(1);
}

TEST_CASE("BoundedWSQ.Feedback.2Consumers" * doctest::timeout(300)) {
  bounded_wsq_feedback_n_consumers(2);
}

TEST_CASE("BoundedWSQ.Feedback.4Consumers" * doctest::timeout(300)) {
  bounded_wsq_feedback_n_consumers(4);
}

TEST_CASE("BoundedWSQ.Feedback.8Consumers" * doctest::timeout(300)) {
  bounded_wsq_feedback_n_consumers(8);
}

// ----------------------------------------------------------------------------
// BoundedWSQ ValueType test
// ----------------------------------------------------------------------------

TEST_CASE("BoundedWSQ.ValueType") {
  tf::BoundedWSQ<void*> Q1;
  tf::BoundedWSQ<int>   Q2;

  // empty_value(): nullptr for pointer types, nullopt for non-pointer
  auto empty1     = Q1.empty_value();
  auto empty2     = Q2.empty_value();
  auto contended1 = Q1.contended_value();
  auto contended2 = Q2.contended_value();

  static_assert(std::is_same_v<decltype(empty1),     void*>);
  static_assert(std::is_same_v<decltype(empty2),     std::optional<int>>);
  static_assert(std::is_same_v<decltype(contended1), void*>);
  static_assert(std::is_same_v<decltype(contended2), std::optional<int>>);

  REQUIRE(empty1     == nullptr);
  REQUIRE(empty2     == std::nullopt);
  // contended_value() for pointer type encodes as 0x1 sentinel
  REQUIRE(reinterpret_cast<uintptr_t>(contended1) == uintptr_t{1});
  // contended_value() for non-pointer type falls back to nullopt (same as empty)
  REQUIRE(contended2 == std::nullopt);
  // pointer type: empty and contended must be distinct
  REQUIRE(empty1 != contended1);

  // push/pop/steal still work normally
  auto v = Q2.pop();
  REQUIRE(v == std::nullopt);

  Q2.try_push(1);
  Q2.try_push(2);
  Q2.try_push(3);
  Q2.try_push(4);

  REQUIRE(Q2.pop()   == 4);
  REQUIRE(Q2.pop()   == 3);
  REQUIRE(Q2.pop()   == 2);
  REQUIRE(Q2.pop()   == 1);
  REQUIRE(Q2.pop()   == std::nullopt);

  Q2.try_push(1);
  Q2.try_push(2);
  Q2.try_push(3);
  Q2.try_push(4);
  REQUIRE(Q2.steal() == 1);
  REQUIRE(Q2.steal() == 2);
  REQUIRE(Q2.steal() == 3);
  REQUIRE(Q2.steal() == 4);
  REQUIRE(Q2.steal() == std::nullopt);
}

// ----------------------------------------------------------------------------
// BoundedWSQ Multiple Consumers Test (original, using plain steal)
// ----------------------------------------------------------------------------

void bounded_wsq_n_consumers(size_t M) {

  tf::BoundedWSQ<void*> queue;

  std::vector<void*> gold;
  std::atomic<size_t> consumed;

  // 1, 4, 13, 40, 121, 364, 1093, 3280, 9841, 29524, 88573
  for(size_t N=1; N<=88573; N=N*3+1) {

    REQUIRE(queue.empty());

    gold.resize(N);
    consumed = 0;

    for(size_t i=0; i<N; ++i) {
      gold[i] = gold.data() + i;
    }

    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> stolens(M);

    for(size_t i=0; i<M; ++i) {
      threads.emplace_back([&, i](){
        while(consumed != N) {
          auto ptr = queue.steal();
          if(ptr != nullptr) {
            stolens[i].push_back(ptr);
            consumed.fetch_add(1, std::memory_order_relaxed);
          }
        }
        REQUIRE(queue.steal() == nullptr);
      });
    }

    for(size_t i=0; i<N; ++i) {
      while(queue.try_push(gold[i]) == false);
    }

    std::vector<void*> items;
    while(consumed != N) {
      auto ptr = queue.pop();
      if(ptr != nullptr) {
        items.push_back(ptr);
        consumed.fetch_add(1, std::memory_order_relaxed);
      }
    }
    REQUIRE(queue.steal() == nullptr);
    REQUIRE(queue.pop()   == nullptr);
    REQUIRE(queue.empty());

    for(auto& thread : threads) thread.join();

    for(size_t i=0; i<M; ++i) {
      for(auto s : stolens[i]) items.push_back(s);
    }

    std::sort(items.begin(), items.end());
    std::sort(gold.begin(),  gold.end());
    REQUIRE(items.size() == N);
    REQUIRE(items == gold);
  }
}

TEST_CASE("BoundedWSQ.1Consumer" * doctest::timeout(300)) {
  bounded_wsq_n_consumers(1);
}

TEST_CASE("BoundedWSQ.2Consumers" * doctest::timeout(300)) {
  bounded_wsq_n_consumers(2);
}

TEST_CASE("BoundedWSQ.3Consumers" * doctest::timeout(300)) {
  bounded_wsq_n_consumers(3);
}

TEST_CASE("BoundedWSQ.4Consumers" * doctest::timeout(300)) {
  bounded_wsq_n_consumers(4);
}

TEST_CASE("BoundedWSQ.5Consumers" * doctest::timeout(300)) {
  bounded_wsq_n_consumers(5);
}

TEST_CASE("BoundedWSQ.6Consumers" * doctest::timeout(300)) {
  bounded_wsq_n_consumers(6);
}

TEST_CASE("BoundedWSQ.7Consumers" * doctest::timeout(300)) {
  bounded_wsq_n_consumers(7);
}

TEST_CASE("BoundedWSQ.8Consumers" * doctest::timeout(300)) {
  bounded_wsq_n_consumers(8);
}

// ----------------------------------------------------------------------------
// BoundedWSQ Multiple Consumers BulkPush Test
// ----------------------------------------------------------------------------

void bounded_wsq_n_consumers_bulk_push(size_t M) {

  tf::BoundedWSQ<void*> queue;

  std::vector<void*> gold;
  std::atomic<size_t> consumed;

  // 1, 4, 13, 40, 121, 364, 1093, 3280, 9841, 29524, 88573
  for(size_t N=1; N<=88573; N=N*3+1) {

    REQUIRE(queue.empty());

    gold.resize(N);
    consumed = 0;

    for(size_t i=0; i<N; ++i) {
      gold[i] = gold.data() + i;
    }

    size_t capacity = queue.capacity();
    const size_t num_pushable = std::min(capacity, N);

    // bulk push and pop
    auto first = gold.data();
    REQUIRE(num_pushable == queue.try_bulk_push(first, N));
    REQUIRE(queue.size() == num_pushable);
    for(size_t i=0; i<num_pushable; i++) {
      REQUIRE(queue.pop() == gold[num_pushable - i - 1]);
    }
    REQUIRE(queue.empty() == true);

    // bulk push and steal
    first = gold.data();
    REQUIRE(num_pushable == queue.try_bulk_push(first, N));
    REQUIRE(queue.size() == num_pushable);
    for(size_t i=0; i<num_pushable; i++) {
      REQUIRE(queue.steal() == gold[i]);
    }
    REQUIRE(queue.empty() == true);

    // concurrent bulk push + thieves
    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> stolens(M);

    for(size_t i=0; i<M; ++i) {
      threads.emplace_back([&, i](){
        while(consumed != N) {
          auto ptr = queue.steal();
          if(ptr != nullptr) {
            stolens[i].push_back(ptr);
            consumed.fetch_add(1, std::memory_order_relaxed);
          }
        }
        REQUIRE(queue.steal() == nullptr);
      });
    }

    first = gold.data();
    for(size_t n=0; n<N;) {
      n += queue.try_bulk_push(first, N-n);
    }

    std::vector<void*> items;
    while(consumed != N) {
      auto ptr = queue.pop();
      if(ptr != nullptr) {
        items.push_back(ptr);
        consumed.fetch_add(1, std::memory_order_relaxed);
      }
    }
    REQUIRE(queue.steal() == nullptr);
    REQUIRE(queue.pop()   == nullptr);
    REQUIRE(queue.empty());

    for(auto& thread : threads) thread.join();

    for(size_t i=0; i<M; ++i) {
      for(auto s : stolens[i]) items.push_back(s);
    }

    std::sort(items.begin(), items.end());
    std::sort(gold.begin(),  gold.end());
    REQUIRE(items.size() == N);
    REQUIRE(items == gold);
  }
}

TEST_CASE("BoundedWSQ.1Consumer.BulkPush" * doctest::timeout(300)) {
  bounded_wsq_n_consumers_bulk_push(1);
}

TEST_CASE("BoundedWSQ.2Consumers.BulkPush" * doctest::timeout(300)) {
  bounded_wsq_n_consumers_bulk_push(2);
}

TEST_CASE("BoundedWSQ.3Consumers.BulkPush" * doctest::timeout(300)) {
  bounded_wsq_n_consumers_bulk_push(3);
}

TEST_CASE("BoundedWSQ.4Consumers.BulkPush" * doctest::timeout(300)) {
  bounded_wsq_n_consumers_bulk_push(4);
}

TEST_CASE("BoundedWSQ.5Consumers.BulkPush" * doctest::timeout(300)) {
  bounded_wsq_n_consumers_bulk_push(5);
}

TEST_CASE("BoundedWSQ.6Consumers.BulkPush" * doctest::timeout(300)) {
  bounded_wsq_n_consumers_bulk_push(6);
}

TEST_CASE("BoundedWSQ.7Consumers.BulkPush" * doctest::timeout(300)) {
  bounded_wsq_n_consumers_bulk_push(7);
}

TEST_CASE("BoundedWSQ.8Consumers.BulkPush" * doctest::timeout(300)) {
  bounded_wsq_n_consumers_bulk_push(8);
}


// ============================================================================
// UnboundedWSQ Tests
// ============================================================================

TEST_CASE("UnboundedWSQ.Resize") {
  tf::UnboundedWSQ<void*> queue(1);
  REQUIRE(queue.capacity() == 2);

  std::vector<void*> data(2048);

  auto first = data.data();
  queue.bulk_push(first, 1);
  REQUIRE(queue.size()     == 1);
  REQUIRE(queue.capacity() == 2);

  first = data.data();
  queue.bulk_push(first, 2);
  REQUIRE(queue.size()     == 3);
  REQUIRE(queue.capacity() == 4);

  first = data.data();
  queue.bulk_push(first, 10);
  REQUIRE(queue.size()     == 13);
  REQUIRE(queue.capacity() == 16);

  first = data.data();
  queue.bulk_push(first, 1200);
  REQUIRE(queue.size()     == 1213);
  REQUIRE(queue.capacity() == 2048);

  for(size_t i=0; i<1213; ++i) {
    REQUIRE(queue.size() == 1213 - i);
    queue.pop();
  }
  REQUIRE(queue.empty() == true);

  // capacity does not shrink after drain
  first = data.data();
  queue.bulk_push(first, 1);
  REQUIRE(queue.size()     == 1);
  REQUIRE(queue.capacity() == 2048);

  first = data.data();
  queue.bulk_push(first, 2);
  REQUIRE(queue.size()     == 3);
  REQUIRE(queue.capacity() == 2048);

  first = data.data();
  queue.bulk_push(first, 10);
  REQUIRE(queue.size()     == 13);
  REQUIRE(queue.capacity() == 2048);

  first = data.data();
  queue.bulk_push(first, 1200);
  REQUIRE(queue.size()     == 1213);
  REQUIRE(queue.capacity() == 2048);
}

// Procedure: unbounded_wsq_owner
void unbounded_wsq_owner() {

  tf::UnboundedWSQ<void*> queue;
  std::vector<void*> gold;

  for(size_t N=1; N<=777777; N=N*2+1) {

    gold.resize(N);
    REQUIRE(queue.empty());

    for(size_t i=0; i<N; ++i) {
      gold[i] = gold.data() + i;
      queue.push(gold[i]);
    }
    for(size_t i=0; i<N; ++i) {
      auto ptr = queue.pop();
      REQUIRE(ptr != nullptr);
      REQUIRE(gold[N-i-1] == ptr);
    }
    REQUIRE(queue.pop() == nullptr);

    for(size_t i=0; i<N; ++i) {
      queue.push(gold[i]);
    }
    for(size_t i=0; i<N; ++i) {
      auto ptr = queue.steal();
      REQUIRE(ptr != nullptr);
      REQUIRE(gold[i] == ptr);
    }
  }
}

TEST_CASE("UnboundedWSQ.Owner" * doctest::timeout(300)) {
  unbounded_wsq_owner();
}

// ----------------------------------------------------------------------------
// UnboundedWSQ steal_with_feedback three-state sentinel test
// ----------------------------------------------------------------------------

void unbounded_wsq_feedback() {

  tf::UnboundedWSQ<void*> queue;

  const auto EMPTY     = tf::UnboundedWSQ<void*>::empty_value();
  const auto CONTENDED = tf::UnboundedWSQ<void*>::contended_value();

  REQUIRE(EMPTY     == nullptr);  // empty_value() must be nullptr for pointer types
  REQUIRE(CONTENDED != nullptr);  // contended_value() must be non-null for pointer types
  REQUIRE(CONTENDED != EMPTY);    // contended_value() and empty_value() must be distinct
  REQUIRE(reinterpret_cast<uintptr_t>(CONTENDED) == uintptr_t{1});

  size_t dummy1, dummy2, dummy3;

  // empty queue returns empty_value()
  REQUIRE(queue.steal_with_feedback() == EMPTY);
  REQUIRE(queue.steal_with_feedback() == EMPTY);

  // push and steal: should get valid pointers in FIFO order
  queue.push(&dummy1);
  queue.push(&dummy2);

  void* r1 = queue.steal_with_feedback();
  REQUIRE(r1 != EMPTY);
  REQUIRE(r1 != CONTENDED);
  REQUIRE(r1 == &dummy1);

  void* r2 = queue.steal_with_feedback();
  REQUIRE(r2 != EMPTY);
  REQUIRE(r2 != CONTENDED);
  REQUIRE(r2 == &dummy2);

  REQUIRE(queue.steal_with_feedback() == EMPTY);

  // interleave push and steal_with_feedback
  queue.push(&dummy1);
  queue.push(&dummy2);
  queue.push(&dummy3);

  REQUIRE(queue.steal_with_feedback() == &dummy1);
  REQUIRE(queue.steal_with_feedback() == &dummy2);
  REQUIRE(queue.steal_with_feedback() == &dummy3);
  REQUIRE(queue.steal_with_feedback() == EMPTY);

  // plain steal() still works alongside steal_with_feedback()
  queue.push(&dummy1);
  REQUIRE(queue.steal() == &dummy1);
  REQUIRE(queue.steal_with_feedback() == EMPTY);
}

TEST_CASE("UnboundedWSQ.Feedback" * doctest::timeout(300)) {
  unbounded_wsq_feedback();
}

// ----------------------------------------------------------------------------
// UnboundedWSQ steal_with_feedback contention stress test
// ----------------------------------------------------------------------------

void unbounded_wsq_feedback_n_consumers(size_t M) {

  tf::UnboundedWSQ<void*> queue;

  const auto EMPTY     = tf::UnboundedWSQ<void*>::empty_value();
  const auto CONTENDED = tf::UnboundedWSQ<void*>::contended_value();

  std::vector<void*> gold;
  std::atomic<size_t> consumed;

  // 1, 4, 13, 40, 121, 364, 1093, 3280, 9841, 29524, 88573, 265720
  for(size_t N=1; N<=265720; N=N*3+1) {

    REQUIRE(queue.empty());

    gold.resize(N);
    consumed = 0;

    for(size_t i=0; i<N; ++i) {
      gold[i] = gold.data() + i;
    }

    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> stolens(M);

    for(size_t i=0; i<M; ++i) {
      threads.emplace_back([&, i](){
        while(consumed != N) {
          auto ptr = queue.steal_with_feedback();
          if(ptr == EMPTY || ptr == CONTENDED) {
            continue;
          }
          stolens[i].push_back(ptr);
          consumed.fetch_add(1, std::memory_order_relaxed);
        }
        auto final = queue.steal_with_feedback();
        REQUIRE((final == EMPTY || final == CONTENDED));
      });
    }

    for(size_t i=0; i<N; ++i) {
      queue.push(gold[i]);
    }

    std::vector<void*> items;
    while(consumed != N) {
      auto ptr = queue.pop();
      if(ptr != nullptr) {
        items.push_back(ptr);
        consumed.fetch_add(1, std::memory_order_relaxed);
      }
    }
    REQUIRE(queue.steal_with_feedback() == EMPTY);
    REQUIRE(queue.pop() == nullptr);
    REQUIRE(queue.empty());

    for(auto& thread : threads) thread.join();

    for(size_t i=0; i<M; ++i) {
      for(auto s : stolens[i]) items.push_back(s);
    }

    std::sort(items.begin(), items.end());
    std::sort(gold.begin(),  gold.end());
    REQUIRE(items.size() == N);
    REQUIRE(items == gold);
  }
}

TEST_CASE("UnboundedWSQ.Feedback.1Consumer" * doctest::timeout(300)) {
  unbounded_wsq_feedback_n_consumers(1);
}

TEST_CASE("UnboundedWSQ.Feedback.2Consumers" * doctest::timeout(300)) {
  unbounded_wsq_feedback_n_consumers(2);
}

TEST_CASE("UnboundedWSQ.Feedback.4Consumers" * doctest::timeout(300)) {
  unbounded_wsq_feedback_n_consumers(4);
}

TEST_CASE("UnboundedWSQ.Feedback.8Consumers" * doctest::timeout(300)) {
  unbounded_wsq_feedback_n_consumers(8);
}

// ----------------------------------------------------------------------------
// UnboundedWSQ ValueType test
// ----------------------------------------------------------------------------

TEST_CASE("UnboundedWSQ.ValueType") {

  tf::UnboundedWSQ<void*> Q1;
  tf::UnboundedWSQ<int>   Q2;

  auto empty1     = Q1.empty_value();
  auto empty2     = Q2.empty_value();
  auto contended1 = Q1.contended_value();
  auto contended2 = Q2.contended_value();

  static_assert(std::is_same_v<decltype(empty1),     void*>);
  static_assert(std::is_same_v<decltype(empty2),     std::optional<int>>);
  static_assert(std::is_same_v<decltype(contended1), void*>);
  static_assert(std::is_same_v<decltype(contended2), std::optional<int>>);

  REQUIRE(empty1     == nullptr);
  REQUIRE(empty2     == std::nullopt);
  REQUIRE(reinterpret_cast<uintptr_t>(contended1) == uintptr_t{1});
  REQUIRE(contended2 == std::nullopt);
  REQUIRE(empty1 != contended1);

  auto v = Q2.pop();
  REQUIRE(v == std::nullopt);

  Q2.push(1);
  Q2.push(2);
  Q2.push(3);
  Q2.push(4);

  REQUIRE(Q2.pop()   == 4);
  REQUIRE(Q2.pop()   == 3);
  REQUIRE(Q2.pop()   == 2);
  REQUIRE(Q2.pop()   == 1);
  REQUIRE(Q2.pop()   == std::nullopt);

  Q2.push(1);
  Q2.push(2);
  Q2.push(3);
  Q2.push(4);
  REQUIRE(Q2.steal() == 1);
  REQUIRE(Q2.steal() == 2);
  REQUIRE(Q2.steal() == 3);
  REQUIRE(Q2.steal() == 4);
  REQUIRE(Q2.steal() == std::nullopt);
}

// ----------------------------------------------------------------------------
// UnboundedWSQ Multiple Consumers Test (original, using plain steal)
// ----------------------------------------------------------------------------

void unbounded_wsq_n_consumers(size_t M) {

  tf::UnboundedWSQ<void*> queue;

  std::vector<void*> gold;
  std::atomic<size_t> consumed;

  // 1, 4, 13, 40, 121, 364, 1093, 3280, 9841, 29524, 88573, 265720
  for(size_t N=1; N<=265720; N=N*3+1) {

    REQUIRE(queue.empty());

    gold.resize(N);
    consumed = 0;

    for(size_t i=0; i<N; ++i) {
      gold[i] = gold.data() + i;
    }

    // master push and pop
    for(size_t i=0; i<N; ++i) {
      queue.push(gold.data() + i);
    }
    REQUIRE(queue.size() == N);
    for(size_t i=0; i<N; i++) {
      REQUIRE(queue.pop() == gold[N - i - 1]);
    }
    REQUIRE(queue.empty() == true);

    // master push and steal
    for(size_t i=0; i<N; ++i) {
      queue.push(gold.data() + i);
    }
    REQUIRE(queue.size() == N);
    for(size_t i=0; i<N; i++) {
      REQUIRE(queue.steal() == gold[i]);
    }
    REQUIRE(queue.empty() == true);

    // concurrent
    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> stolens(M);

    for(size_t i=0; i<M; ++i) {
      threads.emplace_back([&, i](){
        while(consumed != N) {
          auto ptr = queue.steal();
          if(ptr != nullptr) {
            stolens[i].push_back(ptr);
            consumed.fetch_add(1, std::memory_order_relaxed);
          }
        }
        REQUIRE(queue.steal() == nullptr);
      });
    }

    for(size_t i=0; i<N; ++i) {
      queue.push(gold[i]);
    }

    std::vector<void*> items;
    while(consumed != N) {
      auto ptr = queue.pop();
      if(ptr != nullptr) {
        items.push_back(ptr);
        consumed.fetch_add(1, std::memory_order_relaxed);
      }
    }
    REQUIRE(queue.steal() == nullptr);
    REQUIRE(queue.pop()   == nullptr);
    REQUIRE(queue.empty());

    for(auto& thread : threads) thread.join();

    for(size_t i=0; i<M; ++i) {
      for(auto s : stolens[i]) items.push_back(s);
    }

    std::sort(items.begin(), items.end());
    std::sort(gold.begin(),  gold.end());
    REQUIRE(items.size() == N);
    REQUIRE(items == gold);
  }
}

TEST_CASE("UnboundedWSQ.1Consumer" * doctest::timeout(300)) {
  unbounded_wsq_n_consumers(1);
}

TEST_CASE("UnboundedWSQ.2Consumers" * doctest::timeout(300)) {
  unbounded_wsq_n_consumers(2);
}

TEST_CASE("UnboundedWSQ.3Consumers" * doctest::timeout(300)) {
  unbounded_wsq_n_consumers(3);
}

TEST_CASE("UnboundedWSQ.4Consumers" * doctest::timeout(300)) {
  unbounded_wsq_n_consumers(4);
}

TEST_CASE("UnboundedWSQ.5Consumers" * doctest::timeout(300)) {
  unbounded_wsq_n_consumers(5);
}

TEST_CASE("UnboundedWSQ.6Consumers" * doctest::timeout(300)) {
  unbounded_wsq_n_consumers(6);
}

TEST_CASE("UnboundedWSQ.7Consumers" * doctest::timeout(300)) {
  unbounded_wsq_n_consumers(7);
}

TEST_CASE("UnboundedWSQ.8Consumers" * doctest::timeout(300)) {
  unbounded_wsq_n_consumers(8);
}

// ----------------------------------------------------------------------------
// UnboundedWSQ Multiple Consumers BulkPush Test
// ----------------------------------------------------------------------------

void unbounded_wsq_n_consumers_bulk_push(size_t M) {

  tf::UnboundedWSQ<void*> queue;

  std::vector<void*> gold;
  std::atomic<size_t> consumed;

  // 1, 4, 13, 40, 121, 364, 1093, 3280, 9841, 29524, 88573, 265720
  for(size_t N=1; N<=265720; N=N*3+1) {

    REQUIRE(queue.empty());

    gold.resize(N);
    consumed = 0;

    for(size_t i=0; i<N; ++i) {
      gold[i] = gold.data() + i;
    }

    // bulk push and pop
    auto first = gold.data();
    queue.bulk_push(first, N);
    REQUIRE(queue.size() == N);
    for(size_t i=0; i<N; i++) {
      REQUIRE(queue.pop() == gold[N - i - 1]);
    }
    REQUIRE(queue.empty() == true);

    // bulk push and steal
    first = gold.data();
    queue.bulk_push(first, N);
    REQUIRE(queue.size() == N);
    for(size_t i=0; i<N; i++) {
      REQUIRE(queue.steal() == gold[i]);
    }
    REQUIRE(queue.empty() == true);

    // concurrent
    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> stolens(M);

    for(size_t i=0; i<M; ++i) {
      threads.emplace_back([&, i](){
        while(consumed != N) {
          auto ptr = queue.steal();
          if(ptr != nullptr) {
            stolens[i].push_back(ptr);
            consumed.fetch_add(1, std::memory_order_relaxed);
          }
        }
        REQUIRE(queue.steal() == nullptr);
      });
    }

    first = gold.data();
    queue.bulk_push(first, N);

    std::vector<void*> items;
    while(consumed != N) {
      auto ptr = queue.pop();
      if(ptr != nullptr) {
        items.push_back(ptr);
        consumed.fetch_add(1, std::memory_order_relaxed);
      }
    }
    REQUIRE(queue.steal() == nullptr);
    REQUIRE(queue.pop()   == nullptr);
    REQUIRE(queue.empty());

    for(auto& thread : threads) thread.join();

    for(size_t i=0; i<M; ++i) {
      for(auto s : stolens[i]) items.push_back(s);
    }

    std::sort(items.begin(), items.end());
    std::sort(gold.begin(),  gold.end());
    REQUIRE(items.size() == N);
    REQUIRE(items == gold);
  }
}

TEST_CASE("UnboundedWSQ.1Consumer.BulkPush" * doctest::timeout(300)) {
  unbounded_wsq_n_consumers_bulk_push(1);
}

TEST_CASE("UnboundedWSQ.2Consumers.BulkPush" * doctest::timeout(300)) {
  unbounded_wsq_n_consumers_bulk_push(2);
}

TEST_CASE("UnboundedWSQ.3Consumers.BulkPush" * doctest::timeout(300)) {
  unbounded_wsq_n_consumers_bulk_push(3);
}

TEST_CASE("UnboundedWSQ.4Consumers.BulkPush" * doctest::timeout(300)) {
  unbounded_wsq_n_consumers_bulk_push(4);
}

TEST_CASE("UnboundedWSQ.5Consumers.BulkPush" * doctest::timeout(300)) {
  unbounded_wsq_n_consumers_bulk_push(5);
}

TEST_CASE("UnboundedWSQ.6Consumers.BulkPush" * doctest::timeout(300)) {
  unbounded_wsq_n_consumers_bulk_push(6);
}

TEST_CASE("UnboundedWSQ.7Consumers.BulkPush" * doctest::timeout(300)) {
  unbounded_wsq_n_consumers_bulk_push(7);
}

TEST_CASE("UnboundedWSQ.8Consumers.BulkPush" * doctest::timeout(300)) {
  unbounded_wsq_n_consumers_bulk_push(8);
}
