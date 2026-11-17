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
// BoundedWSQ ValueType test
// ----------------------------------------------------------------------------

TEST_CASE("BoundedWSQ.ValueType") {
  tf::BoundedWSQ<void*> Q1;
  tf::BoundedWSQ<int>   Q2;

  // empty_value(): nullptr for pointer types, nullopt for non-pointer
  auto empty1 = Q1.empty_value();
  auto empty2 = Q2.empty_value();

  static_assert(std::is_same_v<decltype(empty1), void*>);
  static_assert(std::is_same_v<decltype(empty2), std::optional<int>>);

  REQUIRE(empty1 == nullptr);
  REQUIRE(empty2 == std::nullopt);

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
// UnboundedWSQ ValueType test
// ----------------------------------------------------------------------------

TEST_CASE("UnboundedWSQ.ValueType") {

  tf::UnboundedWSQ<void*> Q1;
  tf::UnboundedWSQ<int>   Q2;

  auto empty1 = Q1.empty_value();
  auto empty2 = Q2.empty_value();

  static_assert(std::is_same_v<decltype(empty1), void*>);
  static_assert(std::is_same_v<decltype(empty2), std::optional<int>>);

  REQUIRE(empty1 == nullptr);
  REQUIRE(empty2 == std::nullopt);

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

// ============================================================================
// BoundedPriorityWSQ Tests
// ============================================================================

struct PriorityItem {
  size_t prio;
  int    id;
  size_t priority() const { return prio; }
};

struct PriorityItemPtrFn {
  size_t operator()(const PriorityItem* item) const {
    return item->priority();
  }
};

// ============================================================================
// BoundedPriorityWSQ — Owner-thread basics (single-threaded)
// ============================================================================


// Pop respects priority order
TEST_CASE("BoundedPriorityWSQ.Owner.PopPriorityOrder") {

  using Q = tf::BoundedWSQ<PriorityItem*>;
  tf::BoundedPriorityWSQ<Q, 3, PriorityItemPtrFn> pq;

  PriorityItem items[] = {
    {2, 0}, {1, 1}, {0, 2}, {2, 3}, {0, 4}, {1, 5}
  };

  for(auto& item : items) {
    pq.try_push(&item);
  }

  // priority-0 items first (LIFO within level): id=4 then id=2
  auto r = pq.pop();
  REQUIRE(r != nullptr);
  REQUIRE(r->prio == 0);
  REQUIRE(r->id   == 4);

  r = pq.pop();
  REQUIRE(r != nullptr);
  REQUIRE(r->prio == 0);
  REQUIRE(r->id   == 2);

  // priority-1 items next (LIFO): id=5 then id=1
  r = pq.pop();
  REQUIRE(r != nullptr);
  REQUIRE(r->prio == 1);
  REQUIRE(r->id   == 5);

  r = pq.pop();
  REQUIRE(r != nullptr);
  REQUIRE(r->prio == 1);
  REQUIRE(r->id   == 1);

  // priority-2 items last (LIFO): id=3 then id=0
  r = pq.pop();
  REQUIRE(r != nullptr);
  REQUIRE(r->prio == 2);
  REQUIRE(r->id   == 3);

  r = pq.pop();
  REQUIRE(r != nullptr);
  REQUIRE(r->prio == 2);
  REQUIRE(r->id   == 0);

  REQUIRE(pq.pop() == nullptr);
  REQUIRE(pq.empty());
}

// Steal respects priority order and is FIFO within a level
TEST_CASE("BoundedPriorityWSQ.Owner.StealPriorityOrder") {

  using Q = tf::BoundedWSQ<PriorityItem*>;
  tf::BoundedPriorityWSQ<Q, 3, PriorityItemPtrFn> pq;

  PriorityItem items[] = {
    {2, 0}, {1, 1}, {0, 2}, {2, 3}, {0, 4}, {1, 5}
  };

  for(auto& item : items) {
    pq.try_push(&item);
  }

  // priority-0 items first (FIFO within level): id=2 then id=4
  auto r = pq.steal();
  REQUIRE(r != nullptr);
  REQUIRE(r->prio == 0);
  REQUIRE(r->id   == 2);

  r = pq.steal();
  REQUIRE(r != nullptr);
  REQUIRE(r->prio == 0);
  REQUIRE(r->id   == 4);

  // priority-1: FIFO -> id=1 then id=5
  r = pq.steal();
  REQUIRE(r != nullptr);
  REQUIRE(r->prio == 1);
  REQUIRE(r->id   == 1);

  r = pq.steal();
  REQUIRE(r != nullptr);
  REQUIRE(r->prio == 1);
  REQUIRE(r->id   == 5);

  // priority-2: FIFO -> id=0 then id=3
  r = pq.steal();
  REQUIRE(r != nullptr);
  REQUIRE(r->prio == 2);
  REQUIRE(r->id   == 0);

  r = pq.steal();
  REQUIRE(r != nullptr);
  REQUIRE(r->prio == 2);
  REQUIRE(r->id   == 3);

  REQUIRE(pq.steal() == nullptr);
  REQUIRE(pq.empty());
}

// Push when sub-queue is full returns false; other levels still accept
TEST_CASE("BoundedPriorityWSQ.Owner.PushFull") {

  using Q = tf::BoundedWSQ<PriorityItem*, 2>;  // capacity = 4 per sub-queue
  tf::BoundedPriorityWSQ<Q, 2, PriorityItemPtrFn> pq;

  PriorityItem hi[5];
  for(int i = 0; i < 5; ++i) hi[i] = {0, i};

  PriorityItem lo = {1, 99};

  for(int i = 0; i < 4; ++i) {
    REQUIRE(pq.try_push(&hi[i]) == true);
  }
  // sub-queue 0 is full
  REQUIRE(pq.try_push(&hi[4]) == false);

  // sub-queue 1 still has room
  REQUIRE(pq.try_push(&lo) == true);
  REQUIRE(pq.size() == 5);
}

// Pop/steal on empty queue returns empty value
TEST_CASE("BoundedPriorityWSQ.Owner.EmptyReturns") {

  using Q = tf::BoundedWSQ<PriorityItem*>;
  tf::BoundedPriorityWSQ<Q, 3, PriorityItemPtrFn> pq;

  REQUIRE(pq.pop()   == nullptr);
  REQUIRE(pq.steal() == nullptr);
  REQUIRE(pq.empty());
  REQUIRE(pq.size()  == 0);
}

// ============================================================================
// BoundedPriorityWSQ — Priority starvation / interleaving
// ============================================================================

TEST_CASE("BoundedPriorityWSQ.PriorityOrdering.Pop") {

  using Q = tf::BoundedWSQ<PriorityItem*>;
  tf::BoundedPriorityWSQ<Q, 3, PriorityItemPtrFn> pq;

  // push in mixed order: low, high, mid, low, high, mid
  PriorityItem items[] = {
    {2, 0}, {0, 1}, {1, 2}, {2, 3}, {0, 4}, {1, 5},
    {2, 6}, {0, 7}, {1, 8}, {2, 9}, {0, 10}, {1, 11}
  };

  for(auto& item : items) {
    pq.try_push(&item);
  }

  // all priority-0 items must come out before any priority-1
  // all priority-1 items must come out before any priority-2
  size_t last_prio = 0;
  for(size_t i = 0; i < 12; ++i) {
    auto r = pq.pop();
    REQUIRE(r != nullptr);
    REQUIRE(r->prio >= last_prio);
    last_prio = r->prio;
  }
  REQUIRE(pq.pop() == nullptr);
}

TEST_CASE("BoundedPriorityWSQ.PriorityOrdering.Steal") {

  using Q = tf::BoundedWSQ<PriorityItem*>;
  tf::BoundedPriorityWSQ<Q, 3, PriorityItemPtrFn> pq;

  PriorityItem items[] = {
    {2, 0}, {0, 1}, {1, 2}, {2, 3}, {0, 4}, {1, 5},
    {2, 6}, {0, 7}, {1, 8}, {2, 9}, {0, 10}, {1, 11}
  };

  for(auto& item : items) {
    pq.try_push(&item);
  }

  size_t last_prio = 0;
  for(size_t i = 0; i < 12; ++i) {
    auto r = pq.steal();
    REQUIRE(r != nullptr);
    REQUIRE(r->prio >= last_prio);
    last_prio = r->prio;
  }
  REQUIRE(pq.steal() == nullptr);
}

// ============================================================================
// BoundedPriorityWSQ — try_bulk_push
// ============================================================================

// Contiguous same-priority run
TEST_CASE("BoundedPriorityWSQ.BulkPush.SamePriority") {

  using Q = tf::BoundedWSQ<PriorityItem*>;
  tf::BoundedPriorityWSQ<Q, 3, PriorityItemPtrFn> pq;

  std::vector<PriorityItem> items(8, {1, 0});
  for(int i = 0; i < 8; ++i) items[i].id = i;

  std::vector<PriorityItem*> ptrs(items.size());
  for(size_t i = 0; i < items.size(); ++i) ptrs[i] = &items[i];

  auto first = ptrs.data();
  size_t inserted = pq.try_bulk_push(first, ptrs.size());

  REQUIRE(inserted == 8);
  REQUIRE(pq[0].size() == 0);
  REQUIRE(pq[1].size() == 8);
  REQUIRE(pq[2].size() == 0);
}

// Mixed-priority runs
TEST_CASE("BoundedPriorityWSQ.BulkPush.MixedPriority") {

  using Q = tf::BoundedWSQ<PriorityItem*>;
  tf::BoundedPriorityWSQ<Q, 3, PriorityItemPtrFn> pq;

  // priorities: 0,0,0, 1,1, 2, 1,1,1,1, 2,2
  PriorityItem items[] = {
    {0,0},{0,1},{0,2}, {1,3},{1,4}, {2,5}, {1,6},{1,7},{1,8},{1,9}, {2,10},{2,11}
  };
  std::vector<PriorityItem*> ptrs(12);
  for(size_t i = 0; i < 12; ++i) ptrs[i] = &items[i];

  auto first = ptrs.data();
  size_t inserted = pq.try_bulk_push(first, ptrs.size());

  REQUIRE(inserted == 12);
  REQUIRE(pq[0].size() == 3);
  REQUIRE(pq[1].size() == 6);
  REQUIRE(pq[2].size() == 3);
}

// Partial insert when sub-queue fills up
TEST_CASE("BoundedPriorityWSQ.BulkPush.Partial") {

  using Q = tf::BoundedWSQ<PriorityItem*, 2>;  // capacity = 4
  tf::BoundedPriorityWSQ<Q, 2, PriorityItemPtrFn> pq;

  // pre-fill sub-queue 0 with 3 items, leaving room for 1 more
  PriorityItem prefill[3];
  for(int i = 0; i < 3; ++i) {
    prefill[i] = {0, i};
    pq.try_push(&prefill[i]);
  }
  REQUIRE(pq[0].size() == 3);

  // try to bulk-push 4 items at priority 0 — only 1 should fit
  PriorityItem batch[4];
  for(int i = 0; i < 4; ++i) batch[i] = {0, 100 + i};

  std::vector<PriorityItem*> ptrs(4);
  for(int i = 0; i < 4; ++i) ptrs[i] = &batch[i];

  auto first = ptrs.data();
  size_t inserted = pq.try_bulk_push(first, 4);

  REQUIRE(inserted == 1);
  REQUIRE(pq[0].size() == 4);
}

// Empty batch
TEST_CASE("BoundedPriorityWSQ.BulkPush.Empty") {

  using Q = tf::BoundedWSQ<PriorityItem*>;
  tf::BoundedPriorityWSQ<Q, 3, PriorityItemPtrFn> pq;

  PriorityItem* dummy = nullptr;
  auto first = &dummy;
  REQUIRE(pq.try_bulk_push(first, 0) == 0);
  REQUIRE(pq.empty());
}

// Bulk push then pop/steal verifies priority + LIFO/FIFO order
TEST_CASE("BoundedPriorityWSQ.BulkPush.PopOrder") {

  using Q = tf::BoundedWSQ<PriorityItem*>;
  tf::BoundedPriorityWSQ<Q, 3, PriorityItemPtrFn> pq;

  // priorities: 0,0, 1,1, 2,2
  PriorityItem items[] = {
    {0,0},{0,1}, {1,2},{1,3}, {2,4},{2,5}
  };
  std::vector<PriorityItem*> ptrs(6);
  for(size_t i = 0; i < 6; ++i) ptrs[i] = &items[i];

  auto first = ptrs.data();
  pq.try_bulk_push(first, 6);

  // pop is LIFO within priority level, highest priority first
  REQUIRE(pq.pop()->id == 1);
  REQUIRE(pq.pop()->id == 0);
  REQUIRE(pq.pop()->id == 3);
  REQUIRE(pq.pop()->id == 2);
  REQUIRE(pq.pop()->id == 5);
  REQUIRE(pq.pop()->id == 4);
  REQUIRE(pq.pop() == nullptr);
}

TEST_CASE("BoundedPriorityWSQ.BulkPush.StealOrder") {

  using Q = tf::BoundedWSQ<PriorityItem*>;
  tf::BoundedPriorityWSQ<Q, 3, PriorityItemPtrFn> pq;

  PriorityItem items[] = {
    {0,0},{0,1}, {1,2},{1,3}, {2,4},{2,5}
  };
  std::vector<PriorityItem*> ptrs(6);
  for(size_t i = 0; i < 6; ++i) ptrs[i] = &items[i];

  auto first = ptrs.data();
  pq.try_bulk_push(first, 6);

  // steal is FIFO within priority level, highest priority first
  REQUIRE(pq.steal()->id == 0);
  REQUIRE(pq.steal()->id == 1);
  REQUIRE(pq.steal()->id == 2);
  REQUIRE(pq.steal()->id == 3);
  REQUIRE(pq.steal()->id == 4);
  REQUIRE(pq.steal()->id == 5);
  REQUIRE(pq.steal() == nullptr);
}

// ============================================================================
// BoundedPriorityWSQ — Concurrent correctness (M thieves + 1 owner, try_push)
// ============================================================================

void priority_wsq_n_consumers(size_t M) {

  auto fn = [](PriorityItem* item) -> size_t { return item->priority(); };
  using Q = tf::BoundedWSQ<PriorityItem*>;
  using PQ = tf::BoundedPriorityWSQ<Q, 3, decltype(fn)>;

  PQ pq(fn);

  std::vector<PriorityItem> gold;
  std::atomic<size_t> consumed;

  for(size_t N = 1; N <= 88573; N = N * 3 + 1) {

    REQUIRE(pq.empty());

    gold.resize(N);
    consumed = 0;

    for(size_t i = 0; i < N; ++i) {
      gold[i] = {i % 3, static_cast<int>(i)};
    }

    std::vector<std::thread> threads;
    std::vector<std::vector<PriorityItem*>> stolens(M);

    for(size_t i = 0; i < M; ++i) {
      threads.emplace_back([&, i]() {
        while(consumed != N) {
          auto ptr = pq.steal();
          if(ptr != nullptr) {
            stolens[i].push_back(ptr);
            consumed.fetch_add(1, std::memory_order_relaxed);
          }
        }
        REQUIRE(pq.steal() == nullptr);
      });
    }

    for(size_t i = 0; i < N; ++i) {
      while(pq.try_push(&gold[i]) == false);
    }

    std::vector<PriorityItem*> items;
    while(consumed != N) {
      auto ptr = pq.pop();
      if(ptr != nullptr) {
        items.push_back(ptr);
        consumed.fetch_add(1, std::memory_order_relaxed);
      }
    }
    REQUIRE(pq.steal() == nullptr);
    REQUIRE(pq.pop()   == nullptr);
    REQUIRE(pq.empty());

    for(auto& thread : threads) thread.join();

    for(size_t i = 0; i < M; ++i) {
      for(auto s : stolens[i]) items.push_back(s);
    }

    std::sort(items.begin(), items.end());

    std::vector<PriorityItem*> expected(N);
    for(size_t i = 0; i < N; ++i) expected[i] = &gold[i];
    std::sort(expected.begin(), expected.end());

    REQUIRE(items.size() == N);
    REQUIRE(items == expected);
  }
}

TEST_CASE("BoundedPriorityWSQ.1Consumer" * doctest::timeout(300)) {
  priority_wsq_n_consumers(1);
}

TEST_CASE("BoundedPriorityWSQ.2Consumers" * doctest::timeout(300)) {
  priority_wsq_n_consumers(2);
}

TEST_CASE("BoundedPriorityWSQ.3Consumers" * doctest::timeout(300)) {
  priority_wsq_n_consumers(3);
}

TEST_CASE("BoundedPriorityWSQ.4Consumers" * doctest::timeout(300)) {
  priority_wsq_n_consumers(4);
}

TEST_CASE("BoundedPriorityWSQ.5Consumers" * doctest::timeout(300)) {
  priority_wsq_n_consumers(5);
}

TEST_CASE("BoundedPriorityWSQ.6Consumers" * doctest::timeout(300)) {
  priority_wsq_n_consumers(6);
}

TEST_CASE("BoundedPriorityWSQ.7Consumers" * doctest::timeout(300)) {
  priority_wsq_n_consumers(7);
}

TEST_CASE("BoundedPriorityWSQ.8Consumers" * doctest::timeout(300)) {
  priority_wsq_n_consumers(8);
}

TEST_CASE("BoundedPriorityWSQ.16Consumers" * doctest::timeout(300)) {
  priority_wsq_n_consumers(16);
}

TEST_CASE("BoundedPriorityWSQ.32Consumers" * doctest::timeout(300)) {
  priority_wsq_n_consumers(32);
}

// ============================================================================
// BoundedPriorityWSQ — Concurrent correctness with try_bulk_push
// ============================================================================

void priority_wsq_n_consumers_bulk_push(size_t M) {

  auto fn = [](PriorityItem* item) -> size_t { return item->priority(); };
  using Q = tf::BoundedWSQ<PriorityItem*>;
  using PQ = tf::BoundedPriorityWSQ<Q, 3, decltype(fn)>;

  PQ pq(fn);

  std::vector<PriorityItem> gold;
  std::atomic<size_t> consumed;

  for(size_t N = 1; N <= 88573; N = N * 3 + 1) {

    REQUIRE(pq.empty());

    gold.resize(N);
    consumed = 0;

    for(size_t i = 0; i < N; ++i) {
      gold[i] = {i % 3, static_cast<int>(i)};
    }

    // sequential: bulk push (capped at capacity) then pop
    {
      size_t cap = pq.capacity();
      size_t num_pushable = std::min(cap, N);

      std::vector<PriorityItem*> ptrs(N);
      for(size_t i = 0; i < N; ++i) ptrs[i] = &gold[i];

      auto first = ptrs.data();
      size_t inserted = pq.try_bulk_push(first, N);
      REQUIRE(inserted == num_pushable);
      REQUIRE(pq.size() == num_pushable);

      size_t last_prio = 0;
      for(size_t i = 0; i < num_pushable; ++i) {
        auto r = pq.pop();
        REQUIRE(r != nullptr);
        REQUIRE(r->prio >= last_prio);
        last_prio = r->prio;
      }
      REQUIRE(pq.empty());
    }

    // sequential: bulk push (capped at capacity) then steal
    {
      size_t cap = pq.capacity();
      size_t num_pushable = std::min(cap, N);

      std::vector<PriorityItem*> ptrs(N);
      for(size_t i = 0; i < N; ++i) ptrs[i] = &gold[i];

      auto first = ptrs.data();
      size_t inserted = pq.try_bulk_push(first, N);
      REQUIRE(inserted == num_pushable);
      REQUIRE(pq.size() == num_pushable);

      size_t last_prio = 0;
      for(size_t i = 0; i < num_pushable; ++i) {
        auto r = pq.steal();
        REQUIRE(r != nullptr);
        REQUIRE(r->prio >= last_prio);
        last_prio = r->prio;
      }
      REQUIRE(pq.empty());
    }

    // concurrent: bulk push + M thieves
    std::vector<std::thread> threads;
    std::vector<std::vector<PriorityItem*>> stolens(M);

    for(size_t i = 0; i < M; ++i) {
      threads.emplace_back([&, i]() {
        while(consumed != N) {
          auto ptr = pq.steal();
          if(ptr != nullptr) {
            stolens[i].push_back(ptr);
            consumed.fetch_add(1, std::memory_order_relaxed);
          }
        }
        REQUIRE(pq.steal() == nullptr);
      });
    }

    std::vector<PriorityItem*> ptrs(N);
    for(size_t i = 0; i < N; ++i) ptrs[i] = &gold[i];
    auto first = ptrs.data();
    size_t total_pushed = 0;
    while(total_pushed < N) {
      total_pushed += pq.try_bulk_push(first, N - total_pushed);
    }

    std::vector<PriorityItem*> items;
    while(consumed != N) {
      auto ptr = pq.pop();
      if(ptr != nullptr) {
        items.push_back(ptr);
        consumed.fetch_add(1, std::memory_order_relaxed);
      }
    }
    REQUIRE(pq.steal() == nullptr);
    REQUIRE(pq.pop()   == nullptr);
    REQUIRE(pq.empty());

    for(auto& thread : threads) thread.join();

    for(size_t i = 0; i < M; ++i) {
      for(auto s : stolens[i]) items.push_back(s);
    }

    std::sort(items.begin(), items.end());

    std::vector<PriorityItem*> expected(N);
    for(size_t i = 0; i < N; ++i) expected[i] = &gold[i];
    std::sort(expected.begin(), expected.end());

    REQUIRE(items.size() == N);
    REQUIRE(items == expected);
  }
}

TEST_CASE("BoundedPriorityWSQ.1Consumer.BulkPush" * doctest::timeout(300)) {
  priority_wsq_n_consumers_bulk_push(1);
}

TEST_CASE("BoundedPriorityWSQ.2Consumers.BulkPush" * doctest::timeout(300)) {
  priority_wsq_n_consumers_bulk_push(2);
}

TEST_CASE("BoundedPriorityWSQ.3Consumers.BulkPush" * doctest::timeout(300)) {
  priority_wsq_n_consumers_bulk_push(3);
}

TEST_CASE("BoundedPriorityWSQ.4Consumers.BulkPush" * doctest::timeout(300)) {
  priority_wsq_n_consumers_bulk_push(4);
}

TEST_CASE("BoundedPriorityWSQ.5Consumers.BulkPush" * doctest::timeout(300)) {
  priority_wsq_n_consumers_bulk_push(5);
}

TEST_CASE("BoundedPriorityWSQ.6Consumers.BulkPush" * doctest::timeout(300)) {
  priority_wsq_n_consumers_bulk_push(6);
}

TEST_CASE("BoundedPriorityWSQ.7Consumers.BulkPush" * doctest::timeout(300)) {
  priority_wsq_n_consumers_bulk_push(7);
}

TEST_CASE("BoundedPriorityWSQ.8Consumers.BulkPush" * doctest::timeout(300)) {
  priority_wsq_n_consumers_bulk_push(8);
}

TEST_CASE("BoundedPriorityWSQ.16Consumers.BulkPush" * doctest::timeout(300)) {
  priority_wsq_n_consumers_bulk_push(16);
}

TEST_CASE("BoundedPriorityWSQ.32Consumers.BulkPush" * doctest::timeout(300)) {
  priority_wsq_n_consumers_bulk_push(32);
}

// ============================================================================
// BoundedPriorityWSQ — Multiple MaxPriority values
// ============================================================================

// MaxPriority=1 degenerates to a plain WSQ
TEST_CASE("BoundedPriorityWSQ.MaxPriority1") {

  auto fn = [](PriorityItem*) -> size_t { return 0; };
  using Q = tf::BoundedWSQ<PriorityItem*>;
  tf::BoundedPriorityWSQ<Q, 1, decltype(fn)> pq(fn);

  PriorityItem items[] = {{0,0},{0,1},{0,2},{0,3}};

  for(auto& item : items) {
    pq.try_push(&item);
  }

  REQUIRE(pq.size() == 4);

  // pop is LIFO
  REQUIRE(pq.pop()->id == 3);
  REQUIRE(pq.pop()->id == 2);
  REQUIRE(pq.pop()->id == 1);
  REQUIRE(pq.pop()->id == 0);
  REQUIRE(pq.pop() == nullptr);
}

// MaxPriority=2
TEST_CASE("BoundedPriorityWSQ.MaxPriority2") {

  auto fn = [](PriorityItem* item) -> size_t { return item->priority(); };
  using Q = tf::BoundedWSQ<PriorityItem*>;
  tf::BoundedPriorityWSQ<Q, 2, decltype(fn)> pq(fn);

  PriorityItem items[] = {{1,0},{0,1},{1,2},{0,3}};

  for(auto& item : items) {
    pq.try_push(&item);
  }

  // pop: priority-0 first (LIFO), then priority-1 (LIFO)
  REQUIRE(pq.pop()->id == 3);
  REQUIRE(pq.pop()->id == 1);
  REQUIRE(pq.pop()->id == 2);
  REQUIRE(pq.pop()->id == 0);
  REQUIRE(pq.pop() == nullptr);

  // steal path
  for(auto& item : items) pq.try_push(&item);

  // steal: priority-0 first (FIFO), then priority-1 (FIFO)
  REQUIRE(pq.steal()->id == 1);
  REQUIRE(pq.steal()->id == 3);
  REQUIRE(pq.steal()->id == 0);
  REQUIRE(pq.steal()->id == 2);
  REQUIRE(pq.steal() == nullptr);
}

// MaxPriority=5 (larger value to stress the scan loop)
TEST_CASE("BoundedPriorityWSQ.MaxPriority5") {

  auto fn = [](PriorityItem* item) -> size_t { return item->priority(); };
  using Q = tf::BoundedWSQ<PriorityItem*>;
  tf::BoundedPriorityWSQ<Q, 5, decltype(fn)> pq(fn);

  PriorityItem items[] = {
    {4,0},{3,1},{2,2},{1,3},{0,4},
    {4,5},{3,6},{2,7},{1,8},{0,9}
  };

  for(auto& item : items) pq.try_push(&item);

  REQUIRE(pq.size() == 10);
  REQUIRE(pq.capacity() == pq[0].capacity() * 5);

  // pop must return in strict priority order
  size_t last_prio = 0;
  for(int i = 0; i < 10; ++i) {
    auto r = pq.pop();
    REQUIRE(r != nullptr);
    REQUIRE(r->prio >= last_prio);
    last_prio = r->prio;
  }
  REQUIRE(pq.pop() == nullptr);
}

// ============================================================================
// BoundedPriorityWSQ — Custom priority function
// ============================================================================

TEST_CASE("BoundedPriorityWSQ.CustomPriorityFn") {

  // reverse priority: higher id -> lower priority index (higher urgency)
  auto reverse_fn = [](PriorityItem* item) -> size_t {
    return (item->id < 5) ? 1 : 0;
  };

  using Q = tf::BoundedWSQ<PriorityItem*>;
  tf::BoundedPriorityWSQ<Q, 2, decltype(reverse_fn)> pq(reverse_fn);

  PriorityItem items[] = {
    {0, 0}, {0, 1}, {0, 7}, {0, 8}, {0, 3}, {0, 9}
  };

  for(auto& item : items) pq.try_push(&item);

  // items with id >= 5 go to sub-queue 0 (high priority): ids 7, 8, 9
  // items with id < 5 go to sub-queue 1 (low priority): ids 0, 1, 3
  REQUIRE(pq[0].size() == 3);
  REQUIRE(pq[1].size() == 3);

  // pop: sub-queue 0 first (LIFO): 9, 8, 7
  REQUIRE(pq.pop()->id == 9);
  REQUIRE(pq.pop()->id == 8);
  REQUIRE(pq.pop()->id == 7);

  // then sub-queue 1 (LIFO): 3, 1, 0
  REQUIRE(pq.pop()->id == 3);
  REQUIRE(pq.pop()->id == 1);
  REQUIRE(pq.pop()->id == 0);
  REQUIRE(pq.pop() == nullptr);
}