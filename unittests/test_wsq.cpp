#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
//#include <taskflow/utility/mpmc.hpp>
#include <taskflow/taskflow.hpp>


// ============================================================================
// BoundedWSQ Test
// ============================================================================

// Procedure: test_wsq_owner
template<size_t LogSize>
void bounded_tsq_owner() {

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

  // test steal
  size_t dummy1, dummy2;

  REQUIRE(queue.try_push(&dummy1) == true);
  REQUIRE(queue.try_push(&dummy2) == true);

  size_t num_empty_steals = 1234;
  REQUIRE(queue.steal_with_feedback(num_empty_steals) == &dummy1);
  REQUIRE(num_empty_steals == 0);
  
  num_empty_steals = 101;
  REQUIRE(queue.steal_with_feedback(num_empty_steals) == &dummy2);
  REQUIRE(num_empty_steals == 0);
  
  REQUIRE(queue.steal_with_feedback(num_empty_steals) == nullptr);
  REQUIRE(num_empty_steals == 1);
  
  REQUIRE(queue.steal_with_feedback(num_empty_steals) == nullptr);
  REQUIRE(num_empty_steals == 2);
  
  REQUIRE(queue.steal_with_feedback(num_empty_steals) == nullptr);
  REQUIRE(queue.steal_with_feedback(num_empty_steals) == nullptr);
  REQUIRE(num_empty_steals == 4);
  
  REQUIRE(queue.try_push(&dummy1) == true);
  REQUIRE(queue.try_push(&dummy2) == true);
  REQUIRE(queue.steal() == &dummy1);
  REQUIRE(num_empty_steals == 4);
  REQUIRE(queue.steal_with_feedback(num_empty_steals) == &dummy2);
  REQUIRE(num_empty_steals == 0);
  
  REQUIRE(queue.steal_with_feedback(num_empty_steals) == nullptr);
  REQUIRE(num_empty_steals == 1);
  
  REQUIRE(queue.steal_with_feedback(num_empty_steals) == nullptr);
  REQUIRE(num_empty_steals == 2);
}

TEST_CASE("BoundedWSQ.Owner.LogSize=2" * doctest::timeout(300)) {
  bounded_tsq_owner<2>();
}

TEST_CASE("BoundedWSQ.Owner.LogSize=3" * doctest::timeout(300)) {
  bounded_tsq_owner<3>();
}

TEST_CASE("BoundedWSQ.Owner.LogSize=4" * doctest::timeout(300)) {
  bounded_tsq_owner<4>();
}

TEST_CASE("BoundedWSQ.Owner.LogSize=5" * doctest::timeout(300)) {
  bounded_tsq_owner<5>();
}

TEST_CASE("BoundedWSQ.Owner.LogSize=6" * doctest::timeout(300)) {
  bounded_tsq_owner<6>();
}

TEST_CASE("BoundedWSQ.Owner.LogSize=7" * doctest::timeout(300)) {
  bounded_tsq_owner<7>();
}

TEST_CASE("BoundedWSQ.Owner.LogSize=8" * doctest::timeout(300)) {
  bounded_tsq_owner<8>();
}

TEST_CASE("BoundedWSQ.Owner.LogSize=9" * doctest::timeout(300)) {
  bounded_tsq_owner<9>();
}

TEST_CASE("BoundedWSQ.Owner.LogSize=10" * doctest::timeout(300)) {
  bounded_tsq_owner<10>();
}


// ============================================================================
// UnboundedWSQ Test
// ============================================================================

TEST_CASE("UnboundedWSQ.Resize") {
  tf::UnboundedWSQ<void*> queue(1);
  REQUIRE(queue.capacity() == 2);
  
  std::vector<void*> data(2048);

  // insert an element
  auto first = data.data();
  queue.bulk_push(first, 1);
  REQUIRE(queue.size() == 1);
  REQUIRE(queue.capacity() == 2);
  
  // insert 2 elements
  first = data.data();
  queue.bulk_push(first, 2);
  REQUIRE(queue.size() == 3);
  REQUIRE(queue.capacity() == 4);
  
  // insert 10 elements
  first = data.data();
  queue.bulk_push(first, 10);
  REQUIRE(queue.size() == 13);
  REQUIRE(queue.capacity() == 16);
  
  // insert 1200 elements
  first = data.data();
  queue.bulk_push(first, 1200);
  REQUIRE(queue.size() == 1213);
  REQUIRE(queue.capacity() == 2048);
  
  // remove all elements
  for(size_t i=0; i<1213; ++i) {
    REQUIRE(queue.size() == 1213 - i);
    queue.pop();
  }
  REQUIRE(queue.empty() == true);

  // insert an element
  first = data.data();
  queue.bulk_push(first, 1);
  REQUIRE(queue.size() == 1);
  REQUIRE(queue.capacity() == 2048);
  
  // insert 2 elements
  first = data.data();
  queue.bulk_push(first, 2);
  REQUIRE(queue.size() == 3);
  REQUIRE(queue.capacity() == 2048);
  
  // insert 10 elements
  first = data.data();
  queue.bulk_push(first, 10);
  REQUIRE(queue.size() == 13);
  REQUIRE(queue.capacity() == 2048);
  
  // insert 1200 elements
  first = data.data();
  queue.bulk_push(first, 1200);
  REQUIRE(queue.size() == 1213);
  REQUIRE(queue.capacity() == 2048);
}

// Procedure: unbounded_tsq_owner
void unbounded_tsq_owner() {
    
  tf::UnboundedWSQ<void*> queue;
  std::vector<void*> gold;

  for(size_t N=1; N<=777777; N=N*2+1) {

    gold.resize(N);
    REQUIRE(queue.empty());

    // push and pop
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

    // push and steal
    for(size_t i=0; i<N; ++i) {
      queue.push(gold[i]);
    }
    // i starts from 1 to avoid cache effect
    for(size_t i=0; i<N; ++i) {
      auto ptr = queue.steal();
      REQUIRE(ptr != nullptr);
      REQUIRE(gold[i] == ptr);
    }
  }
}


TEST_CASE("UnboundedTSQ.Owner" * doctest::timeout(300)) {
  unbounded_tsq_owner();
}

// ----------------------------------------------------------------------------
// Bounded Work-stealing Queue Multiple Consumers Test
// ----------------------------------------------------------------------------

// Procedure: bounded_tsq_n_consumers
void bounded_tsq_n_consumers(size_t M) {
    
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

    // thieves
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

    // master thread
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
    REQUIRE(queue.pop() == nullptr);
    REQUIRE(queue.empty());

    // join thieves
    for(auto& thread : threads) thread.join();

    // merge items
    for(size_t i=0; i<M; ++i) {
      for(auto s : stolens[i]) {
        items.push_back(s);
      }
    }

    std::sort(items.begin(), items.end());
    std::sort(gold.begin(), gold.end());

    REQUIRE(items.size() == N);
    REQUIRE(items == gold);
  }
}

TEST_CASE("BoundedTSQ.1Consumer" * doctest::timeout(300)) {
  bounded_tsq_n_consumers(1);
}

TEST_CASE("BoundedTSQ.2Consumers" * doctest::timeout(300)) {
  bounded_tsq_n_consumers(2);
}

TEST_CASE("BoundedTSQ.3Consumers" * doctest::timeout(300)) {
  bounded_tsq_n_consumers(3);
}

TEST_CASE("BoundedTSQ.4Consumers" * doctest::timeout(300)) {
  bounded_tsq_n_consumers(4);
}

TEST_CASE("BoundedTSQ.5Consumers" * doctest::timeout(300)) {
  bounded_tsq_n_consumers(5);
}

TEST_CASE("BoundedTSQ.6Consumers" * doctest::timeout(300)) {
  bounded_tsq_n_consumers(6);
}

TEST_CASE("BoundedTSQ.7Consumers" * doctest::timeout(300)) {
  bounded_tsq_n_consumers(7);
}

TEST_CASE("BoundedTSQ.8Consumers" * doctest::timeout(300)) {
  bounded_tsq_n_consumers(8);
}

// Procedure: bounded_tsq_n_consumers_bulk_push
void bounded_tsq_n_consumers_bulk_push(size_t M) {
    
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

    // master bulk push and pop
    size_t size = queue.size();
    size_t capacity = queue.capacity();
    REQUIRE((size == 0 && capacity > 0));
    const size_t num_pushable_elements = std::min(capacity, N);
    
    auto first = gold.data();
    REQUIRE(num_pushable_elements == queue.try_bulk_push(first, N));
    REQUIRE(queue.size() == num_pushable_elements);
    for(size_t i=0; i<num_pushable_elements; i++) {
      REQUIRE(queue.pop() == gold[num_pushable_elements - i - 1]);
    }
    REQUIRE(queue.empty() == true);

    // master bulk push and steal
    first = gold.data();
    REQUIRE(num_pushable_elements == queue.try_bulk_push(first, N));
    REQUIRE(queue.size() == num_pushable_elements);
    for(size_t i=0; i<num_pushable_elements; i++) {
      REQUIRE(queue.steal() == gold[i]);
    }
    REQUIRE(queue.empty() == true);
    

    // thieves
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

    // master thread
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
    REQUIRE(queue.pop() == nullptr);
    REQUIRE(queue.empty());

    // join thieves
    for(auto& thread : threads) thread.join();

    // merge items
    for(size_t i=0; i<M; ++i) {
      for(auto s : stolens[i]) {
        items.push_back(s);
      }
    }

    std::sort(items.begin(), items.end());
    std::sort(gold.begin(), gold.end());

    REQUIRE(items.size() == N);
    REQUIRE(items == gold);
  }
}

TEST_CASE("BoundedTSQ.1Consumer.BulkPush" * doctest::timeout(300)) {
  bounded_tsq_n_consumers_bulk_push(1);
}

TEST_CASE("BoundedTSQ.2Consumers.BulkPush" * doctest::timeout(300)) {
  bounded_tsq_n_consumers_bulk_push(2);
}

TEST_CASE("BoundedTSQ.3Consumers.BulkPush" * doctest::timeout(300)) {
  bounded_tsq_n_consumers_bulk_push(3);
}

TEST_CASE("BoundedTSQ.4Consumers.BulkPush" * doctest::timeout(300)) {
  bounded_tsq_n_consumers_bulk_push(4);
}

TEST_CASE("BoundedTSQ.5Consumers.BulkPush" * doctest::timeout(300)) {
  bounded_tsq_n_consumers_bulk_push(5);
}

TEST_CASE("BoundedTSQ.6Consumers.BulkPush" * doctest::timeout(300)) {
  bounded_tsq_n_consumers_bulk_push(6);
}

TEST_CASE("BoundedTSQ.7Consumers.BulkPush" * doctest::timeout(300)) {
  bounded_tsq_n_consumers_bulk_push(7);
}

TEST_CASE("BoundedTSQ.8Consumers.BulkPush" * doctest::timeout(300)) {
  bounded_tsq_n_consumers_bulk_push(8);
}

// ----------------------------------------------------------------------------
// Testcase: BoundedWSQ ValueType test
// ----------------------------------------------------------------------------

TEST_CASE("BoundedWSQ.ValueType") { 
  tf::BoundedWSQ<void*> Q1;
  tf::BoundedWSQ<int> Q2;

  auto empty1 = Q1.empty_value();
  auto empty2 = Q2.empty_value();

  static_assert(std::is_same_v<decltype(empty1), void*>);
  static_assert(std::is_same_v<decltype(empty2), std::optional<int>>);

  REQUIRE(empty1 == nullptr);
  REQUIRE(empty2 == std::nullopt);

  auto v = Q2.pop();
  REQUIRE(v == std::nullopt);

  Q2.try_push(1);
  Q2.try_push(2);
  Q2.try_push(3);
  Q2.try_push(4);

  REQUIRE(Q2.pop() == 4);
  REQUIRE(Q2.pop() == 3);
  REQUIRE(Q2.pop() == 2);
  REQUIRE(Q2.pop() == 1);
  REQUIRE(Q2.pop() == std::nullopt);
   
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
// Testcase: UnboundedTSQ Multiple Consumers Test
// ----------------------------------------------------------------------------

// Procedure: unbounded_tsq_n_consumers
void unbounded_tsq_n_consumers(size_t M) {
    
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
    size_t size = queue.size();
    size_t capacity = queue.capacity();
    REQUIRE((size == 0 && capacity > 0));
    
    for(size_t i=0; i<N; ++i) {
      queue.push(gold.data() + i);
    }
    REQUIRE(queue.size() == N);
    for(size_t i=0; i<N; i++) {
      REQUIRE(queue.pop() == gold[N - i - 1]);
    }
    REQUIRE(queue.empty() == true);

    // master bulk push and steal
    for(size_t i=0; i<N; ++i) {
      queue.push(gold.data() + i);
    }
    REQUIRE(queue.size() == N);
    for(size_t i=0; i<N; i++) {
      REQUIRE(queue.steal() == gold[i]);
    }
    REQUIRE(queue.empty() == true);

    // thieves
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

    // master thread
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
    REQUIRE(queue.pop() == nullptr);
    REQUIRE(queue.empty());

    // join thieves
    for(auto& thread : threads) thread.join();

    // merge items
    for(size_t i=0; i<M; ++i) {
      for(auto s : stolens[i]) {
        items.push_back(s);
      }
    }

    std::sort(items.begin(), items.end());
    std::sort(gold.begin(), gold.end());

    REQUIRE(items.size() == N);
    REQUIRE(items == gold);
  }

}

TEST_CASE("UnboundedTSQ.1Consumer" * doctest::timeout(300)) {
  unbounded_tsq_n_consumers(1);
}

TEST_CASE("UnboundedTSQ.2Consumers" * doctest::timeout(300)) {
  unbounded_tsq_n_consumers(2);
}

TEST_CASE("UnboundedTSQ.3Consumers" * doctest::timeout(300)) {
  unbounded_tsq_n_consumers(3);
}

TEST_CASE("UnboundedTSQ.4Consumers" * doctest::timeout(300)) {
  unbounded_tsq_n_consumers(4);
}

TEST_CASE("UnboundedTSQ.5Consumers" * doctest::timeout(300)) {
  unbounded_tsq_n_consumers(5);
}

TEST_CASE("UnboundedTSQ.6Consumers" * doctest::timeout(300)) {
  unbounded_tsq_n_consumers(6);
}

TEST_CASE("UnboundedTSQ.7Consumers" * doctest::timeout(300)) {
  unbounded_tsq_n_consumers(7);
}

TEST_CASE("UnboundedTSQ.8Consumers" * doctest::timeout(300)) {
  unbounded_tsq_n_consumers(8);
}

// Procedure: unbounded_tsq_n_consumers_bulk_push
void unbounded_tsq_n_consumers_bulk_push(size_t M) {
    
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
    
    // master bulk push and pop
    size_t size = queue.size();
    size_t capacity = queue.capacity();
    REQUIRE((size == 0 && capacity > 0));

    auto first = gold.data();
    queue.bulk_push(first, N);
    REQUIRE(queue.size() == N);
    for(size_t i=0; i<N; i++) {
      REQUIRE(queue.pop() == gold[N - i - 1]);
    }
    REQUIRE(queue.empty() == true);

    // master bulk push and steal
    first = gold.data();
    queue.bulk_push(first, N);
    REQUIRE(queue.size() == N);
    for(size_t i=0; i<N; i++) {
      REQUIRE(queue.steal() == gold[i]);
    }
    REQUIRE(queue.empty() == true);

    // thieves
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

    // master thread
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
    REQUIRE(queue.pop() == nullptr);
    REQUIRE(queue.empty());

    // join thieves
    for(auto& thread : threads) thread.join();

    // merge items
    for(size_t i=0; i<M; ++i) {
      for(auto s : stolens[i]) {
        items.push_back(s);
      }
    }

    std::sort(items.begin(), items.end());
    std::sort(gold.begin(), gold.end());

    REQUIRE(items.size() == N);
    REQUIRE(items == gold);
  }

}

TEST_CASE("UnboundedTSQ.1Consumer.BulkPush" * doctest::timeout(300)) {
  unbounded_tsq_n_consumers_bulk_push(1);
}

TEST_CASE("UnboundedTSQ.2Consumers.BulkPush" * doctest::timeout(300)) {
  unbounded_tsq_n_consumers_bulk_push(2);
}

TEST_CASE("UnboundedTSQ.3Consumers.BulkPush" * doctest::timeout(300)) {
  unbounded_tsq_n_consumers_bulk_push(3);
}

TEST_CASE("UnboundedTSQ.4Consumers.BulkPush" * doctest::timeout(300)) {
  unbounded_tsq_n_consumers_bulk_push(4);
}

TEST_CASE("UnboundedTSQ.5Consumers.BulkPush" * doctest::timeout(300)) {
  unbounded_tsq_n_consumers_bulk_push(5);
}

TEST_CASE("UnboundedTSQ.6Consumers.BulkPush" * doctest::timeout(300)) {
  unbounded_tsq_n_consumers_bulk_push(6);
}

TEST_CASE("UnboundedTSQ.7Consumers.BulkPush" * doctest::timeout(300)) {
  unbounded_tsq_n_consumers_bulk_push(7);
}

TEST_CASE("UnboundedTSQ.8Consumers.BulkPush" * doctest::timeout(300)) {
  unbounded_tsq_n_consumers_bulk_push(8);
}

// ----------------------------------------------------------------------------
// Testcase: UnboundedWSQ ValueType test
// ----------------------------------------------------------------------------

TEST_CASE("UnboundedWSQ.ValueType") { 

  tf::UnboundedWSQ<void*> Q1;
  tf::UnboundedWSQ<int> Q2;

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

  REQUIRE(Q2.pop() == 4);
  REQUIRE(Q2.pop() == 3);
  REQUIRE(Q2.pop() == 2);
  REQUIRE(Q2.pop() == 1);
  REQUIRE(Q2.pop() == std::nullopt);
   
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

/*
// ----------------------------------------------------------------------------
// BoundedMPMC
// ----------------------------------------------------------------------------

template <typename T, size_t LogSize>
void mpmc_basics() {

  tf::MPMC<T, LogSize> mpmc;
  size_t N = (1<<LogSize);
  std::vector<T> data(N+1, -1);

  REQUIRE(mpmc.capacity() == N);

  REQUIRE(mpmc.empty() == true);
  REQUIRE(mpmc.try_dequeue() == std::nullopt);

  for(size_t i=0; i<N; i++) {
    REQUIRE(mpmc.try_enqueue(data[i]) == true);
  }

  REQUIRE(mpmc.try_enqueue(data[N]) == false);
  REQUIRE(mpmc.empty() == false);

  for(size_t i=0; i<N; i++) {
    REQUIRE(mpmc.try_dequeue() == data[i]);
  }

  REQUIRE(mpmc.empty() == true); 
  REQUIRE(mpmc.try_dequeue() == std::nullopt);

  for(size_t i=0; i<N; i++) {
    mpmc.enqueue(data[i]);
  }
  REQUIRE(mpmc.try_enqueue(data[N]) == false);
  
  for(size_t i=0; i<N; i++) {
    REQUIRE(mpmc.empty() == false);
    REQUIRE(mpmc.try_dequeue() == data[i]);
  }

  REQUIRE(mpmc.empty() == true); 
  REQUIRE(mpmc.try_dequeue() == std::nullopt);
}

TEST_CASE("BoundedMPMC.Basics.LogSize=1") {
  mpmc_basics<int, 1>();
}

TEST_CASE("BoundedMPMC.Basics.LogSize=2") {
  mpmc_basics<int, 2>();
}

TEST_CASE("BoundedMPMC.Basics.LogSize=3") {
  mpmc_basics<int, 3>();
}

TEST_CASE("BoundedMPMC.Basics.LogSize=4") {
  mpmc_basics<int, 4>();
}

TEST_CASE("BoundedMPMC.Basics.LogSize=5") {
  mpmc_basics<int, 5>();
}

TEST_CASE("BoundedMPMC.Basics.LogSize=6") {
  mpmc_basics<int, 6>();
}

TEST_CASE("BoundedMPMC.Basics.LogSize=7") {
  mpmc_basics<int, 7>();
}

TEST_CASE("BoundedMPMC.Basics.LogSize=8") {
  mpmc_basics<int, 8>();
}

TEST_CASE("BoundedMPMC.Basics.LogSize=9") {
  mpmc_basics<int, 9>();
}

TEST_CASE("BoundedMPMC.Basics.LogSize=10") {
  mpmc_basics<int, 10>();
}

// mpmc
template <typename T, size_t LogSize>
void mpmc(unsigned num_producers, unsigned num_consumers) {

  const int N = 6543;

  std::atomic<int> pcnt(0), ccnt(0), ans(0);
  std::vector<std::thread> threads;

  tf::MPMC<T, LogSize> mpmc;

  for(unsigned i=0; i<num_consumers; i++) {
    threads.emplace_back([&](){
      while(ccnt.load(std::memory_order_relaxed) != N) {
        if(auto item = mpmc.try_dequeue(); item) {
          ans.fetch_add(item.value(), std::memory_order_relaxed);
          ccnt.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  for(unsigned i=0; i<num_producers; i++) {
    threads.emplace_back([&](){
      while(true) {
        auto v = pcnt.fetch_add(1, std::memory_order_relaxed);
        if(v >= N) {
          break;
        }
        mpmc.enqueue(v);
      }
    });
  }

  for(auto & thread : threads) {
    thread.join();
  }

  REQUIRE(ans.load() == (((N-1)*N) >> 1));
}

TEST_CASE("BoundedMPMC.1C1P") {
  mpmc<int, 1>(1, 1);
  mpmc<int, 10>(1, 1);
}

TEST_CASE("BoundedMPMC.1C2P") {
  mpmc<int, 1>(1, 2);
  mpmc<int, 10>(1, 2);
}

TEST_CASE("BoundedMPMC.1C3P") {
  mpmc<int, 1>(1, 3);
  mpmc<int, 10>(1, 3);
}

TEST_CASE("BoundedMPMC.1C4P") {
  mpmc<int, 1>(1, 4);
  mpmc<int, 10>(1, 4);
}

TEST_CASE("BoundedMPMC.2C1P") {
  mpmc<int, 1>(2, 1);
  mpmc<int, 10>(2, 1);
}

TEST_CASE("BoundedMPMC.2C2P") {
  mpmc<int, 1>(2, 2);
  mpmc<int, 10>(2, 2);
}

TEST_CASE("BoundedMPMC.2C3P") {
  mpmc<int, 1>(2, 3);
  mpmc<int, 10>(2, 3);
}

TEST_CASE("BoundedMPMC.2C4P") {
  mpmc<int, 1>(2, 4);
  mpmc<int, 10>(2, 4);
}

TEST_CASE("BoundedMPMC.3C1P") {
  mpmc<int, 1>(3, 1);
  mpmc<int, 10>(3, 1);
}

TEST_CASE("BoundedMPMC.3C2P") {
  mpmc<int, 1>(3, 2);
  mpmc<int, 10>(3, 2);
}

TEST_CASE("BoundedMPMC.3C3P") {
  mpmc<int, 1>(3, 3);
  mpmc<int, 10>(3, 3);
}

TEST_CASE("BoundedMPMC.3C4P") {
  mpmc<int, 1>(3, 4);
  mpmc<int, 10>(3, 4);
}

TEST_CASE("BoundedMPMC.4C1P") {
  mpmc<int, 1>(4, 1);
  mpmc<int, 10>(4, 1);
}

TEST_CASE("BoundedMPMC.4C2P") {
  mpmc<int, 1>(4, 2);
  mpmc<int, 10>(4, 2);
}

TEST_CASE("BoundedMPMC.4C3P") {
  mpmc<int, 1>(4, 3);
  mpmc<int, 10>(4, 3);
}

TEST_CASE("BoundedMPMC.4C4P") {
  mpmc<int, 1>(4, 4);
  mpmc<int, 10>(4, 4);
}

// ------------------------------------------------------------------------------------------------
// BoundedMPMC Specialization on Pointer Type
// ------------------------------------------------------------------------------------------------

template <typename T, size_t LogSize>
void mpmc_pointer_basics() {

  tf::MPMC<T, LogSize> mpmc;
  size_t N = (1<<LogSize);
  std::vector<std::remove_pointer_t<T>> data(N+1);

  REQUIRE(mpmc.capacity() == N);

  REQUIRE(mpmc.empty() == true);
  REQUIRE(mpmc.try_dequeue() == nullptr);

  for(size_t i=0; i<N; i++) {
    REQUIRE(mpmc.try_enqueue(&data[i]) == true);
  }

  REQUIRE(mpmc.try_enqueue(&data[N]) == false);
  REQUIRE(mpmc.empty() == false);

  for(size_t i=0; i<N; i++) {
    REQUIRE(mpmc.try_dequeue() == &data[i]);
  }

  REQUIRE(mpmc.empty() == true); 
  REQUIRE(mpmc.try_dequeue() == nullptr);

  for(size_t i=0; i<N; i++) {
    mpmc.enqueue(&data[i]);
  }
  REQUIRE(mpmc.try_enqueue(&data[N]) == false);
  
  for(size_t i=0; i<N; i++) {
    REQUIRE(mpmc.empty() == false);
    REQUIRE(mpmc.try_dequeue() == &data[i]);
  }

  REQUIRE(mpmc.empty() == true); 
  REQUIRE(mpmc.try_dequeue() == nullptr);
}

TEST_CASE("BoundedMPMC.Pointer.Basics.LogSize=1") {
  mpmc_pointer_basics<int*, 1>();
}

TEST_CASE("BoundedMPMC.Pointer.Basics.LogSize=2") {
  mpmc_pointer_basics<int*, 2>();
}

TEST_CASE("BoundedMPMC.Pointer.Basics.LogSize=3") {
  mpmc_pointer_basics<int*, 3>();
}

TEST_CASE("BoundedMPMC.Pointer.Basics.LogSize=4") {
  mpmc_pointer_basics<int*, 4>();
}

TEST_CASE("BoundedMPMC.Pointer.Basics.LogSize=5") {
  mpmc_pointer_basics<int*, 5>();
}

TEST_CASE("BoundedMPMC.Pointer.Basics.LogSize=6") {
  mpmc_pointer_basics<int*, 6>();
}

TEST_CASE("BoundedMPMC.Pointer.Basics.LogSize=7") {
  mpmc_pointer_basics<int*, 7>();
}

TEST_CASE("BoundedMPMC.Pointer.Basics.LogSize=8") {
  mpmc_pointer_basics<int*, 8>();
}

TEST_CASE("BoundedMPMC.Pointer.Basics.LogSize=9") {
  mpmc_pointer_basics<int*, 9>();
}

TEST_CASE("BoundedMPMC.Pointer.Basics.LogSize=10") {
  mpmc_pointer_basics<int*, 10>();
}
*/




