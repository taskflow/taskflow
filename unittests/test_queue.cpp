#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>


// ============================================================================
// BoundedTaskQueue Test
// ============================================================================

// Procedure: test_wsq_owner
template<size_t LogSize>
void bounded_tsq_owner() {

  tf::BoundedTaskQueue<size_t*, LogSize> queue;

  constexpr size_t N = (1 << LogSize) - 1;

  std::vector<size_t*> data;

  for(size_t k=0; k<LogSize*10; k++) {

    data.clear();

    REQUIRE(queue.empty() == true);

    for(size_t i=0; i<N; i++) {
      REQUIRE(queue.try_push(&i) == true);
      data.push_back(&i);
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
}

TEST_CASE("BoundedTaskQueue.Owner.LogSize=2" * doctest::timeout(300)) {
  bounded_tsq_owner<2>();
}

TEST_CASE("BoundedTaskQueue.Owner.LogSize=3" * doctest::timeout(300)) {
  bounded_tsq_owner<3>();
}

TEST_CASE("BoundedTaskQueue.Owner.LogSize=4" * doctest::timeout(300)) {
  bounded_tsq_owner<4>();
}

TEST_CASE("BoundedTaskQueue.Owner.LogSize=5" * doctest::timeout(300)) {
  bounded_tsq_owner<5>();
}

TEST_CASE("BoundedTaskQueue.Owner.LogSize=6" * doctest::timeout(300)) {
  bounded_tsq_owner<6>();
}

TEST_CASE("BoundedTaskQueue.Owner.LogSize=7" * doctest::timeout(300)) {
  bounded_tsq_owner<7>();
}

TEST_CASE("BoundedTaskQueue.Owner.LogSize=8" * doctest::timeout(300)) {
  bounded_tsq_owner<8>();
}

TEST_CASE("BoundedTaskQueue.Owner.LogSize=9" * doctest::timeout(300)) {
  bounded_tsq_owner<9>();
}

TEST_CASE("BoundedTaskQueue.Owner.LogSize=10" * doctest::timeout(300)) {
  bounded_tsq_owner<10>();
}


// ============================================================================
// UnboundedTaskQueue Test
// ============================================================================

// Procedure: unbounded_tsq_owner
void unbounded_tsq_owner() {

  for(size_t N=1; N<=777777; N=N*2+1) {
    tf::UnboundedTaskQueue<void*> queue;
    std::vector<void*> gold(N);

    REQUIRE(queue.empty());

    // push and pop
    for(size_t i=0; i<N; ++i) {
      gold[i] = &i;
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
    for(size_t i=1; i<N; ++i) {
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
// Bounded Task Queue Multiple Consumers Test
// ----------------------------------------------------------------------------

// Procedure: bounded_tsq_n_consumers
void bounded_tsq_n_consumers(size_t M) {
    
  tf::BoundedTaskQueue<void*> queue;

  std::vector<void*> gold;
  std::atomic<size_t> consumed;

  // 1, 4, 13, 40, 121, 364, 1093, 3280, 9841, 29524, 88573
  for(size_t N=1; N<=88573; N=N*3+1) {

    REQUIRE(queue.empty());

    gold.resize(N);
    consumed = 0;

    for(size_t i=0; i<N; ++i) {
      gold[i] = &i;
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

// ----------------------------------------------------------------------------
// Testcase: UnboundedTSQ Multiple Consumers Test
// ----------------------------------------------------------------------------

// Procedure: unbounded_tsq_n_consumers
void unbounded_tsq_n_consumers(size_t M) {
    
  tf::UnboundedTaskQueue<void*> queue;

  std::vector<void*> gold;
  std::atomic<size_t> consumed;

  // 1, 4, 13, 40, 121, 364, 1093, 3280, 9841, 29524, 88573, 265720
  for(size_t N=1; N<=265720; N=N*3+1) {

    REQUIRE(queue.empty());

    gold.resize(N);
    consumed = 0;

    for(size_t i=0; i<N; ++i) {
      gold[i] = &i;
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

  const uint64_t N = 65536;

  std::atomic<uint64_t> pcnt(0), ccnt(0), ans(0);
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



