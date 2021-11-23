// 2020/02/24 - modified by Tsung-Wei Huang
//  - isolaged wsq to a standalone project
//  - here we focus on tsq where T must be a pointer type
//
// 2019/05/15 - modified by Tsung-Wei Huang
//  - temporarily disable executor test
//
// 2019/04/11 - modified by Tsung-Wei Huang
//  - renamed threadpool to executor
//
// 2019/02/15 - modified by Tsung-Wei Huang
//  - modified batch tests (reference instead of move)  
//
// 2018/12/04 - modified by Tsung-Wei Huang
//  - replaced privatized executor with work stealing executor
//
// 2018/12/03 - modified by Tsung-Wei Huang
//  - added work stealing queue tests
//
// 2018/11/29 - modified by Chun-Xun Lin
//  - added batch tests
//
// 2018/10/04 - modified by Tsung-Wei Huang
//  - removed binary tree tests
//  - removed spawn/shutdown tests
//  - removed siltne_async and async tests
//  - added emplace test
//  - adopted the new thread pool implementation
//
// 2018/09/29 - modified by Tsung-Wei Huang
//  - added binary tree tests
//  - added worker queue tests
//  - added external thread tests
//  - refactored executor tests
// 
// 2018/09/13 - modified by Tsung-Wei Huang & Chun-Xun
//  - added tests for ownership
//  - modified spawn-shutdown tests
//
// 2018/09/10 - modified by Tsung-Wei Huang
//  - added tests for SpeculativeExecutor
//  - added dynamic tasking tests
//  - added spawn and shutdown tests
//
// 2018/09/02 - created by Guannan
//  - test_silent_async
//  - test_async
//  - test_wait_for_all

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

// ============================================================================
// WorkStealingQueue tests
// ============================================================================

// Procedure: tsq_test_owner
void tsq_test_owner() {

  for(size_t N=1; N<=777777; N=N*2+1) {
    tf::TaskQueue<void*> queue;
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

// Procedure: tsq_test_n_thieves
void tsq_test_n_thieves(size_t M) {
  
  for(size_t N=1; N<=777777; N=N*2+1) {
    tf::TaskQueue<void*> queue;
    std::vector<void*> gold(N);
    std::atomic<size_t> consumed {0};
    
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

// ----------------------------------------------------------------------------
// Testcase: TSQTest.Owner
// ----------------------------------------------------------------------------
TEST_CASE("TSQ.Owner" * doctest::timeout(300)) {
  tsq_test_owner();
}

// ----------------------------------------------------------------------------
// Testcase: TSQTest.1Thief
// ----------------------------------------------------------------------------
TEST_CASE("TSQ.1Thief" * doctest::timeout(300)) {
  tsq_test_n_thieves(1);
}

// ----------------------------------------------------------------------------
// Testcase: TSQTest.2Thieves
// ----------------------------------------------------------------------------
TEST_CASE("TSQ.2Thieves" * doctest::timeout(300)) {
  tsq_test_n_thieves(2);
}

// ----------------------------------------------------------------------------
// Testcase: TSQTest.3Thieves
// ----------------------------------------------------------------------------
TEST_CASE("TSQ.3Thieves" * doctest::timeout(300)) {
  tsq_test_n_thieves(3);
}

// ----------------------------------------------------------------------------
// Testcase: TSQTest.4Thieves
// ----------------------------------------------------------------------------
TEST_CASE("TSQ.4Thieves" * doctest::timeout(300)) {
  tsq_test_n_thieves(4);
}

// ----------------------------------------------------------------------------
// Testcase: TSQTest.5Thieves
// ----------------------------------------------------------------------------
TEST_CASE("TSQ.5Thieves" * doctest::timeout(300)) {
  tsq_test_n_thieves(5);
}

// ----------------------------------------------------------------------------
// Testcase: TSQTest.6Thieves
// ----------------------------------------------------------------------------
TEST_CASE("TSQ.6Thieves" * doctest::timeout(300)) {
  tsq_test_n_thieves(6);
}

// ----------------------------------------------------------------------------
// Testcase: TSQTest.7Thieves
// ----------------------------------------------------------------------------
TEST_CASE("TSQ.7Thieves" * doctest::timeout(300)) {
  tsq_test_n_thieves(7);
}

// ----------------------------------------------------------------------------
// Testcase: TSQTest.8Thieves
// ----------------------------------------------------------------------------
TEST_CASE("TSQ.8Thieves" * doctest::timeout(300)) {
  tsq_test_n_thieves(8);
}


