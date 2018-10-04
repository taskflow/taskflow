// 2018/10/04 - modified by Tsung-Wei Huang
//   - removed binary tree tests
//   - removed spawn/shutdown tests
//   - removed siltne_async and async tests
//   - added emplace test
//   - adopted the new thread pool implementation
//
// 2018/09/29 - modified by Tsung-Wei Huang
//   - added binary tree tests
//   - added worker queue tests
//   - added external thread tests
//   - refactored threadpool tests
// 
// 2018/09/13 - modified by Tsung-Wei Huang & Chun-Xun
//   - added tests for ownership
//   - modified spawn-shutdown tests
//
// 2018/09/10 - modified by Tsung-Wei Huang
//   - added tests for SpeculativeThreadpool
//   - added dynamic tasking tests
//   - added spawn and shutdown tests
//
// 2018/09/02 - created by Guannan
//   - test_silent_async
//   - test_async
//   - test_wait_for_all

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/threadpool/threadpool.hpp>

// Procedure: test_ownership
template <typename ThreadpoolType>
void test_ownership(ThreadpoolType& tp) {
  
  REQUIRE(tp.is_owner()); 
  
  tp.emplace([&](){
    if(tp.num_workers() == 0) {
      REQUIRE(tp.is_owner());
    }
    else {
      REQUIRE(!tp.is_owner());
    }
  });

  std::vector<std::thread> threads;
  for(int i=0; i<10; ++i) {
    threads.emplace_back([&] () {
      REQUIRE(!tp.is_owner());
    });
  }
  for(auto& t : threads) {
    t.join();
  }
}

// Procedure: test_emplace
template <typename ThreadpoolType>
void test_emplace(ThreadpoolType& tp) {

  constexpr size_t num_tasks = 1024;

  std::atomic<size_t> counter{0};
  
  for(size_t i=0; i<num_tasks; i++){
    tp.emplace([&counter](){ 
      counter.fetch_add(1, std::memory_order_relaxed); }
    );
  }

  while(counter != num_tasks);
}

// Procedure: test_dynamic_tasking
template <typename T>
void test_dynamic_tasking(T& threadpool) {

  std::atomic<size_t> sum {0};
  std::atomic<size_t> cnt {0};

  std::function<void(int)> insert;
  std::promise<int> promise;
  auto future = promise.get_future();
  
  insert = [&threadpool, &insert, &sum, &promise, &cnt] (int i) {
    if(i > 0) {
      ++cnt;
      threadpool.emplace([i=i-1, &insert] () {
        insert(i);
      });
    }
    else {
      if(auto s = ++sum; s == threadpool.num_workers()) {
        promise.set_value(1);
      }
    }
  };

  if(auto W = threadpool.num_workers(); W > 0) {
    for(size_t i=0; i<threadpool.num_workers(); i++){
      insert(100);
    }
  }
  else {
    promise.set_value(1);
  }

  // synchronize until all tasks finish
  REQUIRE(future.get() == 1);
  REQUIRE(cnt == 100 * threadpool.num_workers());
  REQUIRE(sum == threadpool.num_workers());
}

// Procedure: test_external_threads
template <typename T>
void test_external_threads(T& threadpool) {

  constexpr int num_tasks = 65536;

  std::vector<std::thread> threads;
  std::atomic<size_t> sum {0};

  for(int i=0; i<10; ++i) {
    threads.emplace_back([&] () {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      for(int j=0; j<num_tasks; ++j) {
        threadpool.emplace([&] () { 
          sum.fetch_add(1, std::memory_order_relaxed);
        });
      }
    });
  }
  
  // master thread to insert
  for(int j=0; j<num_tasks; ++j) {
    threadpool.emplace([&] () {
      sum.fetch_add(1, std::memory_order_relaxed);
    });
  }

  // worker thread to insert
  for(int i=0; i<10; ++i) {
    threadpool.emplace([&] () {
      for(int j=0; j<num_tasks; ++j) {
        threadpool.emplace([&] () {
          sum.fetch_add(1, std::memory_order_relaxed);
        });
      }
    });
  }
  
  for(auto& t : threads) {
    t.join();
  }

  while(sum != num_tasks * 10 * 2 + num_tasks) ;
}
  
// Procedure: test_threadpool
template <typename T>
void test_threadpool() {  

  SUBCASE("Ownership") {
    for(unsigned i=0; i<=4; ++i) {
      T tp(i);
      test_ownership(tp);
    }
  }

  SUBCASE("Emplace") {
    for(unsigned i=0; i<=4; ++i) {
      T tp(i);
      test_emplace(tp);
    }
  }

  SUBCASE("DynamicTasking") {
    for(unsigned i=0; i<=4; ++i) {
      T tp(i);
      test_dynamic_tasking(tp);
    }
  }

  SUBCASE("ExternalThreads") {
    for(unsigned i=0; i<=4; ++i) {
      T tp(i);
      test_external_threads(tp);
    }
  }
}

// ============================================================================
// Threadpool tests
// ============================================================================

// ----------------------------------------------------------------------------
// Testcase: SimpleThreadpool
// ----------------------------------------------------------------------------
TEST_CASE("SimpleThreadpool" * doctest::timeout(300)) {
  test_threadpool<tf::SimpleThreadpool<std::function<void()>>>();
}

// ----------------------------------------------------------------------------
// Testcase: ProactiveThreadpool
// ----------------------------------------------------------------------------
TEST_CASE("ProactiveThreadpool" * doctest::timeout(300)) {
  test_threadpool<tf::ProactiveThreadpool<std::function<void()>>>();
}

// ----------------------------------------------------------------------------
// Testcase: SpeculativeThreadpool
// ----------------------------------------------------------------------------
TEST_CASE("SpeculativeThreadpool" * doctest::timeout(300)) {
  test_threadpool<tf::SpeculativeThreadpool<std::function<void()>>>();
}

/*
// ============================================================================
// WorkerQueue Tests
// ============================================================================

// ----------------------------------------------------------------------------
// Testcase: OneThread
// ----------------------------------------------------------------------------
TEST_CASE("WorkerQueue.OneThread" * doctest::timeout(300)) {

  constexpr int N = 1024;

  int data;

  tf::RunQueue<int, 1024> queue; 

  REQUIRE(queue.empty());

  // push_front + pop_back
  SUBCASE("PFPB") {
    for(int i=0; i<N; ++i) {
      REQUIRE(queue.push_front(i));
      REQUIRE(!queue.empty());
    }
    REQUIRE(!queue.push_front(data));
    REQUIRE(!queue.push_back(data));

    for(int i=0; i<N; ++i) {
      REQUIRE(queue.pop_back(data));
      REQUIRE(data == i);
    }
    REQUIRE(queue.empty());
  }
  
  // push_front + pop_front
  SUBCASE("PFPF") {
    for(int i=0; i<N; ++i) {
      REQUIRE(queue.push_front(i));
      REQUIRE(!queue.empty());
    }
    REQUIRE(!queue.push_front(data));
    REQUIRE(!queue.push_back(data));

    for(int i=0; i<N; ++i) {
      REQUIRE(queue.pop_front(data));
      REQUIRE(data == N-i-1);
    }
    REQUIRE(queue.empty());
  }

  // push_back + pop_back
  SUBCASE("PBPB") {
    for(int i=0; i<N; ++i) {
      REQUIRE(queue.push_back(i));
      REQUIRE(!queue.empty());
    }
    REQUIRE(!queue.push_front(data));
    REQUIRE(!queue.push_back(data));

    for(int i=0; i<N; ++i) {
      REQUIRE(queue.pop_back(data));
      REQUIRE(data == N-i-1);
    }
    REQUIRE(queue.empty());
  }

  // push_back + pop_front
  SUBCASE("PBPF") {
    for(int i=0; i<N; ++i) {
      REQUIRE(queue.push_back(i));
      REQUIRE(!queue.empty());
    }
    REQUIRE(!queue.push_back(data));
    REQUIRE(!queue.push_front(data));

    for(int i=0; i<N; ++i) {
      REQUIRE(queue.pop_front(data));
      REQUIRE(data == i);
    }
    REQUIRE(queue.empty());
  }

  // half-half
  SUBCASE("HalfHalf") {
    for(int i=0; i<N; ++i) {
      if(i < N/2) {
        REQUIRE(queue.push_front(i));
      }
      else {
        REQUIRE(queue.push_back(i));
      }
      REQUIRE(!queue.empty());
    }
    REQUIRE(!queue.push_front(data));
    REQUIRE(!queue.push_back(data));

    for(int i=0; i<N; ++i) {
      if(i < N/2) {
        REQUIRE(queue.pop_front(data));
        REQUIRE(data == N/2 - i - 1);
      }
      else {
        REQUIRE(queue.pop_front(data));
        REQUIRE(data == i);
      }
    }
    REQUIRE(queue.empty());
  }

  // back-and-forth
  SUBCASE("BackAndForth") {
    for(int i=0; i<N; ++i) {
      if(i % 2 == 0) {
        REQUIRE(queue.push_front(i));
      }
      else {
        REQUIRE(queue.push_back(i));
      }
    }

    for(int i=0; i<N; ++i) {
      if(i < N/2) {
        REQUIRE(queue.pop_back(data));
        REQUIRE(data == N - (2*i+1));
      }
      else {
        REQUIRE(queue.pop_back(data));
        REQUIRE(data == 2*i - N);
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Testcase: TwoThread
// ----------------------------------------------------------------------------
TEST_CASE("WorkerQueue.TwoThread" * doctest::timeout(300)) {

  constexpr int N = (1 << 20);

  tf::RunQueue<int, 64> queue; 

  // push front + pop back
  // notice that there is no test of push front + pop front
  SUBCASE("PFPB") {

    REQUIRE(queue.empty());

    std::thread t1([&, i=0] () mutable {
      while(i < N) {
        if(queue.push_front(i)) {
          ++i;
        }
      }
    });

    std::thread t2([&, i=0] () mutable {
      int data;
      while(i < N) {
        if(queue.pop_back(data)) {
          REQUIRE(data == i);
          ++i;
        }
      }
    });

    t1.join();
    t2.join();
    
    REQUIRE(queue.empty());
  }
  
  // push back + pop front
  SUBCASE("PBPF") {

    REQUIRE(queue.empty());

    std::thread t1([&, i=0] () mutable {
      while(i < N) {
        if(queue.push_back(i)) {
          ++i;
        }
      }
    });

    std::thread t2([&, i=0] () mutable {
      int data;
      while(i < N) {
        if(queue.pop_front(data)) {
          REQUIRE(data == i);
          ++i;
        }
      }
    });

    t1.join();
    t2.join();
    
    REQUIRE(queue.empty());
  }
  
  // push back + pop back
  SUBCASE("PBPB") {

    std::vector<int> res;

    REQUIRE(queue.empty());

    std::thread t1([&, i=0] () mutable {
      while(i < N) {
        if(queue.push_back(i)) {
          ++i;
        }
      }
    });

    std::thread t2([&, i=0] () mutable {
      int data;
      while(i < N) {
        if(queue.pop_back(data)) {
          res.push_back(data);
          ++i;
        }
      }
    });

    t1.join();
    t2.join();
    
    REQUIRE(queue.empty());

    std::sort(res.begin(), res.end());

    REQUIRE(res.size() == N);

    for(int i=0; i<N; ++i) {
      REQUIRE(res[i] == i);
    }
  }
}

// ----------------------------------------------------------------------------
// Testcase: TriThread
// ----------------------------------------------------------------------------
TEST_CASE("WorkerQueue.TriThread" * doctest::timeout(300)) {

  constexpr int N = (1 << 21);

  tf::RunQueue<int, 64> queue; 

  std::vector<int> res;

  // push front + push back + pop back
  REQUIRE(queue.empty());

  std::thread t1([&, i=0] () mutable {
    while(i < N/2) {
      if(queue.push_front(i)) {
        ++i;
      }
    }
  });
  
  std::thread t2([&, i=N/2] () mutable {
    while(i < N) {
      if(queue.push_back(i)) {
        ++i;
      }
    }
  });

  std::thread t3([&, i=0] () mutable {
    int data;
    while(i < N) {
      if(queue.pop_back(data)) {
        res.push_back(data);
        ++i;
      }
    }
  });

  t1.join();
  t2.join();
  t3.join();
  
  REQUIRE(queue.empty());

  std::sort(res.begin(), res.end());

  REQUIRE(res.size() == N);

  for(int i=0; i<N; ++i) {
    REQUIRE(res[i] == i);
  }
  
}
*/




