// 2018/09/? - modified by Tsung-Wei Huang
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
  
  tp.silent_async([&](){
    if(tp.num_workers() == 0) {
      REQUIRE(tp.is_owner());
    }
    else {
      REQUIRE(!tp.is_owner());
      REQUIRE_THROWS(tp.shutdown());
      REQUIRE_THROWS(tp.spawn(4));
      REQUIRE_THROWS(tp.wait_for_all());
    }
  });

  std::vector<std::thread> threads;
  for(int i=0; i<10; ++i) {
    threads.emplace_back([&] () {
      REQUIRE(!tp.is_owner());
      REQUIRE_THROWS(tp.shutdown());
      REQUIRE_THROWS(tp.spawn(4));
      REQUIRE_THROWS(tp.wait_for_all());
    });
  }
  for(auto& t : threads) {
    t.join();
  }
}

// Procedure: test_silent_async
template <typename ThreadpoolType>
void test_silent_async(ThreadpoolType& tp) {

  constexpr size_t num_tasks = 1024;

  std::atomic<size_t> counter{0};
  
  size_t sum = 0;
  for(size_t i=0; i<num_tasks; i++){
    sum += i;
    tp.silent_async([i=i, &counter](){ counter += i; });
  }
  tp.wait_for_all(); //make sure all silent threads end

  REQUIRE(counter == sum);
}

// Procedure: test_async
template <typename ThreadpoolType>
void test_async(ThreadpoolType& tp){
  
  constexpr size_t num_tasks = 1024;
 
  std::vector<std::future<int>> int_future;
  std::vector<int> int_result;

  for(size_t i=0; i<num_tasks; i++){
    int_future.emplace_back(tp.async(
      [size = i](){
        int sum = 0;
        for(int i=0; i<=static_cast<int>(size); i++){
          sum += i;
        }       
        return sum;
      }
      )
    );
    
    int sum_result = 0;
    for(int j=0; j<=static_cast<int>(i); j++) sum_result += j;
    int_result.push_back(sum_result);
  }
  
  REQUIRE(int_future.size() == int_result.size());

  for(size_t i=0; i<int_future.size(); i++){
    REQUIRE(int_future[i].get() == int_result[i]);
  } 
}

// Procedure: test_wait_for_all
template <typename ThreadpoolType>
void test_wait_for_all(ThreadpoolType& tp){

  using namespace std::literals::chrono_literals;

  const size_t num_workers = tp.num_workers();
  const size_t num_tasks = 20;
  std::atomic<size_t> counter;

  for(int i=0; i<10; ++i) {

    counter = 0;
    
    for(size_t i=0; i<num_tasks; i++){
      tp.silent_async([&counter](){ 
        std::this_thread::sleep_for(200us);
        counter++; 
      });
    }
    REQUIRE(counter <= num_tasks);  // pay attention to the case of 0 worker

    tp.wait_for_all();

    REQUIRE(counter == num_tasks);
    REQUIRE(tp.num_workers() == num_workers);

    tp.wait_for_all();
  }
}

// Procedure: test_spawn_shutdown
template <typename T>
void test_spawn_shutdown(T& tp) {
  
  using namespace std::literals::chrono_literals;

  const size_t num_workers = tp.num_workers();
  const size_t num_tasks = 1024;

  tp.spawn(num_workers);
  REQUIRE(tp.num_workers() == num_workers * 2);

  tp.shutdown();
  REQUIRE(tp.num_workers() == 0);
  REQUIRE(tp.num_tasks() == 0);

  tp.spawn(num_workers);
  REQUIRE(tp.num_workers() == num_workers);

  for(int k=0; k<5; ++k) {

    REQUIRE(tp.num_workers() == num_workers);
    
    std::atomic<size_t> counter {0};

    for(size_t i=0; i<num_tasks; i++){
      tp.silent_async([&] () { 
        counter++;
        tp.silent_async([&] () {
          counter++;
          tp.silent_async([&] () {
            counter++;
            tp.silent_async([&] () {
              counter++;
            }); 
          });
        });
      });
    }

    tp.spawn(num_workers);

    REQUIRE(counter <= num_tasks * 4);

    tp.shutdown();

    REQUIRE(counter == num_tasks * 4);
    REQUIRE(tp.num_workers() == 0);
    REQUIRE(tp.num_tasks() == 0);

    tp.shutdown();
    REQUIRE(tp.num_workers() == 0);

    tp.spawn(num_workers);
    REQUIRE(tp.num_workers() == num_workers);
    REQUIRE(tp.num_tasks() == 0);
  }
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
      threadpool.silent_async([i=i-1, &insert] () {
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
  threadpool.wait_for_all();
  
  REQUIRE(cnt == 100 * threadpool.num_workers());
  REQUIRE(future.get() == 1);
  REQUIRE(sum == threadpool.num_workers());
}

// Procedure: test_binary_tree
template <typename T>
void test_binary_tree(T& threadpool) {
  
  for(int num_levels = 0; num_levels <= 8; ++num_levels) {
    std::atomic<size_t> sum {0};
    std::function<void(int)> insert;
    insert = [&threadpool, &insert, &sum, num_levels] (int l) {
      sum.fetch_add(1, std::memory_order_relaxed);
      if(l < num_levels) {
        for(int i=0; i<2; ++i) {
          threadpool.silent_async([&insert, l] () {
            insert(l+1);
          });
        }
      }
    };
    insert(0);
    threadpool.wait_for_all();
    REQUIRE(sum == (1 << (num_levels + 1)) - 1);
  }
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
        threadpool.silent_async([&] () { 
          sum.fetch_add(1, std::memory_order_relaxed);
        });
      }
    });
  }
  
  // master thread to insert
  for(int j=0; j<num_tasks; ++j) {
    threadpool.silent_async([&] () {
      sum.fetch_add(1, std::memory_order_relaxed);
    });
  }

  // worker thread to insert
  for(int i=0; i<10; ++i) {
    threadpool.silent_async([&] () {
      for(int j=0; j<num_tasks; ++j) {
        threadpool.silent_async([&] () {
          sum.fetch_add(1, std::memory_order_relaxed);
        });
      }
    });
  }
  
  for(auto& t : threads) {
    t.join();
  }

  threadpool.wait_for_all();

  REQUIRE(sum == num_tasks * 10 * 2 + num_tasks);
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

  SUBCASE("PlaceTask"){
    for(unsigned i=0; i<=4; ++i) {
      T tp(i);
      test_async(tp);
      test_silent_async(tp);
    }
  }
  
  SUBCASE("WaitForAll"){
    for(unsigned i=0; i<=4; ++i) {
      T tp(i);
      test_wait_for_all(tp);
    }
  }
  
  SUBCASE("SpawnShutdown") {
    for(unsigned i=0; i<=4; ++i) {
      T tp(i);
      test_spawn_shutdown(tp);
    }
  }

  SUBCASE("DynamicTasking") {
    for(unsigned i=0; i<=4; ++i) {
      T tp(i);
      test_dynamic_tasking(tp);
    }
  }

  SUBCASE("BinaryTree") {
    for(unsigned i=0; i<=4; ++i) {
      T tp(i);
      test_binary_tree(tp);
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

// ============================================================================
// Threadpool tests
// ============================================================================

// ----------------------------------------------------------------------------
// Testcase: SimpleThreadpool
// ----------------------------------------------------------------------------
TEST_CASE("SimpleThreadpool" * doctest::timeout(300)) {
  test_threadpool<tf::SimpleThreadpool>();
}

// ----------------------------------------------------------------------------
// Testcase: ProactiveThreadpool
// ----------------------------------------------------------------------------
TEST_CASE("ProactiveThreadpool" * doctest::timeout(300)) {
  test_threadpool<tf::ProactiveThreadpool>();
}

// ----------------------------------------------------------------------------
// Testcase: SpeculativeThreadpool
// ----------------------------------------------------------------------------
TEST_CASE("SpeculativeThreadpool" * doctest::timeout(300)) {
  test_threadpool<tf::SpeculativeThreadpool>();
}

// ----------------------------------------------------------------------------
// Testcase: PrivatizedThreadpool
// ----------------------------------------------------------------------------
TEST_CASE("PrivatizedThreadpool" * doctest::timeout(300)) {
  test_threadpool<tf::PrivatizedThreadpool>();
}



