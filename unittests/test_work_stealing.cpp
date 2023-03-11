#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

// ============================================================================
// Test without Priority
// ============================================================================

// Procedure: tsq_owner
void tsq_owner() {

  for(size_t N=1; N<=777777; N=N*2+1) {
    tf::TaskQueue<void*> queue;
    std::vector<void*> gold(N);

    REQUIRE(queue.empty());

    // push and pop
    for(size_t i=0; i<N; ++i) {
      gold[i] = &i;
      queue.push(gold[i], 0);
    }
    for(size_t i=0; i<N; ++i) {
      auto ptr = queue.pop();
      REQUIRE(ptr != nullptr);
      REQUIRE(gold[N-i-1] == ptr);
    }
    REQUIRE(queue.pop() == nullptr);

    // push and steal
    for(size_t i=0; i<N; ++i) {
      queue.push(gold[i], 0);
    }
    // i starts from 1 to avoid cache effect
    for(size_t i=1; i<N; ++i) {
      auto ptr = queue.steal();
      REQUIRE(ptr != nullptr);
      REQUIRE(gold[i] == ptr);
    }
  }
}

// Procedure: tsq_n_thieves
void tsq_n_thieves(size_t M) {

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
      queue.push(gold[i], 0);
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
TEST_CASE("WorkStealing.QueueOwner" * doctest::timeout(300)) {
  tsq_owner();
}

// ----------------------------------------------------------------------------
// Testcase: TSQTest.1Thief
// ----------------------------------------------------------------------------
TEST_CASE("WorkStealing.Queue1Thief" * doctest::timeout(300)) {
  tsq_n_thieves(1);
}

// ----------------------------------------------------------------------------
// Testcase: TSQTest.2Thieves
// ----------------------------------------------------------------------------
TEST_CASE("WorkStealing.Queue2Thieves" * doctest::timeout(300)) {
  tsq_n_thieves(2);
}

// ----------------------------------------------------------------------------
// Testcase: TSQTest.3Thieves
// ----------------------------------------------------------------------------
TEST_CASE("WorkStealing.Queue3Thieves" * doctest::timeout(300)) {
  tsq_n_thieves(3);
}

// ----------------------------------------------------------------------------
// Testcase: TSQTest.4Thieves
// ----------------------------------------------------------------------------
TEST_CASE("WorkStealing.Queue4Thieves" * doctest::timeout(300)) {
  tsq_n_thieves(4);
}

// ----------------------------------------------------------------------------
// Testcase: TSQTest.5Thieves
// ----------------------------------------------------------------------------
TEST_CASE("WorkStealing.Queue5Thieves" * doctest::timeout(300)) {
  tsq_n_thieves(5);
}

// ----------------------------------------------------------------------------
// Testcase: TSQTest.6Thieves
// ----------------------------------------------------------------------------
TEST_CASE("WorkStealing.Queue6Thieves" * doctest::timeout(300)) {
  tsq_n_thieves(6);
}

// ----------------------------------------------------------------------------
// Testcase: TSQTest.7Thieves
// ----------------------------------------------------------------------------
TEST_CASE("WorkStealing.Queue7Thieves" * doctest::timeout(300)) {
  tsq_n_thieves(7);
}

// ----------------------------------------------------------------------------
// Testcase: TSQTest.8Thieves
// ----------------------------------------------------------------------------
TEST_CASE("WorkStealing.Queue8Thieves" * doctest::timeout(300)) {
  tsq_n_thieves(8);
}

// ============================================================================
// Test with Priority
// ============================================================================

// Procedure: priority_tsq_owner
void priority_tsq_owner() {

  const unsigned P = 5;

  tf::TaskQueue<void*, P> queue;

  //for(unsigned p=0; p<P; p++) {
  //  REQUIRE(queue.push(nullptr, p) == true);
  //  REQUIRE(queue.push(nullptr, p) == false);
  //  REQUIRE(queue.push(nullptr, p) == false);
  //  REQUIRE(queue.push(nullptr, p) == false);

  //  REQUIRE(queue.pop(p) == nullptr);
  //  REQUIRE(queue.pop(p) == nullptr);
  //  REQUIRE(queue.pop(p) == nullptr);
  //  REQUIRE(queue.pop(p) == nullptr);
  //  REQUIRE(queue.pop(p) == nullptr);
  //  
  //  REQUIRE(queue.push(nullptr, p) == true);
  //  REQUIRE(queue.push(nullptr, p) == false);
  //  
  //  REQUIRE(queue.pop(p) == nullptr);
  //  REQUIRE(queue.pop(p) == nullptr);

  //  REQUIRE(queue.empty(p) == true);
  //}

  for(size_t N=1; N<=777777; N=N*2+1) {

    std::vector<std::pair<void*, unsigned>> gold(N);

    REQUIRE(queue.empty());
    REQUIRE(queue.pop() == nullptr);

    for(unsigned p=0; p<P; p++) {
      REQUIRE(queue.empty(p));
      REQUIRE(queue.pop(p) == nullptr);
      REQUIRE(queue.steal(p) == nullptr);
    }
    REQUIRE(queue.empty());

    // push 
    for(size_t i=0; i<N; ++i) {
      auto p = rand() % P;
      gold[i] = {&i, p};
      queue.push(&i, p);
    }

    // pop
    for(size_t i=0; i<N; ++i) {
      auto [g_ptr, g_pri]= gold[N-i-1];
      auto ptr = queue.pop(g_pri);
      REQUIRE(ptr != nullptr);
      REQUIRE(ptr == g_ptr);
    }
    REQUIRE(queue.pop() == nullptr);

    // push and steal
    for(size_t i=0; i<N; ++i) {
      queue.push(gold[i].first, gold[i].second);
    }

    // i starts from 1 to avoid cache effect
    for(size_t i=0; i<N; ++i) {
      auto [g_ptr, g_pri] = gold[i];
      auto ptr = queue.steal(g_pri);
      REQUIRE(ptr != nullptr);
      REQUIRE(g_ptr == ptr);
    }
    
    for(unsigned p=0; p<P; p++) {
      REQUIRE(queue.empty(p));
      REQUIRE(queue.pop(p) == nullptr);
      REQUIRE(queue.steal(p) == nullptr);
    }
    REQUIRE(queue.empty());
  }
}

TEST_CASE("WorkStealing.PriorityQueue.Owner" * doctest::timeout(300)) {
  priority_tsq_owner();
}

// ----------------------------------------------------------------------------
// Starvation Test
// ----------------------------------------------------------------------------

void starvation_test(size_t W) {

  tf::Taskflow taskflow;
  tf::Executor executor(W);
  std::atomic<size_t> counter{0};

  tf::Task prev, curr;
  
  // simple linear chain
  for(size_t l=0; l<100; l++) {

    curr = taskflow.emplace([&](){
      counter.fetch_add(1, std::memory_order_relaxed);
    });

    if(l) {
      curr.succeed(prev);
    }

    prev = curr;
  }

  // branches
  for(size_t b=W/2; b<W; b++) {
    taskflow.emplace([&](){
      counter.fetch_add(1, std::memory_order_relaxed);
    }).succeed(curr);
  }

  for(size_t b=0; b<W/2; b++) {
    taskflow.emplace([&](){
      while(counter.load(std::memory_order_relaxed) != W - W/2 + 100){
        std::this_thread::yield();
      }
    }).succeed(curr);
  }

  executor.run(taskflow).wait();

  REQUIRE(counter == W - W/2 + 100);

  // large linear chain followed by many branches
  size_t N = 1000;
  size_t target = 0;
  taskflow.clear();
  counter = 0;
  
  for(size_t l=0; l<N; l++) {
    curr = taskflow.emplace([&](){
      counter.fetch_add(1, std::memory_order_relaxed);
    });
    if(l) {
      curr.succeed(prev);
    }
    prev = curr;
  }

  const int w = rand() % W;

  for(size_t b=0; b<N; b++) {
    // wait with a probability of 0.9
    if(rand() % 10 != 0) {
      taskflow.emplace([&](){ 
        if(executor.this_worker_id() != w) {
          while(counter.load(std::memory_order_relaxed) != target + N) {
            std::this_thread::yield();
          }
        }
      }).succeed(curr);
    }
    // increment the counter with a probability of 0.1
    else {
      target++;
      taskflow.emplace([&](){ 
        counter.fetch_add(1, std::memory_order_relaxed);
      }).succeed(curr);
    }
  }

  executor.run(taskflow).wait();

  REQUIRE(counter == target + N);
  
}

TEST_CASE("WorkStealing.Starvation.1thread" * doctest::timeout(300)) {
  starvation_test(1);
}

TEST_CASE("WorkStealing.Starvation.2threads" * doctest::timeout(300)) {
  starvation_test(2);
}

TEST_CASE("WorkStealing.Starvation.3threads" * doctest::timeout(300)) {
  starvation_test(3);
}

TEST_CASE("WorkStealing.Starvation.4threads" * doctest::timeout(300)) {
  starvation_test(4);
}

TEST_CASE("WorkStealing.Starvation.5threads" * doctest::timeout(300)) {
  starvation_test(5);
}

TEST_CASE("WorkStealing.Starvation.6threads" * doctest::timeout(300)) {
  starvation_test(6);
}

TEST_CASE("WorkStealing.Starvation.7threads" * doctest::timeout(300)) {
  starvation_test(7);
}

TEST_CASE("WorkStealing.Starvation.8threads" * doctest::timeout(300)) {
  starvation_test(8);
}

// ----------------------------------------------------------------------------
// Starvation Loop Test
// ----------------------------------------------------------------------------

void starvation_loop_test(size_t W) {

  size_t L=100, B = 1024;

  REQUIRE(B > W);
  
  tf::Taskflow taskflow;
  tf::Executor executor(W);

  std::atomic<size_t> counter{0};
  std::atomic<size_t> barrier{0};

  // all worker must be involved
  std::mutex mutex;
  std::unordered_set<int> set;

  auto [merge, cond, stop] = taskflow.emplace(
    [&](){  
      REQUIRE(barrier.load(std::memory_order_relaxed) == B);
      REQUIRE(counter.load(std::memory_order_relaxed) == (L + B - 1));
      REQUIRE(set.size() == W);
      counter = 0;
      barrier = 0;
      set.clear();
    },
    [n=0]() mutable { 
      return ++n >= 10 ? 1 : 0;
    },
    [&](){
      REQUIRE(barrier.load(std::memory_order_relaxed) == 0);
      REQUIRE(counter.load(std::memory_order_relaxed) == 0);
      REQUIRE(set.size() == 0);
    }
  );

  tf::Task prev, curr, second;
  
  // linear chain with delay to make workers sleep
  for(size_t l=0; l<L; l++) {

    curr = taskflow.emplace([&, l](){
      if(l) {
        counter.fetch_add(1, std::memory_order_relaxed);
      }
    });

    if(l) {
      curr.succeed(prev);
    }

    if(l==1) {
      second = curr;
    }

    prev = curr;
  }
  
  cond.precede(second, stop);


  // fork
  for(size_t b=0; b<B; b++) {
    tf::Task task = taskflow.emplace([&](){
      // record the threads
      {
        std::scoped_lock lock(mutex);
        set.insert(executor.this_worker_id());
      }

      // all threads should be notified
      barrier.fetch_add(1, std::memory_order_relaxed);
      while(barrier.load(std::memory_order_relaxed) < W){
        std::this_thread::yield();
      }
      
      // increment the counter
      counter.fetch_add(1, std::memory_order_relaxed);
    });
    task.succeed(curr)
        .precede(merge);
  }

  merge.precede(cond);

  //taskflow.dump(std::cout);
  executor.run(taskflow).wait();
}

TEST_CASE("WorkStealing.StarvationLoop.1thread" * doctest::timeout(300)) {
  starvation_loop_test(1);
}

TEST_CASE("WorkStealing.StarvationLoop.2threads" * doctest::timeout(300)) {
  starvation_loop_test(2);
}

TEST_CASE("WorkStealing.StarvationLoop.3threads" * doctest::timeout(300)) {
  starvation_loop_test(3);
}

TEST_CASE("WorkStealing.StarvationLoop.4threads" * doctest::timeout(300)) {
  starvation_loop_test(4);
}

TEST_CASE("WorkStealing.StarvationLoop.5threads" * doctest::timeout(300)) {
  starvation_loop_test(5);
}

TEST_CASE("WorkStealing.StarvationLoop.6threads" * doctest::timeout(300)) {
  starvation_loop_test(6);
}

TEST_CASE("WorkStealing.StarvationLoop.7threads" * doctest::timeout(300)) {
  starvation_loop_test(7);
}

TEST_CASE("WorkStealing.StarvationLoop.8threads" * doctest::timeout(300)) {
  starvation_loop_test(8);
}

// ----------------------------------------------------------------------------
// Starvation Loop Test
// ----------------------------------------------------------------------------

void subflow_starvation_test(size_t W) {

  size_t L=100, B = 1024;

  REQUIRE(B > W);
  
  tf::Taskflow taskflow;
  tf::Executor executor(W);

  std::atomic<size_t> counter{0};
  std::atomic<size_t> barrier{0};

  // all worker must be involved
  std::mutex mutex;
  std::unordered_set<int> set;

  taskflow.emplace([&](tf::Subflow& subflow){

    auto [merge, cond, stop] = subflow.emplace(
      [&](){  
        REQUIRE(barrier.load(std::memory_order_relaxed) == B);
        REQUIRE(counter.load(std::memory_order_relaxed) == (L + B - 1));
        REQUIRE(set.size() == W);
        counter = 0;
        barrier = 0;
        set.clear();
      },
      [n=0]() mutable { 
        return ++n >= 5 ? 1 : 0;
      },
      [&](){
        REQUIRE(barrier.load(std::memory_order_relaxed) == 0);
        REQUIRE(counter.load(std::memory_order_relaxed) == 0);
        REQUIRE(set.size() == 0);
      }
    );

    tf::Task prev, curr, second;
    
    // linear chain with delay to make workers sleep
    for(size_t l=0; l<L; l++) {

      curr = subflow.emplace([&, l](){
        if(l) {
          counter.fetch_add(1, std::memory_order_relaxed);
        }
      });

      if(l) {
        curr.succeed(prev);
      }

      if(l==1) {
        second = curr;
      }

      prev = curr;
    }
    
    cond.precede(second, stop);


    // fork
    for(size_t b=0; b<B; b++) {
      tf::Task task = subflow.emplace([&](){
        // record the threads
        {
          std::scoped_lock lock(mutex);
          set.insert(executor.this_worker_id());
        }

        // all threads should be notified
        barrier.fetch_add(1, std::memory_order_relaxed);
        while(barrier.load(std::memory_order_relaxed) < W) {
          std::this_thread::yield();
        }
        
        // increment the counter
        counter.fetch_add(1, std::memory_order_relaxed);
      });
      task.succeed(curr)
          .precede(merge);
    }

    merge.precede(cond);
  });

  //taskflow.dump(std::cout);
  executor.run_n(taskflow, 5).wait();
}

TEST_CASE("WorkStealing.SubflowStarvation.1thread" * doctest::timeout(300)) {
  subflow_starvation_test(1);
}

TEST_CASE("WorkStealing.SubflowStarvation.2threads" * doctest::timeout(300)) {
  subflow_starvation_test(2);
}

TEST_CASE("WorkStealing.SubflowStarvation.3threads" * doctest::timeout(300)) {
  subflow_starvation_test(3);
}

TEST_CASE("WorkStealing.SubflowStarvation.4threads" * doctest::timeout(300)) {
  subflow_starvation_test(4);
}

TEST_CASE("WorkStealing.SubflowStarvation.5threads" * doctest::timeout(300)) {
  subflow_starvation_test(5);
}

TEST_CASE("WorkStealing.SubflowStarvation.6threads" * doctest::timeout(300)) {
  subflow_starvation_test(6);
}

TEST_CASE("WorkStealing.SubflowStarvation.7threads" * doctest::timeout(300)) {
  subflow_starvation_test(7);
}

TEST_CASE("WorkStealing.SubflowStarvation.8threads" * doctest::timeout(300)) {
  subflow_starvation_test(8);
}

// ----------------------------------------------------------------------------
// Embarrassing Starvation Test
// ----------------------------------------------------------------------------

void embarrasing_starvation_test(size_t W) {

  size_t B = 65536;

  REQUIRE(B > W);
  
  tf::Taskflow taskflow, parent;
  tf::Executor executor(W);

  std::atomic<size_t> barrier{0};

  // all worker must be involved
  std::mutex mutex;
  std::unordered_set<int> set;

  // fork
  for(size_t b=0; b<B; b++) {
    taskflow.emplace([&](){
      // record worker
      {
        std::scoped_lock lock(mutex);
        set.insert(executor.this_worker_id());
      }

      // all threads should be notified
      barrier.fetch_add(1, std::memory_order_relaxed);
      while(barrier.load(std::memory_order_relaxed) < W) {
        std::this_thread::yield();
      }
    });
  }

  parent.composed_of(taskflow);

  executor.run(parent).wait();

  REQUIRE(set.size() == W);
}

TEST_CASE("WorkStealing.EmbarrasingStarvation.1thread" * doctest::timeout(300)) {
  embarrasing_starvation_test(1);
}

TEST_CASE("WorkStealing.EmbarrasingStarvation.2threads" * doctest::timeout(300)) {
  embarrasing_starvation_test(2);
}

TEST_CASE("WorkStealing.EmbarrasingStarvation.3threads" * doctest::timeout(300)) {
  embarrasing_starvation_test(3);
}

TEST_CASE("WorkStealing.EmbarrasingStarvation.4threads" * doctest::timeout(300)) {
  embarrasing_starvation_test(4);
}

TEST_CASE("WorkStealing.EmbarrasingStarvation.5threads" * doctest::timeout(300)) {
  embarrasing_starvation_test(5);
}

TEST_CASE("WorkStealing.EmbarrasingStarvation.6threads" * doctest::timeout(300)) {
  embarrasing_starvation_test(6);
}

TEST_CASE("WorkStealing.EmbarrasingStarvation.7threads" * doctest::timeout(300)) {
  embarrasing_starvation_test(7);
}

TEST_CASE("WorkStealing.EmbarrasingStarvation.8threads" * doctest::timeout(300)) {
  embarrasing_starvation_test(8);
}

// ----------------------------------------------------------------------------
// skewed starvation
// ----------------------------------------------------------------------------

void skewed_starvation(size_t W) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(W);

  std::atomic<size_t> stop, count;
  
  // all worker must be involved
  std::mutex mutex;
  std::unordered_set<int> set;

  tf::Task parent = taskflow.emplace([&](){ 
    set.clear();
    count.store(0, std::memory_order_relaxed);
    stop.store(false, std::memory_order_relaxed);
  }).name("root");

  tf::Task left, right;

  for(size_t w=0; w<W; w++) {
    right = taskflow.emplace([&, w](){
      if(w) {
        // record the worker
        {
          std::scoped_lock lock(mutex);
          set.insert(executor.this_worker_id());
        }

        count.fetch_add(1, std::memory_order_release);

        // block the worker
        while(stop.load(std::memory_order_relaxed) == false) {
          std::this_thread::yield();
        }
      }
    }).name(std::string("right-") + std::to_string(w));

    left = taskflow.emplace([&](){
      std::this_thread::yield();
    }).name(std::string("left-") + std::to_string(w));
    
    // we want to remove the effect of parent stealing
    if(rand() & 1) {
      parent.precede(left, right);
    }
    else {
      parent.precede(right, left);
    }

    parent = left;
  }

  left = taskflow.emplace([&](){
    // wait for the other W-1 workers to block
    while(count.load(std::memory_order_acquire) + 1 != W) {
      std::this_thread::yield();
    }
    stop.store(true, std::memory_order_relaxed);

    REQUIRE(set.size() + 1 == W);
  }).name("stop");

  parent.precede(left);

  //taskflow.dump(std::cout);

  executor.run_n(taskflow, 1024).wait();
}

TEST_CASE("WorkStealing.SkewedStarvation.1thread") {
  skewed_starvation(1);
}

TEST_CASE("WorkStealing.SkewedStarvation.2threads") {
  skewed_starvation(2);
}

TEST_CASE("WorkStealing.SkewedStarvation.3threads") {
  skewed_starvation(3);
}

TEST_CASE("WorkStealing.SkewedStarvation.4threads") {
  skewed_starvation(4);
}

TEST_CASE("WorkStealing.SkewedStarvation.5threads") {
  skewed_starvation(5);
}

TEST_CASE("WorkStealing.SkewedStarvation.6threads") {
  skewed_starvation(6);
}

TEST_CASE("WorkStealing.SkewedStarvation.7threads") {
  skewed_starvation(7);
}

TEST_CASE("WorkStealing.SkewedStarvation.8threads") {
  skewed_starvation(8);
}

// ----------------------------------------------------------------------------
// NAryStravtion
// ----------------------------------------------------------------------------

void nary_starvation(size_t W) {
  
  size_t N = 1024;

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  std::atomic<size_t> stop, count;
  
  // all worker must be involved
  std::mutex mutex;
  std::unordered_set<int> set;

  tf::Task parent = taskflow.emplace([&](){ 
    set.clear();
    count.store(0, std::memory_order_relaxed);
    stop.store(false, std::memory_order_relaxed);
  }).name("root");

  tf::Task pivot;

  for(size_t w=0; w<W; w++) {
    
    // randomly choose a pivot to be the parent for the next level
    size_t p = rand()%N;

    for(size_t i=0; i<N; i++) {
      tf::Task task = taskflow.emplace([&, w, p, i](){
        
        // the blocker cannot be the pivot - others simply return
        if(i != (p+1)%N) {
          std::this_thread::yield();
          return;
        }
        
        // now I need to block
        if(w) {
          // record the worker
          {
            std::scoped_lock lock(mutex);
            set.insert(executor.this_worker_id());
          }

          count.fetch_add(1, std::memory_order_release);

          // block the worker
          while(stop.load(std::memory_order_relaxed) == false) {
            std::this_thread::yield();
          }
        }
      }).name(std::to_string(w));

      parent.precede(task);

      if(p == i) {
        pivot = task;
      }
    }
    parent = pivot;
  }

  pivot = taskflow.emplace([&](){
    // wait for the other W-1 workers to block
    while(count.load(std::memory_order_acquire) + 1 != W) {
      std::this_thread::yield();
    }
    stop.store(true, std::memory_order_relaxed);
    REQUIRE(set.size() + 1 == W);
  }).name("stop");

  parent.precede(pivot);

  //taskflow.dump(std::cout);

  executor.run_n(taskflow, 5).wait();
}

TEST_CASE("WorkStealing.NAryStarvation.1thread") {
  nary_starvation(1);
}

TEST_CASE("WorkStealing.NAryStarvation.2threads") {
  nary_starvation(2);
}

TEST_CASE("WorkStealing.NAryStarvation.3threads") {
  nary_starvation(3);
}

TEST_CASE("WorkStealing.NAryStarvation.4threads") {
  nary_starvation(4);
}

TEST_CASE("WorkStealing.NAryStarvation.5threads") {
  nary_starvation(5);
}

TEST_CASE("WorkStealing.NAryStarvation.6threads") {
  nary_starvation(6);
}

TEST_CASE("WorkStealing.NAryStarvation.7threads") {
  nary_starvation(7);
}

TEST_CASE("WorkStealing.NAryStarvation.8threads") {
  nary_starvation(8);
}

// ----------------------------------------------------------------------------
// Wavefront Starvation Test
// ----------------------------------------------------------------------------

void wavefront_starvation(size_t W) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(W);

  std::atomic<size_t> stop, count{0}, blocked;
  
  // all worker must be involved
  std::mutex mutex;
  std::unordered_set<int> set;

  std::vector<std::vector<tf::Task>> G;
  G.resize(W);

  // create tasks
  for(size_t i=0; i<W; i++) {
    G[i].resize(W);
    for(size_t j=0; j<W; j++) {
      // source
      if(i + j == 0) {
        G[i][j] = taskflow.emplace([&](){
          count.fetch_add(1, std::memory_order_relaxed);
          stop.store(false, std::memory_order_relaxed);
          blocked.store(0, std::memory_order_relaxed);
          set.clear();
        });
      }
      // diagonal tasks
      else if(i + j + 1 == W) {
        
        G[i][j] = taskflow.emplace([&, i, j](){

          count.fetch_add(1, std::memory_order_relaxed);
          
          // top-right will unblock all W-1 workers
          if(i == 0 && j + 1 == W) {
            while(blocked.load(std::memory_order_acquire) + 1 != W) {
              std::this_thread::yield();
            }
            stop.store(true, std::memory_order_relaxed);
            REQUIRE(set.size() + 1 == W);
          }
          else {
            // record the worker
            {
              std::scoped_lock lock(mutex);
              set.insert(executor.this_worker_id());
            }

            blocked.fetch_add(1, std::memory_order_release);

            // block the worker
            while(stop.load(std::memory_order_relaxed) == false) {
              std::this_thread::yield();
            }
          }
        });
      }
      // other tasks
      else {
        G[i][j] = taskflow.emplace([&](){
          count.fetch_add(1, std::memory_order_relaxed);
        });
      }

      // name the task
      G[i][j].name(std::to_string(i) + ", " + std::to_string(j));
    }
  }

  // create dependency
  for(size_t i=0; i<W; i++) {
    for(size_t j=0; j<W; j++) {
      size_t next_i = i + 1;
      size_t next_j = j + 1;
      if(next_i < W) {
        G[i][j].precede(G[next_i][j]);
      }
      if(next_j < W) {
        G[i][j].precede(G[i][next_j]);
      }
    }
  }

  //taskflow.dump(std::cout);
  executor.run_n(taskflow, 1024).wait();

  REQUIRE(count == W*W*1024);
}

TEST_CASE("WorkStealing.WavefrontStarvation.1thread") {
  wavefront_starvation(1);
}

TEST_CASE("WorkStealing.WavefrontStarvation.2threads") {
  wavefront_starvation(2);
}

TEST_CASE("WorkStealing.WavefrontStarvation.3threads") {
  wavefront_starvation(3);
}

TEST_CASE("WorkStealing.WavefrontStarvation.4threads") {
  wavefront_starvation(4);
}

TEST_CASE("WorkStealing.WavefrontStarvation.5threads") {
  wavefront_starvation(5);
}

TEST_CASE("WorkStealing.WavefrontStarvation.6threads") {
  wavefront_starvation(6);
}

TEST_CASE("WorkStealing.WavefrontStarvation.7threads") {
  wavefront_starvation(7);
}

TEST_CASE("WorkStealing.WavefrontStarvation.8threads") {
  wavefront_starvation(8);
}

// ----------------------------------------------------------------------------
// Oversubscription Test
// ----------------------------------------------------------------------------

void oversubscription_test(size_t W) {

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  std::atomic<size_t> counter{0};
  
  // all worker must be involved
  std::mutex mutex;
  std::unordered_set<int> set;

  for(size_t n = 0; n<W/2; n++) { 
  
    tf::Task prev, curr;

    for(size_t l=0; l<100; l++) {

      curr = taskflow.emplace([&](){
        // record worker
        {
          std::scoped_lock lock(mutex);
          set.insert(executor.this_worker_id());
        }
        counter.fetch_add(1, std::memory_order_relaxed);
      });

      if(l) {
        curr.succeed(prev);
      }

      prev = curr;
    }
  }

  for(size_t t=1; t<=100; t++) {
    set.clear();
    executor.run(taskflow).wait();
    REQUIRE(counter == 100*(W/2)*t);
    REQUIRE(set.size() <= W/2);
  }
}

TEST_CASE("WorkStealing.Oversubscription.2threads" * doctest::timeout(300)) {
  oversubscription_test(2);
}

TEST_CASE("WorkStealing.Oversubscription.3threads" * doctest::timeout(300)) {
  oversubscription_test(3);
}

TEST_CASE("WorkStealing.Oversubscription.4threads" * doctest::timeout(300)) {
  oversubscription_test(4);
}

TEST_CASE("WorkStealing.Oversubscription.5threads" * doctest::timeout(300)) {
  oversubscription_test(5);
}

TEST_CASE("WorkStealing.Oversubscription.6threads" * doctest::timeout(300)) {
  oversubscription_test(6);
}

TEST_CASE("WorkStealing.Oversubscription.7threads" * doctest::timeout(300)) {
  oversubscription_test(7);
}

TEST_CASE("WorkStealing.Oversubscription.8threads" * doctest::timeout(300)) {
  oversubscription_test(8);
}

//TEST_CASE("WorkStealing.Oversubscription.16threads" * doctest::timeout(300)) {
//  oversubscription_test(16);
//}
//
//TEST_CASE("WorkStealing.Oversubscription.32threads" * doctest::timeout(300)) {
//  oversubscription_test(32);
//}

// ----------------------------------------------------------------------------

void ws_broom(size_t W) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(W);
  
  tf::Task task, prev;
  for(size_t i=0; i<10; i++) {
    task = taskflow.emplace([&](){
      //std::cout << executor.this_worker() << std::endl;
      printf("linear by worker %d\n", executor.this_worker_id());
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    });

    if(i) {
      prev.precede(task);
    }

    prev = task;
  }

  for(size_t i=0; i<10; i++) {
    taskflow.emplace([&](){
      //std::cout << executor.this_worker() << std::endl;
      printf("parallel by worker %d\n", executor.this_worker_id());
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }).succeed(task);
  }

  executor.run(taskflow).wait();

}
 
//TEST_CASE("WS.broom.2threads") {
//  ws_broom(10);
//}

// ----------------------------------------------------------------------------
// Continuation
// ----------------------------------------------------------------------------

void continuation_test(size_t W) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(W);
  
  tf::Task curr, prev;

  int w = executor.this_worker_id();

  REQUIRE(w == -1);

  for(size_t i=0; i<1000; i++) {
    curr = taskflow.emplace([&, i]() mutable {
      if(i == 0) {
        w = executor.this_worker_id();
      } 
      else {
        REQUIRE(w == executor.this_worker_id());
      }
    });

    if(i) {
      prev.precede(curr);
    }

    prev = curr;
  }

  executor.run(taskflow).wait();

}

TEST_CASE("WorkStealing.Continuation.1thread" * doctest::timeout(300)) {
  continuation_test(1);
}

TEST_CASE("WorkStealing.Continuation.2threads" * doctest::timeout(300)) {
  continuation_test(2);
}

TEST_CASE("WorkStealing.Continuation.3threads" * doctest::timeout(300)) {
  continuation_test(3);
}

TEST_CASE("WorkStealing.Continuation.4threads" * doctest::timeout(300)) {
  continuation_test(4);
}

TEST_CASE("WorkStealing.Continuation.5threads" * doctest::timeout(300)) {
  continuation_test(5);
}

TEST_CASE("WorkStealing.Continuation.6threads" * doctest::timeout(300)) {
  continuation_test(6);
}

TEST_CASE("WorkStealing.Continuation.7threads" * doctest::timeout(300)) {
  continuation_test(7);
}

TEST_CASE("WorkStealing.Continuation.8threads" * doctest::timeout(300)) {
  continuation_test(8);
}









