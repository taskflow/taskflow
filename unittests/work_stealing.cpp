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

  tf::Task prev, curr;

  for(size_t l=0; l<100; l++) {

    curr = taskflow.emplace([&](){
      // wait until all workers sleep
      //while(executor.num_thieves() != 0);
    });

    if(l) {
      curr.succeed(prev);
    }

    prev = curr;
  }

  // branches
  std::atomic<size_t> counter{0};
  for(size_t b=W/2; b<W; b++) {
    taskflow.emplace([&](){
      counter++;
    }).succeed(curr);
  }

  for(size_t b=0; b<W/2; b++) {
    taskflow.emplace([&](){
      while(counter != W - W/2);
    }).succeed(curr);
  }

  executor.run(taskflow).wait();

  //while(executor.num_thieves() != 0);

  REQUIRE(counter == W - W/2);
  
  
  //TODO: bug? (some extreme situations may run forever ...)
  // large linear chain followed by many branches
  size_t N = 100000;
  size_t target = 0;
  taskflow.clear();
  counter = 0;
  
  for(size_t l=0; l<N; l++) {
    curr = taskflow.emplace([&, l](){
      //while(executor.num_thieves() != 0);
      //if(l == N-1) {
        //printf("worker %d at the last node of the chain\n", executor.this_worker_id());
      //}
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
          //printf("worker %lu enters the loop (t=%lu, c=%lu, w=%d, n=%lu)\n", 
          //  worker->id(), target, counter.load(), w, worker->queue_size()
          //);
          while(counter != target); 
        }
      }).succeed(curr);
    }
    // increment the counter with a probability of 0.1
    else {
      target++;
      taskflow.emplace([&](){ ++counter; }).succeed(curr);
    }
  }

  executor.run(taskflow).wait();

  REQUIRE(counter == target);
  
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
// Oversubscription Test
// ----------------------------------------------------------------------------

void oversubscription_test(size_t W) {

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  std::atomic<size_t> counter{0};

  for(size_t n = 0; n<W/2; n++) { 
  
    tf::Task prev, curr;

    for(size_t l=0; l<100; l++) {

      curr = taskflow.emplace([&](){
        counter++;
        //while(executor.num_thieves() != 0);
      });

      if(l) {
        curr.succeed(prev);
      }

      prev = curr;
    }
  }

  executor.run(taskflow).wait();

  REQUIRE(counter == 100*(W/2));
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











