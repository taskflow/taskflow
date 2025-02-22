#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

// ----------------------------------------------------------------------------
// Starvation Test
// ----------------------------------------------------------------------------

void starvation_branches(size_t W) {

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

}

TEST_CASE("WorkStealing.Starvation.Branches.1thread" * doctest::timeout(300)) {
  starvation_branches(1);
}

TEST_CASE("WorkStealing.Starvation.Branches.2threads" * doctest::timeout(300)) {
  starvation_branches(2);
}

TEST_CASE("WorkStealing.Starvation.Branches.3threads" * doctest::timeout(300)) {
  starvation_branches(3);
}

TEST_CASE("WorkStealing.Starvation.Branches.4threads" * doctest::timeout(300)) {
  starvation_branches(4);
}

TEST_CASE("WorkStealing.Starvation.Branches.5threads" * doctest::timeout(300)) {
  starvation_branches(5);
}

TEST_CASE("WorkStealing.Starvation.Branches.6threads" * doctest::timeout(300)) {
  starvation_branches(6);
}

TEST_CASE("WorkStealing.Starvation.Branches.7threads" * doctest::timeout(300)) {
  starvation_branches(7);
}

TEST_CASE("WorkStealing.Starvation.Branches.8threads" * doctest::timeout(300)) {
  starvation_branches(8);
}

// ----------------------------------------------------------------------------
// Starvation Linear Chain with Branches
// ----------------------------------------------------------------------------

void starvation_branch_with_pivot(size_t W) {

  tf::Taskflow taskflow;
  tf::Executor executor(W);
  std::atomic<size_t> counter{0};

  tf::Task prev, curr;
  
  // large linear chain followed by many branches
  size_t N = 1000;
  size_t target = 0;
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

TEST_CASE("WorkStealing.Starvation.BranchesWithPivot.1thread" * doctest::timeout(300)) {
  starvation_branch_with_pivot(1);
}

TEST_CASE("WorkStealing.Starvation.BranchesWithPivot.2threads" * doctest::timeout(300)) {
  starvation_branch_with_pivot(2);
}

TEST_CASE("WorkStealing.Starvation.BranchesWithPivot.3threads" * doctest::timeout(300)) {
  starvation_branch_with_pivot(3);
}

TEST_CASE("WorkStealing.Starvation.BranchesWithPivot.4threads" * doctest::timeout(300)) {
  starvation_branch_with_pivot(4);
}

TEST_CASE("WorkStealing.Starvation.BranchesWithPivot.5threads" * doctest::timeout(300)) {
  starvation_branch_with_pivot(5);
}

TEST_CASE("WorkStealing.Starvation.BranchesWithPivot.6threads" * doctest::timeout(300)) {
  starvation_branch_with_pivot(6);
}

TEST_CASE("WorkStealing.Starvation.BranchesWithPivot.7threads" * doctest::timeout(300)) {
  starvation_branch_with_pivot(7);
}

TEST_CASE("WorkStealing.Starvation.BranchesWithPivot.8threads" * doctest::timeout(300)) {
  starvation_branch_with_pivot(8);
}

// ----------------------------------------------------------------------------
// Starvation Loop Test
// ----------------------------------------------------------------------------

void starvation_loop(size_t W) {

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

TEST_CASE("WorkStealing.Starvation.Loop.1thread" * doctest::timeout(300)) {
  starvation_loop(1);
}

TEST_CASE("WorkStealing.Starvation.Loop.2threads" * doctest::timeout(300)) {
  starvation_loop(2);
}

TEST_CASE("WorkStealing.Starvation.Loop.3threads" * doctest::timeout(300)) {
  starvation_loop(3);
}

TEST_CASE("WorkStealing.Starvation.Loop.4threads" * doctest::timeout(300)) {
  starvation_loop(4);
}

TEST_CASE("WorkStealing.Starvation.Loop.5threads" * doctest::timeout(300)) {
  starvation_loop(5);
}

TEST_CASE("WorkStealing.Starvation.Loop.6threads" * doctest::timeout(300)) {
  starvation_loop(6);
}

TEST_CASE("WorkStealing.Starvation.Loop.7threads" * doctest::timeout(300)) {
  starvation_loop(7);
}

TEST_CASE("WorkStealing.Starvation.Loop.8threads" * doctest::timeout(300)) {
  starvation_loop(8);
}

// ----------------------------------------------------------------------------
// Starvation Loop Test
// ----------------------------------------------------------------------------

void subflow_starvation(size_t W) {

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
  subflow_starvation(1);
}

TEST_CASE("WorkStealing.SubflowStarvation.2threads" * doctest::timeout(300)) {
  subflow_starvation(2);
}

TEST_CASE("WorkStealing.SubflowStarvation.3threads" * doctest::timeout(300)) {
  subflow_starvation(3);
}

TEST_CASE("WorkStealing.SubflowStarvation.4threads" * doctest::timeout(300)) {
  subflow_starvation(4);
}

TEST_CASE("WorkStealing.SubflowStarvation.5threads" * doctest::timeout(300)) {
  subflow_starvation(5);
}

TEST_CASE("WorkStealing.SubflowStarvation.6threads" * doctest::timeout(300)) {
  subflow_starvation(6);
}

TEST_CASE("WorkStealing.SubflowStarvation.7threads" * doctest::timeout(300)) {
  subflow_starvation(7);
}

TEST_CASE("WorkStealing.SubflowStarvation.8threads" * doctest::timeout(300)) {
  subflow_starvation(8);
}

// ----------------------------------------------------------------------------
// Embarrassing Starvation Test
// ----------------------------------------------------------------------------

void embarrasing_starvation(size_t W) {

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
  embarrasing_starvation(1);
}

TEST_CASE("WorkStealing.EmbarrasingStarvation.2threads" * doctest::timeout(300)) {
  embarrasing_starvation(2);
}

TEST_CASE("WorkStealing.EmbarrasingStarvation.3threads" * doctest::timeout(300)) {
  embarrasing_starvation(3);
}

TEST_CASE("WorkStealing.EmbarrasingStarvation.4threads" * doctest::timeout(300)) {
  embarrasing_starvation(4);
}

TEST_CASE("WorkStealing.EmbarrasingStarvation.5threads" * doctest::timeout(300)) {
  embarrasing_starvation(5);
}

TEST_CASE("WorkStealing.EmbarrasingStarvation.6threads" * doctest::timeout(300)) {
  embarrasing_starvation(6);
}

TEST_CASE("WorkStealing.EmbarrasingStarvation.7threads" * doctest::timeout(300)) {
  embarrasing_starvation(7);
}

TEST_CASE("WorkStealing.EmbarrasingStarvation.8threads" * doctest::timeout(300)) {
  embarrasing_starvation(8);
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

void oversubscription(size_t W) {

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
  oversubscription(2);
}

TEST_CASE("WorkStealing.Oversubscription.3threads" * doctest::timeout(300)) {
  oversubscription(3);
}

TEST_CASE("WorkStealing.Oversubscription.4threads" * doctest::timeout(300)) {
  oversubscription(4);
}

TEST_CASE("WorkStealing.Oversubscription.5threads" * doctest::timeout(300)) {
  oversubscription(5);
}

TEST_CASE("WorkStealing.Oversubscription.6threads" * doctest::timeout(300)) {
  oversubscription(6);
}

TEST_CASE("WorkStealing.Oversubscription.7threads" * doctest::timeout(300)) {
  oversubscription(7);
}

TEST_CASE("WorkStealing.Oversubscription.8threads" * doctest::timeout(300)) {
  oversubscription(8);
}

//TEST_CASE("WorkStealing.Oversubscription.16threads" * doctest::timeout(300)) {
//  oversubscription(16);
//}
//
//TEST_CASE("WorkStealing.Oversubscription.32threads" * doctest::timeout(300)) {
//  oversubscription(32);
//}

// ----------------------------------------------------------------------------
// Continuation
// ----------------------------------------------------------------------------

void continuation(size_t W) {
  
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
  continuation(1);
}

TEST_CASE("WorkStealing.Continuation.2threads" * doctest::timeout(300)) {
  continuation(2);
}

TEST_CASE("WorkStealing.Continuation.3threads" * doctest::timeout(300)) {
  continuation(3);
}

TEST_CASE("WorkStealing.Continuation.4threads" * doctest::timeout(300)) {
  continuation(4);
}

TEST_CASE("WorkStealing.Continuation.5threads" * doctest::timeout(300)) {
  continuation(5);
}

TEST_CASE("WorkStealing.Continuation.6threads" * doctest::timeout(300)) {
  continuation(6);
}

TEST_CASE("WorkStealing.Continuation.7threads" * doctest::timeout(300)) {
  continuation(7);
}

TEST_CASE("WorkStealing.Continuation.8threads" * doctest::timeout(300)) {
  continuation(8);
}


