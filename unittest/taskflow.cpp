#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <vector>
#include <utility>
#include <chrono>
#include <limits.h>

// --------------------------------------------------------
// Testcase: Builder
// --------------------------------------------------------
TEST_CASE("Builder" * doctest::timeout(300)) {

  const size_t num_tasks = 100;

  tf::Taskflow taskflow;
  tf::Executor executor;

  std::atomic<int> counter {0};
  std::vector<tf::Task> silent_tasks;
  std::vector<tf::Task> tasks;
    
  SUBCASE("Placeholder") {
    
    for(size_t i=0; i<num_tasks; ++i) {
      silent_tasks.emplace_back(taskflow.placeholder().name(std::to_string(i)));
    }

    for(size_t i=0; i<num_tasks; ++i) {
      REQUIRE(silent_tasks[i].name() == std::to_string(i));
      REQUIRE(silent_tasks[i].num_dependents() == 0);
      REQUIRE(silent_tasks[i].num_successors() == 0);
    }

    for(auto& task : silent_tasks) {
      task.work([&counter](){ counter++; });
    }

    executor.run(taskflow).get();
    REQUIRE(counter == num_tasks);
  }

  SUBCASE("EmbarrassinglyParallel"){

    for(size_t i=0;i<num_tasks;i++) {
      tasks.emplace_back(taskflow.emplace([&counter]() {counter += 1;}));
    }

    REQUIRE(taskflow.num_nodes() == num_tasks);
    executor.run(taskflow).get();
    REQUIRE(counter == num_tasks);
    REQUIRE(taskflow.num_nodes() == 100);

    counter = 0;
    
    for(size_t i=0;i<num_tasks;i++){
      silent_tasks.emplace_back(taskflow.emplace([&counter]() {counter += 1;}));
    }

    REQUIRE(taskflow.num_nodes() == num_tasks * 2);
    executor.run(taskflow).get();
    REQUIRE(counter == num_tasks * 2);
    REQUIRE(taskflow.num_nodes() == 200);
  }
  
  SUBCASE("BinarySequence"){
    for(size_t i=0;i<num_tasks;i++){
      if(i%2 == 0){
        tasks.emplace_back(
          taskflow.emplace([&counter]() { REQUIRE(counter == 0); counter += 1;})
        );
      }
      else{
        tasks.emplace_back(
          taskflow.emplace([&counter]() { REQUIRE(counter == 1); counter -= 1;})
        );
      }
      if(i>0){
        //tasks[i-1].first.precede(tasks[i].first);
        tasks[i-1].precede(tasks[i]);
      }

      if(i==0) {
        //REQUIRE(tasks[i].first.num_dependents() == 0);
        REQUIRE(tasks[i].num_dependents() == 0);
      }
      else {
        //REQUIRE(tasks[i].first.num_dependents() == 1);
        REQUIRE(tasks[i].num_dependents() == 1);
      }
    }
    executor.run(taskflow).get();
  }

  SUBCASE("LinearCounter"){
    for(size_t i=0;i<num_tasks;i++){
      tasks.emplace_back(
        taskflow.emplace([&counter, i]() { REQUIRE(counter == i); counter += 1;})
      );
      if(i>0){
        taskflow.precede(tasks[i-1], tasks[i]);
      }
    }
    executor.run(taskflow).get();
    REQUIRE(counter == num_tasks);
    REQUIRE(taskflow.num_nodes() == num_tasks);
  }
 
  SUBCASE("Broadcast"){
    auto src = taskflow.emplace([&counter]() {counter -= 1;});
    for(size_t i=1; i<num_tasks; i++){
      silent_tasks.emplace_back(
        taskflow.emplace([&counter]() {REQUIRE(counter == -1);})
      );
    }
    taskflow.broadcast(src, silent_tasks);
    executor.run(taskflow).get();
    REQUIRE(counter == - 1);
    REQUIRE(taskflow.num_nodes() == num_tasks);
  }

  SUBCASE("Gather"){
    auto dst = taskflow.emplace([&counter, num_tasks]() { REQUIRE(counter == num_tasks - 1);});
    for(size_t i=1;i<num_tasks;i++){
      silent_tasks.emplace_back(taskflow.emplace([&counter]() {counter += 1;}));
    }
    dst.gather(silent_tasks);
    executor.run(taskflow).get();
    REQUIRE(counter == num_tasks - 1);
    REQUIRE(taskflow.num_nodes() == num_tasks);
  }

  SUBCASE("MapReduce"){
    auto src = taskflow.emplace([&counter]() {counter = 0;});
    for(size_t i=0;i<num_tasks;i++){
      silent_tasks.emplace_back(taskflow.emplace([&counter]() {counter += 1;}));
    }
    taskflow.broadcast(src, silent_tasks);
    auto dst = taskflow.emplace(
      [&counter, num_tasks]() { REQUIRE(counter == num_tasks);}
    );
    taskflow.gather(silent_tasks, dst);
    executor.run(taskflow).get();
    REQUIRE(taskflow.num_nodes() == num_tasks + 2);
  }

  SUBCASE("Linearize"){
    for(size_t i=0;i<num_tasks;i++){
      silent_tasks.emplace_back(
        taskflow.emplace([&counter, i]() { REQUIRE(counter == i); counter += 1;})
      );
    }
    taskflow.linearize(silent_tasks);
    executor.run(taskflow).get();
    REQUIRE(counter == num_tasks);
    REQUIRE(taskflow.num_nodes() == num_tasks);
  }

  SUBCASE("Kite"){
    auto src = taskflow.emplace([&counter]() {counter = 0;});
    for(size_t i=0;i<num_tasks;i++){
      silent_tasks.emplace_back(
        taskflow.emplace([&counter, i]() { REQUIRE(counter == i); counter += 1; })
      );
    }
    taskflow.broadcast(src, silent_tasks);
    taskflow.linearize(silent_tasks);
    auto dst = taskflow.emplace(
      [&counter, num_tasks]() { REQUIRE(counter == num_tasks);}
    );
    taskflow.gather(silent_tasks, dst);
    executor.run(taskflow).get();
    REQUIRE(taskflow.num_nodes() == num_tasks + 2);
  }
}

// --------------------------------------------------------
// Testcase: Run
// --------------------------------------------------------
TEST_CASE("Run" * doctest::timeout(300)) {
    
  using namespace std::chrono_literals;
  
  size_t num_workers = 4;
  size_t num_tasks = 100;
  
  tf::Executor executor(num_workers);
  tf::Taskflow taskflow;

  REQUIRE(executor.num_workers() == num_workers);

  std::atomic<int> counter {0};
  std::vector<tf::Task> silent_tasks;
    
  for(size_t i=0;i<num_tasks;i++){
    silent_tasks.emplace_back(taskflow.emplace([&counter]() {counter += 1;}));
  }

  SUBCASE("RunOnce"){
    auto fu = executor.run(taskflow);
    REQUIRE(taskflow.num_nodes() == num_tasks);
    REQUIRE(fu.wait_for(1s) == std::future_status::ready);
    REQUIRE(counter == num_tasks);
  }

  SUBCASE("WaitForAll") {
    executor.run(taskflow);
    executor.wait_for_all();
    REQUIRE(counter == num_tasks); 
  }
  
  SUBCASE("RunVariants") {
    // Empty subflow test
    for(unsigned W=0; W<=4; ++W) {

      std::atomic<size_t> count {0};
      tf::Taskflow f;
      auto A = f.emplace([&](){ count ++; });
      auto B = f.emplace([&](auto& subflow){ 
        count ++; 
        auto B1 = subflow.emplace([&](){ count++; });
        auto B2 = subflow.emplace([&](){ count++; });
        auto B3 = subflow.emplace([&](){ count++; });
        B1.precede(B3); B2.precede(B3);
      });
      auto C = f.emplace([&](){ count ++; });
      auto D = f.emplace([&](){ count ++; });

      A.precede(B, C);
      B.precede(D); 
      C.precede(D);

      tf::Executor executor(W);
      std::list<std::shared_future<void>> fu_list;
      for(size_t i=0; i<500; i++) {
        if(i == 499) {
          executor.run(f).get();   // Synchronize the first 500 runs
          executor.run_n(f, 500);  // Run 500 times more
        }
        else if(i % 2) {
          fu_list.push_back(executor.run(f));
        }
        else {
          fu_list.push_back(executor.run(f, [&, i=i](){ REQUIRE(count == (i+1)*7); }));
        }
      }

      for(auto& fu: fu_list) {
        REQUIRE(fu.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
      }

      executor.wait_for_all();

      REQUIRE(count == 7000);
    }

    // TODO: test correctness when taskflow got changed between runs 
    for(unsigned W=0; W<=4; ++W) {

      std::atomic<size_t> count {0};
      tf::Taskflow f;
      auto A = f.emplace([&](){ count ++; });
      auto B = f.emplace([&](auto& subflow){ 
        count ++; 
        auto B1 = subflow.emplace([&](){ count++; });
        auto B2 = subflow.emplace([&](){ count++; });
        auto B3 = subflow.emplace([&](){ count++; });
        B1.precede(B3); B2.precede(B3);
      });
      auto C = f.emplace([&](){ count ++; });
      auto D = f.emplace([&](){ count ++; });

      A.precede(B, C);
      B.precede(D); 
      C.precede(D);

      tf::Executor executor(W);
      executor.run_n(f, 10).get();
      REQUIRE(count == 70);    

      auto E = f.emplace([](){});
      D.precede(E);
      executor.run_n(f, 10).get();
      REQUIRE(count == 140);    

      auto F = f.emplace([](){});
      E.precede(F);
      executor.run_n(f, 10);
      executor.wait_for_all();
      REQUIRE(count == 210);    
    }

    // Test run_until 
    for(unsigned W=0; W<=4; ++W) {

      std::atomic<size_t> count {0};
      tf::Taskflow f;
      auto A = f.emplace([&](){ count ++; });
      auto B = f.emplace([&](auto& subflow){ 
        count ++; 
        auto B1 = subflow.emplace([&](){ count++; });
        auto B2 = subflow.emplace([&](){ count++; });
        auto B3 = subflow.emplace([&](){ count++; });
        B1.precede(B3); B2.precede(B3);
      });
      auto C = f.emplace([&](){ count ++; });
      auto D = f.emplace([&](){ count ++; });

      A.precede(B, C);
      B.precede(D); 
      C.precede(D);

      tf::Executor executor(W);
      executor.run_until(f, [run=10]() mutable { return run-- == 0; }, 
        [&](){
          REQUIRE(count == 70);
          count = 0;
        }
      ).get();


      executor.run_until(f, [run=10]() mutable { return run-- == 0; }, 
        [&](){
          REQUIRE(count == 70);
          count = 0;
          auto E = f.emplace([&](){ count ++; });
          auto F = f.emplace([&](){ count ++; });
          A.precede(E).precede(F);
      });

      executor.run_until(f, [run=10]() mutable { return run-- == 0; }, 
        [&](){
          REQUIRE(count == 90);
          count = 0;
        }
      ).get();
    }
  }
}

// --------------------------------------------------------
// Testcase: ParallelFor
// --------------------------------------------------------
TEST_CASE("ParallelFor" * doctest::timeout(300)) {
    
  using namespace std::chrono_literals;

  const auto mapper = [](size_t num_workers, size_t num_data, bool group){
    tf::Executor executor(static_cast<unsigned>(num_workers));
    tf::Taskflow tf;
    std::vector<int> vec(num_data, 0);
    tf.parallel_for(
      vec.begin(), vec.end(), [] (int& v) { v = 64; }, group ? ::rand() : 0
    );
    for(const auto v : vec) {
      REQUIRE(v == 0);
    }
    executor.run(tf);
    executor.wait_for_all();
    for(const auto v : vec) {
      REQUIRE(v == 64);
    }
  };

  const auto reducer = [](size_t num_workers, size_t num_data, bool group){
    tf::Executor executor(static_cast<unsigned>(num_workers));
    tf::Taskflow tf;
    std::vector<int> vec(num_data, 0);
    std::atomic<int> sum(0);
    tf.parallel_for(vec.begin(), vec.end(), [&](auto) { ++sum; }, group ? ::rand() : 0);
    REQUIRE(sum == 0);
    executor.run(tf);
    executor.wait_for_all();
    REQUIRE(sum == vec.size());
  };

  // map
  SUBCASE("Map") {
    for(size_t num_workers=0; num_workers<=4; ++num_workers){
      for(size_t num_data=1; num_data<=59049; num_data *= 3){
        mapper(num_workers, num_data, true);
        mapper(num_workers, num_data, false);
      }
    }
  }

  // reduce
  SUBCASE("Reduce") {
    for(size_t num_workers=0; num_workers<=4; ++num_workers){
      for(size_t num_data=1; num_data<=59049; num_data *= 3){
        reducer(num_workers, num_data, true);
        reducer(num_workers, num_data, false);
      }
    }
  }
}

// --------------------------------------------------------
// Testcase: ParallelForOnIndex
// --------------------------------------------------------
TEST_CASE("ParallelForOnIndex" * doctest::timeout(300)) {
    
  using namespace std::chrono_literals;

  auto exception_test = [] (unsigned num_workers) {

    tf::Executor executor(num_workers);
    tf::Taskflow tf;

    // invalid index
    REQUIRE_THROWS(tf.parallel_for(0, 10, 0, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0, 10, -1, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(10, 0, 0, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(10, 0, 1, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0u, 10u, 0u, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(10u, 0u, 0u, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(10u, 0u, 1u, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0.0f, 10.0f, 0.0f, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0.0f, 10.0f, -1.0f, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(10.0f, 0.0f, 0.0f, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(10.0f, 0.0f, 1.0f, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0.0, 10.0, 0.0, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0.0, 10.0, -1.0, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(10.0, 0.0, 0.0, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(10.0, 0.0, 1.0, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0, 0, 0, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0u, 0u, 0u, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0.0, 0.0, 0.0, [] (auto) {}));
    
    // graceful case
    REQUIRE_NOTHROW(tf.parallel_for(0, 0, -1, [] (auto) {}));
    REQUIRE_NOTHROW(tf.parallel_for(0, 0, 1, [] (auto) {}));
    REQUIRE_NOTHROW(tf.parallel_for(0u, 0u, 1u, [] (auto) {}));
    REQUIRE_NOTHROW(tf.parallel_for(0.0f, 0.0f, -1.0f, [] (auto) {}));
    REQUIRE_NOTHROW(tf.parallel_for(0.0f, 0.0f, 1.0f, [] (auto) {}));
    REQUIRE_NOTHROW(tf.parallel_for(0.0, 0.0, -1.0, [] (auto) {}));
    REQUIRE_NOTHROW(tf.parallel_for(0.0, 0.0, 1.0, [] (auto) {}));
  };

  auto positive_integer_step = [] (unsigned num_workers) {
    tf::Executor executor(num_workers);
    for(int beg=-10; beg<=10; ++beg) {
      for(int end=beg; end<=10; ++end) {
        for(int s=1; s<=end-beg; ++s) {
          int n = 0;
          for(int b = beg; b<end; b+=s) {
            ++n;
          }
          tf::Taskflow tf;
          std::atomic<int> counter {0};
          tf.parallel_for(beg, end, s, [&] (auto) {
            counter.fetch_add(1, std::memory_order_relaxed);
          });
          executor.run(tf);
          executor.wait_for_all();
          REQUIRE(n == counter);
        }
      }
    }
  };
  
  auto negative_integer_step = [] (unsigned num_workers) {
    tf::Executor executor(num_workers);
    for(int beg=10; beg>=-10; --beg) {
      for(int end=beg; end>=-10; --end) {
        for(int s=1; s<=beg-end; ++s) {
          int n = 0;
          for(int b = beg; b>end; b-=s) {
            ++n;
          }
          tf::Taskflow tf;
          std::atomic<int> counter {0};
          tf.parallel_for(beg, end, -s, [&] (auto) {
            counter.fetch_add(1, std::memory_order_relaxed);
          });
          executor.run(tf);
          executor.wait_for_all();
          REQUIRE(n == counter);
        }
      }
    }
  };
  
  auto positive_floating_step = [] (unsigned num_workers) {
    tf::Executor executor(num_workers);
    for(float beg=-10.0f; beg<=10.0f; ++beg) {
      for(float end=beg; end<=10.0f; ++end) {
        for(float s=1.0f; s<=end-beg; s+=0.1f) {
          int n = 0;
          for(float b = beg; b<end; b+=s) {
            ++n;
          }
          tf::Taskflow tf;
          std::atomic<int> counter {0};
          tf.parallel_for(beg, end, s, [&] (auto) {
            counter.fetch_add(1, std::memory_order_relaxed);
          });
          executor.run(tf);
          executor.wait_for_all();
          REQUIRE(n == counter);
        }
      }
    }
  };
  
  auto negative_floating_step = [] (unsigned num_workers) {
    tf::Executor executor(num_workers);
    for(float beg=10.0f; beg>=-10.0f; --beg) {
      for(float end=beg; end>=-10.0f; --end) {
        for(float s=1.0f; s<=beg-end; s+=0.1f) {
          int n = 0;
          for(float b = beg; b>end; b-=s) {
            ++n;
          }
          tf::Taskflow tf;
          std::atomic<int> counter {0};
          tf.parallel_for(beg, end, -s, [&] (auto) {
            counter.fetch_add(1, std::memory_order_relaxed);
          });
          executor.run(tf);
          executor.wait_for_all();
          REQUIRE(n == counter);
        }
      }
    }
  };
  
  SUBCASE("Exception") {
    for(unsigned w=0; w<=4; w++) {
      exception_test(w);
    }
  }

  SUBCASE("PositiveIntegerStep") {
    for(unsigned w=0; w<=4; w++) {
      positive_integer_step(w);  
    }
  }
  
  SUBCASE("NegativeIntegerStep") {
    for(unsigned w=0; w<=4; w++) {
      negative_integer_step(w);  
    }
  }
  
  SUBCASE("PositiveFloatingStep") {
    for(unsigned w=0; w<=4; w++) {
      positive_floating_step(w);  
    }
  }
  
  SUBCASE("NegativeFloatingStep") {
    for(unsigned w=0; w<=4; w++) {
      negative_floating_step(w);  
    }
  }

}

// --------------------------------------------------------
// Testcase: Reduce
// --------------------------------------------------------
TEST_CASE("Reduce" * doctest::timeout(300)) {

  const auto plus_test = [](size_t num_workers, auto &&data){
    tf::Executor executor(static_cast<unsigned>(num_workers));
    tf::Taskflow tf;
    int result {0};
    std::iota(data.begin(), data.end(), 1);
    tf.reduce(data.begin(), data.end(), result, std::plus<int>());
    executor.run(tf).get();
    REQUIRE(result == std::accumulate(data.begin(), data.end(), 0, std::plus<int>()));
  };

  const auto multiply_test = [](size_t num_workers, auto &&data){
    tf::Executor executor(static_cast<unsigned>(num_workers));
    tf::Taskflow tf;
    std::fill(data.begin(), data.end(), 1.0);
    double result {2.0};
    tf.reduce(data.begin(), data.end(), result, std::multiplies<double>());
    executor.run(tf).get();
    REQUIRE(result == std::accumulate(data.begin(), data.end(), 2.0, std::multiplies<double>()));
  };

  const auto max_test = [](size_t num_workers, auto &&data){
    tf::Executor executor(static_cast<unsigned>(num_workers));
    tf::Taskflow tf;
    std::iota(data.begin(), data.end(), 1);
    int result {0};
    auto lambda = [](const auto& l, const auto& r){return std::max(l, r);};
    tf.reduce(data.begin(), data.end(), result, lambda);
    executor.run(tf).get();
    REQUIRE(result == std::accumulate(data.begin(), data.end(), 0, lambda));
  };

  const auto min_test = [](size_t num_workers, auto &&data){
    tf::Executor executor(static_cast<unsigned>(num_workers));
    tf::Taskflow tf;
    std::iota(data.begin(), data.end(), 1);
    int result {std::numeric_limits<int>::max()};
    auto lambda = [](const auto& l, const auto& r){return std::min(l, r);};
    tf.reduce(data.begin(), data.end(), result, lambda);
    executor.run(tf).get();
    REQUIRE(result == std::accumulate(
      data.begin(), data.end(), std::numeric_limits<int>::max(), lambda)
    );
  };

  for(size_t i=0; i<=4; ++i){
    for(size_t j=0; j<=256; j=j*2+1){
      plus_test(i, std::vector<int>(j));
      plus_test(i, std::list<int>(j));

      multiply_test(i, std::vector<double>(j));
      multiply_test(i, std::list<double>(j));

      max_test(i, std::vector<int>(j));
      max_test(i, std::list<int>(j));

      min_test(i, std::vector<int>(j));
      min_test(i, std::list<int>(j));
    }
  }
}

// --------------------------------------------------------
// Testcase: ReduceMin
// --------------------------------------------------------
TEST_CASE("ReduceMin" * doctest::timeout(300)) {

  for(int w=0; w<=4; w++) {
    tf::Executor executor(w);
    for(int i=0; i<=65536; i = (i <= 1024) ? i + 1 : i*2 + 1) {
      tf::Taskflow tf;
      std::vector<int> data(i);
      int gold = std::numeric_limits<int>::max();
      int test = std::numeric_limits<int>::max();
      for(auto& d : data) {
        d = ::rand();
        gold = std::min(gold, d);
      }
      tf.reduce_min(data.begin(), data.end(), test);
      executor.run(tf).get();
      REQUIRE(test == gold);
    }
  }

}

// --------------------------------------------------------
// Testcase: ReduceMax
// --------------------------------------------------------
TEST_CASE("ReduceMax" * doctest::timeout(300)) {

  for(int w=0; w<=4; w++) {
    tf::Executor executor(w);
    for(int i=0; i<=65536; i = (i <= 1024) ? i + 1 : i*2 + 1) {
      tf::Taskflow tf;
      std::vector<int> data(i);
      int gold = std::numeric_limits<int>::min();
      int test = std::numeric_limits<int>::min();
      for(auto& d : data) {
        d = ::rand();
        gold = std::max(gold, d);
      }
      tf.reduce_max(data.begin(), data.end(), test);
      executor.run(tf).get();
      REQUIRE(test == gold);
    }
  }
}

// --------------------------------------------------------
// Testcase: JoinedSubflow
// -------------------------------------------------------- 
TEST_CASE("JoinedSubflow" * doctest::timeout(300)){

  using namespace std::literals::chrono_literals;
  
  SUBCASE("Trivial") {
    // Empty subflow test
    for(unsigned W=0; W<=4; ++W) {
      
      tf::Executor executor(W);
      tf::Taskflow tf;
      
      // empty flow with future
      tf::Task subflow3, subflow3_;
      //std::future<int> fu3, fu3_;
      std::atomic<int> fu3v{0}, fu3v_{0};
      
      // empty flow
      auto subflow1 = tf.emplace([&] (auto& fb) {
        fu3v++;
      }).name("subflow1");
      
      // nested empty flow
      auto subflow2 = tf.emplace([&] (auto& fb) {
        fu3v++;
        fb.emplace([&] (auto& fb) {
          fu3v++;
          fb.emplace( [&] (auto& fb) {
            fu3v++;
          }).name("subflow2_1_1");
        }).name("subflow2_1");
      }).name("subflow2");
      
      //std::tie(subflow3, fu3) = tf.emplace([&] (auto& fb) {
      subflow3 = tf.emplace([&] (auto& fb) {

        REQUIRE(fu3v == 4);

        fu3v++;
        fu3v_++;
        
        //std::tie(subflow3_, fu3_) = fb.emplace([&] (auto& fb) {
        subflow3_ = fb.emplace([&] (auto& fb) {
          REQUIRE(fu3v_ == 3);
          fu3v++;
          fu3v_++;
          //return 200;
        });
        subflow3_.name("subflow3_");

        // hereafter we use 100us to avoid dangling reference ...
        auto s1 = fb.emplace([&] () { 
          fu3v_++;
          fu3v++;
        }).name("s1");
        
        auto s2 = fb.emplace([&] () {
          fu3v_++;
          fu3v++;
        }).name("s2");
        
        auto s3 = fb.emplace([&] () {
          fu3v++;
          REQUIRE(fu3v_ == 4);
        }).name("s3");

        s1.precede(subflow3_);
        s2.precede(subflow3_);
        subflow3_.precede(s3);

        REQUIRE(fu3v_ == 1);

        //return 100;
      });
      subflow3.name("subflow3");

      // empty flow to test future
      auto subflow4 = tf.emplace([&] () {
        fu3v++;
      }).name("subflow4");

      subflow1.precede(subflow2);
      subflow2.precede(subflow3);
      subflow3.precede(subflow4);

      executor.run(tf).get();
    } // End of for loop
  }
  
  // Mixed intra- and inter- operations
  SUBCASE("Complex") {

    for(unsigned W=0; W<=4; ++W) {

      tf::Executor executor(W);
      tf::Taskflow tf;

      std::vector<int> data;
      int sum {0};

      auto A = tf.emplace([&data] () {
        for(int i=0; i<10; ++i) {
          data.push_back(1);
        }
      });

      std::atomic<size_t> count = 0;

      auto B = tf.emplace([&count, &data, &sum](auto& fb){

        auto [src, tgt] = fb.reduce(data.begin(), data.end(), sum, std::plus<int>());

        fb.emplace([&sum] () { REQUIRE(sum == 0); }).precede(src);

        tgt.precede(fb.emplace([&sum] () { REQUIRE(sum == 10); }));

        for(size_t i=0; i<10; i ++){
          ++count;
        }

        auto n = fb.emplace([&count](auto& fb){

          REQUIRE(count == 20);
          ++count;

          auto prev = fb.emplace([&count](){
            REQUIRE(count == 21);
            ++count;
          });

          for(size_t i=0; i<10; i++){
            auto next = fb.emplace([&count, i](){
              REQUIRE(count == 22+i);
              ++count;
            });
            prev.precede(next);
            prev = next;
          }
        });

        for(size_t i=0; i<10; i++){
          fb.emplace([&count](){ ++count; }).precede(n);
        }
      });

      A.precede(B);

      executor.run(tf).get();
      REQUIRE(count == 32);
      REQUIRE(sum == 10);
    }
  }
}

// --------------------------------------------------------
// Testcase: DetachedSubflow
// --------------------------------------------------------
TEST_CASE("DetachedSubflow" * doctest::timeout(300)) {
  
  using namespace std::literals::chrono_literals;

  SUBCASE("Trivial") {

    // Empty subflow test
    for(unsigned W=0; W<=4; ++W) {

      tf::Executor executor(W);
      tf::Taskflow tf;
      
      // empty flow with future
      tf::Task subflow3, subflow3_;
      std::atomic<int> fu3v{0}, fu3v_{0};
      
      // empty flow
      auto subflow1 = tf.emplace([&] (auto& fb) {
        fu3v++;
        fb.detach();
      }).name("subflow1");
      
      // nested empty flow
      auto subflow2 = tf.emplace([&] (auto& fb) {
        fu3v++;
        fb.emplace([&] (auto& fb) {
          fu3v++;
          fb.emplace( [&] (auto& fb) {
            fu3v++;
          }).name("subflow2_1_1");
          fb.detach();
        }).name("subflow2_1");
        fb.detach();
      }).name("subflow2");
      
      //std::tie(subflow3, fu3) = tf.emplace([&] (auto& fb) {
      subflow3 = tf.emplace([&] (auto& fb) {

        REQUIRE((fu3v >= 2 && fu3v <= 4));

        fu3v++;
        fu3v_++;
        
        //std::tie(subflow3_, fu3_) = fb.emplace([&] (auto& fb) {
        subflow3_ = fb.emplace([&] (auto& fb) {
          REQUIRE(fu3v_ == 3);
          fu3v++;
          fu3v_++;
        });
        subflow3_.name("subflow3_");

        // hereafter we use 100us to avoid dangling reference ...
        auto s1 = fb.emplace([&] () { 
          fu3v_++;
          fu3v++;
        }).name("s1");
        
        auto s2 = fb.emplace([&] () {
          fu3v_++;
          fu3v++;
        }).name("s2");
        
        auto s3 = fb.emplace([&] () {
          fu3v++;
          REQUIRE(fu3v_ == 4);
        }).name("s3");

        s1.precede(subflow3_);
        s2.precede(subflow3_);
        subflow3_.precede(s3);

        REQUIRE(fu3v_ == 1);

        fb.detach();

        return 100;
      });
      subflow3.name("subflow3");

      // empty flow to test future
      auto subflow4 = tf.emplace([&] () {
        REQUIRE((fu3v >= 3 && fu3v <= 9));
        fu3v++;
      }).name("subflow4");

      subflow1.precede(subflow2);
      subflow2.precede(subflow3);
      subflow3.precede(subflow4);

      executor.run(tf).get();

      REQUIRE(fu3v  == 10);
      REQUIRE(fu3v_ == 4);
    }
  }
}

// --------------------------------------------------------
// Testcase: Composition
// --------------------------------------------------------
TEST_CASE("Composition-1" * doctest::timeout(300)) {

  for(unsigned w=0; w<=8; ++w) {

    tf::Executor executor(w);

    tf::Taskflow f0;

    int cnt {0};

    auto [A, B, C, D, E] = f0.emplace(
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; }
    );

    A.precede(B);
    B.precede(C);
    C.precede(D);
    D.precede(E);

    tf::Taskflow f1;
    
    // module 1
    std::tie(A, B, C, D, E) = f1.emplace(
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; }
    );
    A.precede(B);
    B.precede(C);
    C.precede(D);
    D.precede(E);
    auto m1_1 = f1.composed_of(f0);
    E.precede(m1_1);
    
    executor.run(f1).get();
    REQUIRE(cnt == 10);

    cnt = 0;
    executor.run_n(f1, 100).get();
    REQUIRE(cnt == 10 * 100);

    auto m1_2 = f1.composed_of(f0);
    m1_1.precede(m1_2);
    
    for(int n=0; n<100; n++) {
      cnt = 0;
      executor.run_n(f1, n).get();
      REQUIRE(cnt == 15*n);
    }

    cnt = 0;
    for(int n=0; n<100; n++) {
      executor.run(f1);
    }
    
    executor.wait_for_all();

    REQUIRE(cnt == 1500);
  }
}

// TESTCASE: composition-2
TEST_CASE("Composition-2" * doctest::timeout(300)) {

  for(unsigned w=0; w<=8; ++w) {

    tf::Executor executor(w);

    int cnt {0};
    
    // level 0 (+5)
    tf::Taskflow f0;

    auto [A, B, C, D, E] = f0.emplace(
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; }
    );

    A.precede(B);
    B.precede(C);
    C.precede(D);
    D.precede(E);

    // level 1 (+10)
    tf::Taskflow f1;
    auto m1_1 = f1.composed_of(f0);
    auto m1_2 = f1.composed_of(f0);
    m1_1.precede(m1_2);

    // level 2 (+20)
    tf::Taskflow f2;
    auto m2_1 = f2.composed_of(f1);
    auto m2_2 = f2.composed_of(f1);
    m2_1.precede(m2_2);
    
    // synchronous run
    for(int n=0; n<100; n++) {
      cnt = 0;
      executor.run_n(f2, n).get();
      REQUIRE(cnt == 20*n);
    }

    // asynchronous run
    cnt = 0;
    for(int n=0; n<100; n++) {
      executor.run(f2);
    }
    executor.wait_for_all();
    REQUIRE(cnt == 100*20);
  }
}

// TESTCASE: composition-3
TEST_CASE("Composition-3" * doctest::timeout(300)) {
  
  for(unsigned w=0; w<=8; ++w) {
  
    tf::Executor executor(w);

    int cnt {0};
    
    // level 0 (+2)
    tf::Taskflow f0;

    auto [A, B] = f0.emplace(
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; }
    );
    A.precede(B);

    // level 1 (+4)
    tf::Taskflow f1;
    auto m1_1 = f1.composed_of(f0);
    auto m1_2 = f1.composed_of(f0);
    m1_1.precede(m1_2);

    // level 2 (+8)
    tf::Taskflow f2;
    auto m2_1 = f2.composed_of(f1);
    auto m2_2 = f2.composed_of(f1);
    m2_1.precede(m2_2);

    // level 3 (+16)
    tf::Taskflow f3;
    auto m3_1 = f3.composed_of(f2);
    auto m3_2 = f3.composed_of(f2);
    m3_1.precede(m3_2);

    // synchronous run
    for(int n=0; n<100; n++) {
      cnt = 0;
      executor.run_n(f3, n).get();
      REQUIRE(cnt == 16*n);
    }

    // asynchronous run
    cnt = 0;
    for(int n=0; n<100; n++) {
      executor.run(f3);
    }
    executor.wait_for_all();
    REQUIRE(cnt == 16*100);
  }
}

// --------------------------------------------------------
// Testcase: Observer 
// -------------------------------------------------------- 
TEST_CASE("Observer" * doctest::timeout(300)) {

  for(unsigned w=0; w<=8; ++w) {
    tf::Executor executor(w);

    auto observer = executor.make_observer<tf::ExecutorObserver>();    

    tf::Taskflow taskflowA;
    std::vector<tf::Task> tasks;
    // Static tasking 
    for(auto i=0; i < 64; i ++) {
      tasks.emplace_back(taskflowA.emplace([](){}));
    }

    // Randomly specify dependency
    for(auto i=0; i < 64; i ++) {
      for(auto j=i+1; j < 64; j++) {
        if(rand()%2 == 0) {
          tasks[i].precede(tasks[j]);
        }
      }
    }

    executor.run_n(taskflowA, 16).get();

    if(w == 0) {
      REQUIRE(observer->num_tasks() == 0);
    }
    else {
      REQUIRE(observer->num_tasks() == 64*16);
    }

    observer->clear();
    REQUIRE(observer->num_tasks() == 0);
    tasks.clear();

    // Dynamic tasking  
    tf::Taskflow taskflowB;
    std::atomic<int> num_tasks {0};
    // Static tasking 
    for(auto i=0; i < 64; i ++) {
      tasks.emplace_back(taskflowB.emplace([&](auto &subflow){
        num_tasks ++;
        auto num_spawn = rand() % 10 + 1;
        // Randomly spawn tasks
        for(auto i=0; i<num_spawn; i++) {
          subflow.emplace([&](){ num_tasks ++; });
        }    
        if(rand() % 2) {
          subflow.detach();
        }
        else {
          // In join mode, this task will be visited twice
          num_tasks ++;
        }
      }));
    }

    // Randomly specify dependency
    for(auto i=0; i < 64; i ++) {
      for(auto j=i+1; j < 64; j++) {
        if(rand()%2 == 0) {
          tasks[i].precede(tasks[j]);
        }
      }
    }

    executor.run_n(taskflowB, 16).get();

    if(w == 0) {
      REQUIRE(observer->num_tasks() == 0);
    }
    else {
      REQUIRE(observer->num_tasks() == num_tasks);
    }
  }
}


