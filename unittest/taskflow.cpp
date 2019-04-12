#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <vector>
#include <utility>
#include <chrono>
#include <limits.h>

// --------------------------------------------------------
// Testcase: Executor
// --------------------------------------------------------
TEST_CASE("Executor" * doctest::timeout(300)) {

  SUBCASE("Empty Executor") {
    REQUIRE_THROWS(tf::Taskflow{nullptr});
  }

  SUBCASE("Default Executor") {
    tf::Taskflow tf1;
    tf::Taskflow tf2;
    REQUIRE(tf1.share_executor() != nullptr);
    REQUIRE(tf2.share_executor() != nullptr);
    REQUIRE(tf1.share_executor() != tf2.share_executor());
  }

  SUBCASE("Shared Executor") {
    tf::Taskflow tf1;
    tf::Taskflow tf2(tf1.share_executor());
    REQUIRE(tf1.share_executor() == tf2.share_executor());
  }

  SUBCASE("Custom Executor") {
    auto executor = std::make_shared<tf::Taskflow::Executor>(4);
    tf::Taskflow tf1(executor);
    tf::Taskflow tf2(executor);
    REQUIRE(executor != nullptr);
    REQUIRE(executor.use_count() == 3);
    auto e1 = tf1.share_executor();
    auto e2 = tf2.share_executor();
    REQUIRE(e1 == executor);
    REQUIRE(e2 == executor);
    REQUIRE(executor.use_count() == 5);
  }

  SUBCASE("Shared Dispatch") {
    
    for(int t=0; t<=4; ++t) {

      std::mutex mutex;
      std::unordered_set<std::thread::id> threads;
      std::atomic<int> counter {0};

      tf::Taskflow tf1(t);
      tf::Taskflow tf2(tf1.share_executor());
      tf::Taskflow tf3(tf2.share_executor());
      tf::Taskflow tf4(tf2.share_executor());

      for(int n = 0; n<10000; ++n) {

        tf1.emplace([&] () {
          std::scoped_lock lock(mutex);
          threads.insert(std::this_thread::get_id());
          counter.fetch_add(1, std::memory_order_relaxed);
        });

        tf2.emplace([&] () {
          std::scoped_lock lock(mutex);
          threads.insert(std::this_thread::get_id());
          counter.fetch_add(1, std::memory_order_relaxed);
        });
        
        tf3.emplace([&] () {
          std::scoped_lock lock(mutex);
          threads.insert(std::this_thread::get_id());
          counter.fetch_add(1, std::memory_order_relaxed);
        });
        
        tf4.emplace([&] () {
          std::scoped_lock lock(mutex);
          threads.insert(std::this_thread::get_id());
          counter.fetch_add(1, std::memory_order_relaxed);
        });
      }

      auto f1 = tf1.dispatch();
      auto f2 = tf2.dispatch();
      auto f3 = tf3.dispatch();
      auto f4 = tf4.dispatch();

      f1.get();
      f2.get();
      f3.get();
      f4.get();

      auto max = t == 0 ? 1 : t;

      REQUIRE(counter == 40000);
      REQUIRE(threads.size() <= max + 1);
    }
  }
}

// --------------------------------------------------------
// Testcase: Builder
// --------------------------------------------------------
TEST_CASE("Builder" * doctest::timeout(300)) {

  size_t num_workers = 4;
  size_t num_tasks = 100;

  tf::Taskflow tf(static_cast<unsigned>(num_workers));
  REQUIRE(tf.num_workers() == num_workers);

  std::atomic<int> counter {0};
  std::vector<tf::Task> silent_tasks;
  std::vector<tf::Task> tasks;
  //std::vector<std::pair<tf::Task, std::future<void>>> tasks;

  SUBCASE("Placeholder") {
    
    for(size_t i=0; i<num_tasks; ++i) {
      silent_tasks.emplace_back(tf.placeholder().name(std::to_string(i)));
    }

    for(size_t i=0; i<num_tasks; ++i) {
      REQUIRE(silent_tasks[i].name() == std::to_string(i));
      REQUIRE(silent_tasks[i].num_dependents() == 0);
      REQUIRE(silent_tasks[i].num_successors() == 0);
    }

    for(auto& task : silent_tasks) {
      task.work([&counter](){ counter++; });
    }

    tf.wait_for_all();
    REQUIRE(counter == num_tasks);
  }

  SUBCASE("EmbarrassinglyParallel"){

    for(size_t i=0;i<num_tasks;i++) {
      tasks.emplace_back(tf.emplace([&counter]() {counter += 1;}));
    }

    REQUIRE(tf.num_nodes() == num_tasks);
    tf.wait_for_all();
    REQUIRE(counter == num_tasks);
    REQUIRE(tf.num_nodes() == 0);

    counter = 0;
    
    for(size_t i=0;i<num_tasks;i++){
      silent_tasks.emplace_back(tf.emplace([&counter]() {counter += 1;}));
    }

    REQUIRE(tf.num_nodes() == num_tasks);
    tf.wait_for_all();
    REQUIRE(counter == num_tasks);
    REQUIRE(tf.num_nodes() == 0);
  }
  
  SUBCASE("BinarySequence"){
    for(size_t i=0;i<num_tasks;i++){
      if(i%2 == 0){
        tasks.emplace_back(
          tf.emplace([&counter]() { REQUIRE(counter == 0); counter += 1;})
        );
      }
      else{
        tasks.emplace_back(
          tf.emplace([&counter]() { REQUIRE(counter == 1); counter -= 1;})
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
    tf.wait_for_all();
  }

  SUBCASE("LinearCounter"){
    for(size_t i=0;i<num_tasks;i++){
      tasks.emplace_back(
        tf.emplace([&counter, i]() { REQUIRE(counter == i); counter += 1;})
      );
      if(i>0){
        tf.precede(tasks[i-1], tasks[i]);
        //tf.precede(std::get<0>(tasks[i-1]), std::get<0>(tasks[i]));
      }
    }
    tf.wait_for_all();
    REQUIRE(counter == num_tasks);
    REQUIRE(tf.num_nodes() == 0);
  }
 
  SUBCASE("Broadcast"){
    auto src = tf.emplace([&counter]() {counter -= 1;});
    for(size_t i=1; i<num_tasks; i++){
      silent_tasks.emplace_back(
        tf.emplace([&counter]() {REQUIRE(counter == -1);})
      );
    }
    tf.broadcast(src, silent_tasks);
    tf.wait_for_all();
    REQUIRE(counter == - 1);
    REQUIRE(tf.num_nodes() == 0);
  }

  SUBCASE("Gather"){
    auto dst = tf.emplace([&counter, num_tasks]() { REQUIRE(counter == num_tasks - 1);});
    for(size_t i=1;i<num_tasks;i++){
      silent_tasks.emplace_back(tf.emplace([&counter]() {counter += 1;}));
    }
    dst.gather(silent_tasks);
    tf.wait_for_all();
    REQUIRE(counter == num_tasks - 1);
    REQUIRE(tf.num_nodes() == 0);
  }

  SUBCASE("MapReduce"){
    auto src = tf.emplace([&counter]() {counter = 0;});
    for(size_t i=0;i<num_tasks;i++){
      silent_tasks.emplace_back(tf.emplace([&counter]() {counter += 1;}));
    }
    tf.broadcast(src, silent_tasks);
    auto dst = tf.emplace(
      [&counter, num_tasks]() { REQUIRE(counter == num_tasks);}
    );
    tf.gather(silent_tasks, dst);
    tf.wait_for_all();
    REQUIRE(tf.num_nodes() == 0);
  }

  SUBCASE("Linearize"){
    for(size_t i=0;i<num_tasks;i++){
      silent_tasks.emplace_back(
        tf.emplace([&counter, i]() { REQUIRE(counter == i); counter += 1;})
      );
    }
    tf.linearize(silent_tasks);
    tf.wait_for_all();
    REQUIRE(counter == num_tasks);
    REQUIRE(tf.num_nodes() == 0);
  }

  SUBCASE("Kite"){
    auto src = tf.emplace([&counter]() {counter = 0;});
    for(size_t i=0;i<num_tasks;i++){
      silent_tasks.emplace_back(
        tf.emplace([&counter, i]() { REQUIRE(counter == i); counter += 1; })
      );
    }
    tf.broadcast(src, silent_tasks);
    tf.linearize(silent_tasks);
    auto dst = tf.emplace(
      [&counter, num_tasks]() { REQUIRE(counter == num_tasks);}
    );
    tf.gather(silent_tasks, dst);
    tf.wait_for_all();
    REQUIRE(tf.num_nodes() == 0);
  }
}

// --------------------------------------------------------
// Testcase: Dispatch
// --------------------------------------------------------
TEST_CASE("Dispatch" * doctest::timeout(300)) {
    
  using namespace std::chrono_literals;
  
  size_t num_workers = 4;
  size_t num_tasks = 100;
  
  tf::Taskflow tf(static_cast<unsigned>(num_workers));
  REQUIRE(tf.num_workers() == num_workers);

  std::atomic<int> counter {0};
  std::vector<tf::Task> silent_tasks;
    
  for(size_t i=0;i<num_tasks;i++){
    silent_tasks.emplace_back(tf.emplace([&counter]() {counter += 1;}));
  }

  SUBCASE("Dispatch"){
    auto fu = tf.dispatch();
    REQUIRE(tf.num_nodes() == 0);
    REQUIRE(fu.wait_for(1s) == std::future_status::ready);
    REQUIRE(counter == num_tasks);
  }

  SUBCASE("SilentDispatch"){
    tf.dispatch();
    REQUIRE(tf.num_nodes() == 0);
    std::this_thread::sleep_for(1s);
    REQUIRE(counter == num_tasks);
  }

  SUBCASE("WaitForAll") {
    tf.wait_for_all();
    REQUIRE(tf.num_nodes() == 0);
    REQUIRE(counter == num_tasks); 
  }
}

// --------------------------------------------------------
// Testcase: ParallelFor
// --------------------------------------------------------
TEST_CASE("ParallelFor" * doctest::timeout(300)) {
    
  using namespace std::chrono_literals;

  const auto mapper = [](size_t num_workers, size_t num_data, bool group){
    tf::Taskflow tf(static_cast<unsigned>(num_workers));
    std::vector<int> vec(num_data, 0);
    tf.parallel_for(vec.begin(), vec.end(), [] (int& v) { v = 64; }, group ? ::rand() : 0);
    for(const auto v : vec) {
      REQUIRE(v == 0);
    }
    tf.wait_for_all();
    for(const auto v : vec) {
      REQUIRE(v == 64);
    }
  };

  const auto reducer = [](size_t num_workers, size_t num_data, bool group){
    tf::Taskflow tf(static_cast<unsigned>(num_workers));
    std::vector<int> vec(num_data, 0);
    std::atomic<int> sum(0);
    tf.parallel_for(vec.begin(), vec.end(), [&](auto) { ++sum; }, group ? ::rand() : 0);
    REQUIRE(sum == 0);
    tf.wait_for_all();
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
    tf::Taskflow tf{num_workers};

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
    tf::Taskflow tf{num_workers};
    for(int beg=-10; beg<=10; ++beg) {
      for(int end=beg; end<=10; ++end) {
        for(int s=1; s<=end-beg; ++s) {
          int n = 0;
          for(int b = beg; b<end; b+=s) {
            ++n;
          }
          std::atomic<int> counter {0};
          tf.parallel_for(beg, end, s, [&] (auto) {
            counter.fetch_add(1, std::memory_order_relaxed);
          });
          tf.wait_for_all();
          REQUIRE(n == counter);
        }
      }
    }
  };
  
  auto negative_integer_step = [] (unsigned num_workers) {
    tf::Taskflow tf{num_workers};
    for(int beg=10; beg>=-10; --beg) {
      for(int end=beg; end>=-10; --end) {
        for(int s=1; s<=beg-end; ++s) {
          int n = 0;
          for(int b = beg; b>end; b-=s) {
            ++n;
          }
          std::atomic<int> counter {0};
          tf.parallel_for(beg, end, -s, [&] (auto) {
            counter.fetch_add(1, std::memory_order_relaxed);
          });
          tf.wait_for_all();
          REQUIRE(n == counter);
        }
      }
    }
  };
  
  auto positive_floating_step = [] (unsigned num_workers) {
    tf::Taskflow tf{num_workers};
    for(float beg=-10.0f; beg<=10.0f; ++beg) {
      for(float end=beg; end<=10.0f; ++end) {
        for(float s=1.0f; s<=end-beg; s+=0.1f) {
          int n = 0;
          for(float b = beg; b<end; b+=s) {
            ++n;
          }
          std::atomic<int> counter {0};
          tf.parallel_for(beg, end, s, [&] (auto) {
            counter.fetch_add(1, std::memory_order_relaxed);
          });
          tf.wait_for_all();
          REQUIRE(n == counter);
        }
      }
    }
  };
  
  auto negative_floating_step = [] (unsigned num_workers) {
    tf::Taskflow tf{num_workers};
    for(float beg=10.0f; beg>=-10.0f; --beg) {
      for(float end=beg; end>=-10.0f; --end) {
        for(float s=1.0f; s<=beg-end; s+=0.1f) {
          int n = 0;
          for(float b = beg; b>end; b-=s) {
            ++n;
          }
          std::atomic<int> counter {0};
          tf.parallel_for(beg, end, -s, [&] (auto) {
            counter.fetch_add(1, std::memory_order_relaxed);
          });
          tf.wait_for_all();
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
    tf::Taskflow tf(static_cast<unsigned>(num_workers));
    int result {0};
    std::iota(data.begin(), data.end(), 1);
    tf.reduce(data.begin(), data.end(), result, std::plus<int>());
    tf.wait_for_all();
    REQUIRE(result == std::accumulate(data.begin(), data.end(), 0, std::plus<int>()));
  };

  const auto multiply_test = [](size_t num_workers, auto &&data){
    tf::Taskflow tf(static_cast<unsigned>(num_workers));
    std::fill(data.begin(), data.end(), 1.0);
    double result {2.0};
    tf.reduce(data.begin(), data.end(), result, std::multiplies<double>());
    tf.wait_for_all();
    REQUIRE(result == std::accumulate(data.begin(), data.end(), 2.0, std::multiplies<double>()));
  };

  const auto max_test = [](size_t num_workers, auto &&data){
    tf::Taskflow tf(static_cast<unsigned>(num_workers));
    std::iota(data.begin(), data.end(), 1);
    int result {0};
    auto lambda = [](const auto& l, const auto& r){return std::max(l, r);};
    tf.reduce(data.begin(), data.end(), result, lambda);
    tf.wait_for_all();
    REQUIRE(result == std::accumulate(data.begin(), data.end(), 0, lambda));
  };

  const auto min_test = [](size_t num_workers, auto &&data){
    tf::Taskflow tf(static_cast<unsigned>(num_workers));
    std::iota(data.begin(), data.end(), 1);
    int result {std::numeric_limits<int>::max()};
    auto lambda = [](const auto& l, const auto& r){return std::min(l, r);};
    tf.reduce(data.begin(), data.end(), result, lambda);
    tf.wait_for_all();
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
    tf::Taskflow tf(w);
    for(int i=0; i<=65536; i = (i <= 1024) ? i + 1 : i*2 + 1) {
      std::vector<int> data(i);
      int gold = std::numeric_limits<int>::max();
      int test = std::numeric_limits<int>::max();
      for(auto& d : data) {
        d = ::rand();
        gold = std::min(gold, d);
      }
      tf.reduce_min(data.begin(), data.end(), test);
      tf.wait_for_all();
      REQUIRE(test == gold);
    }
  }

}

// --------------------------------------------------------
// Testcase: ReduceMax
// --------------------------------------------------------
TEST_CASE("ReduceMax" * doctest::timeout(300)) {

  for(int w=0; w<=4; w++) {
    tf::Taskflow tf(w);
    for(int i=0; i<=65536; i = (i <= 1024) ? i + 1 : i*2 + 1) {
      std::vector<int> data(i);
      int gold = std::numeric_limits<int>::min();
      int test = std::numeric_limits<int>::min();
      for(auto& d : data) {
        d = ::rand();
        gold = std::max(gold, d);
      }
      tf.reduce_max(data.begin(), data.end(), test);
      tf.wait_for_all();
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

      tf::Taskflow tf(W);
      
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

      tf.dispatch().get();
    } // End of for loop
  }
  
  // Mixed intra- and inter- operations
  SUBCASE("Complex") {

    for(unsigned W=0; W<=4; ++W) {

      tf::Taskflow tf(W);

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

      tf.wait_for_all();
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

      tf::Taskflow tf(W);
      
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

      tf.dispatch().get();

      REQUIRE(fu3v  == 10);
      REQUIRE(fu3v_ == 4);
    }
  }
}

// --------------------------------------------------------
// Testcase: Framework
// --------------------------------------------------------
TEST_CASE("Framework" * doctest::timeout(300)) {

  // Empty subflow test
  for(unsigned W=0; W<=4; ++W) {

    std::atomic<size_t> count {0};
    tf::Framework f;
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

    tf::Taskflow tf(W);
    std::list<std::shared_future<void>> fu_list;
    for(size_t i=0; i<500; i++) {
      if(i == 499) {
        tf.run(f).get();   // Synchronize the first 500 runs
        tf.run_n(f, 500);  // Run 500 times more
      }
      else if(i % 2) {
        fu_list.push_back(tf.run(f));
      }
      else {
        fu_list.push_back(tf.run(f, [&, i=i](){ REQUIRE(count == (i+1)*7); }));
      }
    }

    for(auto& fu: fu_list) {
      REQUIRE(fu.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
    }

    tf.wait_for_all();

    REQUIRE(count == 7000);
  }

  // TODO: test correctness when framework got changed between runs 
  for(unsigned W=0; W<=4; ++W) {

    std::atomic<size_t> count {0};
    tf::Framework f;
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

    tf::Taskflow tf(W);
    tf.run_n(f, 10).get();
    REQUIRE(count == 70);    

    auto E = f.emplace([](){});
    D.precede(E);
    tf.run_n(f, 10).get();
    REQUIRE(count == 140);    

    auto F = f.emplace([](){});
    E.precede(F);
    tf.run_n(f, 10);
    tf.wait_for_all();
    REQUIRE(count == 210);    
  }


  // Test run_until 
  for(unsigned W=0; W<=4; ++W) {

    std::atomic<size_t> count {0};
    tf::Framework f;
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

    tf::Taskflow tf(W);
    tf.run_until(f, [run=10]() mutable { return run-- == 0; }, 
      [&](){
        REQUIRE(count == 70);
        count = 0;
      }
    ).get();


    tf.run_until(f, [run=10]() mutable { return run-- == 0; }, 
      [&](){
        REQUIRE(count == 70);
        count = 0;
        auto E = f.emplace([&](){ count ++; });
        auto F = f.emplace([&](){ count ++; });
        A.precede(E).precede(F);
    });

    tf.run_until(f, [run=10]() mutable { return run-- == 0; }, 
      [&](){
        REQUIRE(count == 90);
        count = 0;
      }
    ).get();
  }

}

// --------------------------------------------------------
// Testcase: Composition
// --------------------------------------------------------
TEST_CASE("Composition-1" * doctest::timeout(300)) {

  for(unsigned w=0; w<=8; ++w) {

    tf::Taskflow taskflow;

    tf::Framework f0;

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

    tf::Framework f1;
    
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
    
    taskflow.run(f1).get();
    REQUIRE(cnt == 10);

    cnt = 0;
    taskflow.run_n(f1, 100).get();
    REQUIRE(cnt == 10 * 100);

    auto m1_2 = f1.composed_of(f0);
    m1_1.precede(m1_2);
    
    for(int n=0; n<100; n++) {
      cnt = 0;
      taskflow.run_n(f1, n).get();
      REQUIRE(cnt == 15*n);
    }

    cnt = 0;
    for(int n=0; n<100; n++) {
      taskflow.run(f1);
    }
    
    taskflow.wait_for_all();

    REQUIRE(cnt == 1500);
  }
}

// TESTCASE: composition-2
TEST_CASE("Composition-2" * doctest::timeout(300)) {

  for(unsigned w=0; w<=8; ++w) {

    tf::Taskflow taskflow {w};

    int cnt {0};
    
    // level 0 (+5)
    tf::Framework f0;

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
    tf::Framework f1;
    auto m1_1 = f1.composed_of(f0);
    auto m1_2 = f1.composed_of(f0);
    m1_1.precede(m1_2);

    // level 2 (+20)
    tf::Framework f2;
    auto m2_1 = f2.composed_of(f1);
    auto m2_2 = f2.composed_of(f1);
    m2_1.precede(m2_2);
    
    // synchronous run
    for(int n=0; n<100; n++) {
      cnt = 0;
      taskflow.run_n(f2, n).get();
      REQUIRE(cnt == 20*n);
    }

    // asynchronous run
    cnt = 0;
    for(int n=0; n<100; n++) {
      taskflow.run(f2);
    }
    taskflow.wait_for_all();
    REQUIRE(cnt == 100*20);
  }
}

// TESTCASE: composition-3
TEST_CASE("Composition-3" * doctest::timeout(300)) {

  
  for(unsigned w=0; w<=8; ++w) {

    tf::Taskflow taskflow {w};

    int cnt {0};
    
    // level 0 (+2)
    tf::Framework f0;

    auto [A, B] = f0.emplace(
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; }
    );
    A.precede(B);

    // level 1 (+4)
    tf::Framework f1;
    auto m1_1 = f1.composed_of(f0);
    auto m1_2 = f1.composed_of(f0);
    m1_1.precede(m1_2);

    // level 2 (+8)
    tf::Framework f2;
    auto m2_1 = f2.composed_of(f1);
    auto m2_2 = f2.composed_of(f1);
    m2_1.precede(m2_2);

    // level 3 (+16)
    tf::Framework f3;
    auto m3_1 = f3.composed_of(f2);
    auto m3_2 = f3.composed_of(f2);
    m3_1.precede(m3_2);

    // synchronous run
    for(int n=0; n<100; n++) {
      cnt = 0;
      taskflow.run_n(f3, n).get();
      REQUIRE(cnt == 16*n);
    }

    // asynchronous run
    cnt = 0;
    for(int n=0; n<100; n++) {
      taskflow.run(f3);
    }
    taskflow.wait_for_all();
    REQUIRE(cnt == 16*100);
  }
}














