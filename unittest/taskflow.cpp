#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"

#include <taskflow.hpp>
#include <vector>
#include <utility>
#include <chrono>
#include <limits.h>

// --------------------------------------------------------
// Testcase: Taskflow.Builder
// --------------------------------------------------------
TEST_CASE("Taskflow.Builder" * doctest::timeout(5)){

  constexpr auto num_workers = 4;
  constexpr auto num_tasks = 100;

  tf::Taskflow tf(num_workers);
  REQUIRE(tf.num_workers() == num_workers);

  std::atomic<int> counter {0};
  std::vector<tf::Task> silent_tasks;
  std::vector<std::pair<tf::Task, std::future<void>>> tasks;

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
      silent_tasks.emplace_back(tf.silent_emplace([&counter]() {counter += 1;}));
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
        tasks[i-1].first.precede(tasks[i].first);
      }

      if(i==0) {
        REQUIRE(tasks[i].first.num_dependents() == 0);
      }
      else {
        REQUIRE(tasks[i].first.num_dependents() == 1);
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
        tf.precede(std::get<0>(tasks[i-1]), std::get<0>(tasks[i]));
      }
    }
    tf.wait_for_all();
    REQUIRE(counter == num_tasks);
    REQUIRE(tf.num_nodes() == 0);
  }
 
  SUBCASE("Broadcast"){
    auto src = tf.silent_emplace([&counter]() {counter -= 1;});
    for(size_t i=1; i<num_tasks; i++){
      silent_tasks.emplace_back(
        tf.silent_emplace([&counter]() {REQUIRE(counter == -1);})
      );
    }
    tf.broadcast(src, silent_tasks);
    tf.wait_for_all();
    REQUIRE(counter == - 1);
    REQUIRE(tf.num_nodes() == 0);
  }

  SUBCASE("Gather"){
    auto dst = tf.silent_emplace([&counter, num_tasks]() { REQUIRE(counter == num_tasks - 1);});
    for(size_t i=1;i<num_tasks;i++){
      silent_tasks.emplace_back(tf.silent_emplace([&counter]() {counter += 1;}));
    }
    dst.gather(silent_tasks);
    tf.wait_for_all();
    REQUIRE(counter == num_tasks - 1);
    REQUIRE(tf.num_nodes() == 0);
  }

  SUBCASE("MapReduce"){
    auto src = tf.silent_emplace([&counter]() {counter = 0;});
    for(size_t i=0;i<num_tasks;i++){
      silent_tasks.emplace_back(tf.silent_emplace([&counter]() {counter += 1;}));
    }
    tf.broadcast(src, silent_tasks);
    auto dst = tf.silent_emplace(
      [&counter, num_tasks]() { REQUIRE(counter == num_tasks);}
    );
    tf.gather(silent_tasks, dst);
    tf.wait_for_all();
    REQUIRE(tf.num_nodes() == 0);
  }

  SUBCASE("Linearize"){
    for(size_t i=0;i<num_tasks;i++){
      silent_tasks.emplace_back(
        tf.silent_emplace([&counter, i]() { REQUIRE(counter == i); counter += 1;})
      );
    }
    tf.linearize(silent_tasks);
    tf.wait_for_all();
    REQUIRE(counter == num_tasks);
    REQUIRE(tf.num_nodes() == 0);
  }

  SUBCASE("Kite"){
    auto src = tf.silent_emplace([&counter]() {counter = 0;});
    for(size_t i=0;i<num_tasks;i++){
      silent_tasks.emplace_back(
        tf.silent_emplace([&counter, i]() { REQUIRE(counter == i); counter += 1; })
      );
    }
    tf.broadcast(src, silent_tasks);
    tf.linearize(silent_tasks);
    auto dst = tf.silent_emplace(
      [&counter, num_tasks]() { REQUIRE(counter == num_tasks);}
    );
    tf.gather(silent_tasks, dst);
    tf.wait_for_all();
    REQUIRE(tf.num_nodes() == 0);
  }
}

// --------------------------------------------------------
// Testcase: Taskflow.Dispatch
// --------------------------------------------------------
TEST_CASE("Taskflow.Dispatch" * doctest::timeout(5)) {
    
  using namespace std::chrono_literals;
  
  constexpr auto num_workers = 4;
  constexpr auto num_tasks = 100;
  
  tf::Taskflow tf(num_workers);
  REQUIRE(tf.num_workers() == num_workers);

  std::atomic<int> counter {0};
  std::vector<tf::Task> silent_tasks;
    
  for(size_t i=0;i<num_tasks;i++){
    silent_tasks.emplace_back(tf.silent_emplace([&counter]() {counter += 1;}));
  }

  SUBCASE("Dispatch"){
    auto fu = tf.dispatch();
    REQUIRE(tf.num_nodes() == 0);
    REQUIRE(fu.wait_for(1s) == std::future_status::ready);
    REQUIRE(counter == num_tasks);
  }

  SUBCASE("SilentDispatch"){
    tf.silent_dispatch();
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
// Testcase: Taskflow.ParallelFor
// --------------------------------------------------------
TEST_CASE("Taskflow.ParallelFor" * doctest::timeout(5)) {
    
  using namespace std::chrono_literals;

  const auto mapper = [](size_t num_workers, size_t num_data, bool group){
    tf::Taskflow tf(num_workers);
    std::vector<int> vec(num_data, 0);
    tf.parallel_for(vec, [] (int& v) { v = 64; }, group ? ::rand() : 0);
    for(const auto v : vec) {
      REQUIRE(v == 0);
    }
    tf.wait_for_all();
    for(const auto v : vec) {
      REQUIRE(v == 64);
    }
  };

  const auto reducer = [](size_t num_workers, size_t num_data, bool group){
    tf::Taskflow tf(num_workers);
    std::vector<int> vec(num_data, 0);
    std::atomic<int> sum(0);
    tf.parallel_for(vec, [&](auto) { ++sum; }, group ? ::rand() : 0);
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
// Testcase: Taskflow.Reduce
// --------------------------------------------------------
TEST_CASE("Taskflow.Reduce" * doctest::timeout(5)) {

  const auto plus_test = [](const size_t num_workers, auto &&data){
    tf::Taskflow tf(num_workers);
    int result {0};
    std::iota(data.begin(), data.end(), 1);
    tf.reduce(data.begin(), data.end(), result, std::plus<int>());
    tf.wait_for_all();
    REQUIRE(result == std::accumulate(data.begin(), data.end(), 0, std::plus<int>()));
  };

  const auto multiply_test = [](const size_t num_workers, auto &&data){
    tf::Taskflow tf(num_workers);
    std::fill(data.begin(), data.end(), 1.0);
    double result {2.0};
    tf.reduce(data.begin(), data.end(), result, std::multiplies<double>());
    tf.wait_for_all();
    REQUIRE(result == std::accumulate(data.begin(), data.end(), 2.0, std::multiplies<double>()));
  };

  const auto max_test = [](const size_t num_workers, auto &&data){
    tf::Taskflow tf(num_workers);
    std::iota(data.begin(), data.end(), 1);
    int result {0};
    auto lambda = [](const auto& l, const auto& r){return std::max(l, r);};
    tf.reduce(data.begin(), data.end(), result, lambda);
    tf.wait_for_all();
    REQUIRE(result == std::accumulate(data.begin(), data.end(), 0, lambda));
  };

  const auto min_test = [](const size_t num_workers, auto &&data){
    tf::Taskflow tf(num_workers);
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
// Testcase: Taskflow.ReduceMin
// --------------------------------------------------------
TEST_CASE("Taskflow.ReduceMin" * doctest::timeout(5)) {

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
// Testcase: Taskflow.ReduceMax
// --------------------------------------------------------
TEST_CASE("Taskflow.ReduceMax" * doctest::timeout(5)) {

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
// Testcase: Taskflow.JoinedSubflow
// -------------------------------------------------------- 
TEST_CASE("Taskflow.JoinedSubflow" * doctest::timeout(5)){

  using namespace std::literals::chrono_literals;
  
  SUBCASE("Trivial") {
    // Empty subflow test
    for(unsigned W=0; W<=4; ++W) {

      tf::Taskflow tf(W);
      
      // empty flow with future
      tf::Task subflow3, subflow3_;
      std::future<int> fu3, fu3_;
      std::atomic<int> fu3v{0}, fu3v_{0};
      
      // empty flow
      auto subflow1 = tf.silent_emplace([&] (auto& fb) {
        fu3v++;
      }).name("subflow1");
      
      // nested empty flow
      auto subflow2 = tf.silent_emplace([&] (auto& fb) {
        fu3v++;
        fb.silent_emplace([&] (auto& fb) {
          fu3v++;
          fb.silent_emplace( [&] (auto& fb) {
            fu3v++;
          }).name("subflow2_1_1");
        }).name("subflow2_1");
      }).name("subflow2");
      
      std::tie(subflow3, fu3) = tf.emplace([&] (auto& fb) {

        REQUIRE(fu3v == 4);

        fu3v++;
        fu3v_++;
        
        std::tie(subflow3_, fu3_) = fb.emplace([&] (auto& fb) {
          REQUIRE(fu3v_ == 3);
          fu3v++;
          fu3v_++;
          return 200;
        });
        subflow3_.name("subflow3_");

        // hereafter we use 100us to avoid dangling reference ...
        auto s1 = fb.silent_emplace([&] () { 
          fu3v_++;
          fu3v++;
          REQUIRE(fu3.valid());
          REQUIRE(fu3.wait_for (100us) != std::future_status::ready);
          REQUIRE(fu3_.valid());
          REQUIRE(fu3_.wait_for(100us) != std::future_status::ready);
        }).name("s1");
        
        auto s2 = fb.silent_emplace([&] () {
          fu3v_++;
          fu3v++;
          REQUIRE(fu3.valid());
          REQUIRE(fu3.wait_for (100us) != std::future_status::ready);
          REQUIRE(fu3_.valid());
          REQUIRE(fu3_.wait_for(100us) != std::future_status::ready);
        }).name("s2");
        
        auto s3 = fb.silent_emplace([&] () {
          fu3v++;
          REQUIRE(fu3v_ == 4);
          REQUIRE(fu3.valid());
          REQUIRE(fu3.wait_for (100us) != std::future_status::ready);
          REQUIRE(fu3_.valid());
          REQUIRE(fu3_.wait_for(100us) == std::future_status::ready);
        }).name("s3");

        s1.precede(subflow3_);
        s2.precede(subflow3_);
        subflow3_.precede(s3);

        REQUIRE(fu3v_ == 1);

        return 100;
      });
      subflow3.name("subflow3");

      // empty flow to test future
      auto subflow4 = tf.silent_emplace([&] () {
        REQUIRE(fu3v == 9);
        REQUIRE(fu3.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
        fu3v++;
      }).name("subflow4");

      subflow1.precede(subflow2);
      subflow2.precede(subflow3);
      subflow3.precede(subflow4);

      tf.dispatch().get();

      REQUIRE(fu3v  == 10);
      REQUIRE(fu3v_ == 4);
      REQUIRE(fu3.get()  == 100);
      REQUIRE(fu3_.get() == 200);
    } // End of for loop
  }
  
  // Mixed intra- and inter- operations
  SUBCASE("Complex") {

    for(unsigned W=0; W<=4; ++W) {

      tf::Taskflow tf(W);

      std::vector<int> data;
      int sum {0};

      auto A = tf.silent_emplace([&data] () {
        for(int i=0; i<10; ++i) {
          data.push_back(1);
        }
      });

      std::atomic<size_t> count = 0;

      auto B = tf.silent_emplace([&count, &data, &sum](auto& fb){

        auto [src, tgt] = fb.reduce(data.begin(), data.end(), sum, std::plus<int>());

        fb.silent_emplace([&sum] () { REQUIRE(sum == 0); }).precede(src);

        tgt.precede(fb.silent_emplace([&sum] () { REQUIRE(sum == 10); }));

        for(size_t i=0; i<10; i ++){
          ++count;
        }

        auto n = fb.silent_emplace([&count](auto& fb){

          REQUIRE(count == 20);
          ++count;

          auto prev = fb.silent_emplace([&count](){
            REQUIRE(count == 21);
            ++count;
          });

          for(size_t i=0; i<10; i++){
            auto next = fb.silent_emplace([&count, i](){
              REQUIRE(count == 22+i);
              ++count;
            });
            prev.precede(next);
            prev = next;
          }
        });

        for(size_t i=0; i<10; i++){
          fb.silent_emplace([&count](){ ++count; }).precede(n);
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
// Testcase: Taskflow.DetachedSubflow
// --------------------------------------------------------
TEST_CASE("Taskflow.DetachedSubflow" * doctest::timeout(5)) {
  
  using namespace std::literals::chrono_literals;

  SUBCASE("Trivial") {

    // Empty subflow test
    for(unsigned W=0; W<=4; ++W) {

      tf::Taskflow tf(W);
      
      // empty flow with future
      tf::Task subflow3, subflow3_;
      std::future<int> fu3, fu3_;
      std::atomic<int> fu3v{0}, fu3v_{0};
      
      // empty flow
      auto subflow1 = tf.silent_emplace([&] (auto& fb) {
        fu3v++;
        fb.detach();
      }).name("subflow1");
      
      // nested empty flow
      auto subflow2 = tf.silent_emplace([&] (auto& fb) {
        fu3v++;
        fb.silent_emplace([&] (auto& fb) {
          fu3v++;
          fb.silent_emplace( [&] (auto& fb) {
            fu3v++;
          }).name("subflow2_1_1");
          fb.detach();
        }).name("subflow2_1");
        fb.detach();
      }).name("subflow2");
      
      std::tie(subflow3, fu3) = tf.emplace([&] (auto& fb) {

        REQUIRE((fu3v >= 2 && fu3v <= 4));

        fu3v++;
        fu3v_++;
        
        std::tie(subflow3_, fu3_) = fb.emplace([&] (auto& fb) {
          REQUIRE(fu3v_ == 3);
          fu3v++;
          fu3v_++;
          return 200;
        });
        subflow3_.name("subflow3_");

        // hereafter we use 100us to avoid dangling reference ...
        auto s1 = fb.silent_emplace([&] () { 
          fu3v_++;
          fu3v++;
          REQUIRE(fu3.valid());
          REQUIRE(fu3.wait_for (100us) == std::future_status::ready);
          REQUIRE(fu3_.valid());
          REQUIRE(fu3_.wait_for(100us) != std::future_status::ready);
        }).name("s1");
        
        auto s2 = fb.silent_emplace([&] () {
          fu3v_++;
          fu3v++;
          REQUIRE(fu3.valid());
          REQUIRE(fu3.wait_for (100us) == std::future_status::ready);
          REQUIRE(fu3_.valid());
          REQUIRE(fu3_.wait_for(100us) != std::future_status::ready);
        }).name("s2");
        
        auto s3 = fb.silent_emplace([&] () {
          fu3v++;
          REQUIRE(fu3v_ == 4);
          REQUIRE(fu3.valid());
          REQUIRE(fu3.wait_for (100us) == std::future_status::ready);
          REQUIRE(fu3_.valid());
          REQUIRE(fu3_.wait_for(100us) == std::future_status::ready);
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
      auto subflow4 = tf.silent_emplace([&] () {
        REQUIRE((fu3v >= 3 && fu3v <= 9));
        REQUIRE(fu3.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
        fu3v++;
      }).name("subflow4");

      subflow1.precede(subflow2);
      subflow2.precede(subflow3);
      subflow3.precede(subflow4);

      tf.dispatch().get();

      REQUIRE(fu3v  == 10);
      REQUIRE(fu3v_ == 4);
      REQUIRE(fu3.get()  == 100);
      REQUIRE(fu3_.get() == 200);
    }
  }

}


/*// --------------------------------------------------------
// Testcase: Taskflow.ParallelRange
// --------------------------------------------------------
TEST_CASE("Taskflow.ParallelRange" * doctest::timeout(5)) {
    
  using namespace std::chrono_literals;

  const auto mapper = [](size_t num_workers, size_t num_data, bool group){
    tf::Taskflow tf(num_workers);
    std::vector<int> vec(num_data, 0);
    tf.parallel_range(0ul, num_data, [&] (size_t i) { vec[i] = 64; }, group ? ::rand() : 0);
    for(const auto v : vec) {
      REQUIRE(v == 0);
    }
    tf.wait_for_all();
    for(const auto v : vec) {
      REQUIRE(v == 64);
    }
  };

  const auto reducer = [](size_t num_workers, size_t num_data, bool group){
    tf::Taskflow tf(num_workers);
    std::atomic<int> sum(0);
    tf.parallel_range(0ul, num_data, [&](size_t i) { sum += i; }, group ? ::rand() : 0);
    REQUIRE(sum == 0);
    tf.wait_for_all();
    REQUIRE(sum == (num_data-1)*num_data/2);
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
}*/
