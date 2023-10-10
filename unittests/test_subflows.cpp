#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/reduce.hpp>

// --------------------------------------------------------
// Testcase: JoinedSubflow
// --------------------------------------------------------

void joined_subflow(unsigned W) {

  using namespace std::literals::chrono_literals;

  SUBCASE("Trivial") {
    tf::Executor executor(W);
    tf::Taskflow tf;

    // empty flow with future
    tf::Task subflow3, subflow3_;
    //std::future<int> fu3, fu3_;
    std::atomic<int> fu3v{0}, fu3v_{0};

    // empty flow
    auto subflow1 = tf.emplace([&] (tf::Subflow& fb) {
      fu3v++;
      fb.join();
    }).name("subflow1");

    // nested empty flow
    auto subflow2 = tf.emplace([&] (tf::Subflow& fb) {
      fu3v++;
      fb.emplace([&] (tf::Subflow& fb2) {
        fu3v++;
        fb2.emplace( [&] (tf::Subflow& fb3) {
          fu3v++;
          fb3.join();
        }).name("subflow2_1_1");
      }).name("subflow2_1");
    }).name("subflow2");

    subflow3 = tf.emplace([&] (tf::Subflow& fb) {

      REQUIRE(fu3v == 4);

      fu3v++;
      fu3v_++;

      subflow3_ = fb.emplace([&] (tf::Subflow& fb2) {
        REQUIRE(fu3v_ == 3);
        fu3v++;
        fu3v_++;
        //return 200;
        fb2.join();
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
    // End of for loop
  }

  // Mixed intra- and inter- operations
  SUBCASE("Complex") {
    tf::Executor executor(W);
    tf::Taskflow tf;

    std::vector<int> data;
    int sum {0};

    auto A = tf.emplace([&data] () {
      for(int i=0; i<10; ++i) {
        data.push_back(1);
      }
    });

    std::atomic<size_t> count {0};

    auto B = tf.emplace([&count, &data, &sum](tf::Subflow& fb){

      //auto [src, tgt] = fb.reduce(data.begin(), data.end(), sum, std::plus<int>());
      auto task = fb.reduce(data.begin(), data.end(), sum, std::plus<int>());

      fb.emplace([&sum] () { REQUIRE(sum == 0); }).precede(task);

      task.precede(fb.emplace([&sum] () { REQUIRE(sum == 10); }));

      for(size_t i=0; i<10; i ++){
        ++count;
      }

      auto n = fb.emplace([&count](tf::Subflow& fb2){

        REQUIRE(count == 20);
        ++count;

        auto prev = fb2.emplace([&count](){
          REQUIRE(count == 21);
          ++count;
        });

        for(size_t i=0; i<10; i++){
          auto next = fb2.emplace([&count, i](){
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

TEST_CASE("JoinedSubflow.1thread" * doctest::timeout(300)){
  joined_subflow(1);
}

TEST_CASE("JoinedSubflow.2threads" * doctest::timeout(300)){
  joined_subflow(2);
}

TEST_CASE("JoinedSubflow.3threads" * doctest::timeout(300)){
  joined_subflow(3);
}

TEST_CASE("JoinedSubflow.4threads" * doctest::timeout(300)){
  joined_subflow(4);
}

TEST_CASE("JoinedSubflow.5threads" * doctest::timeout(300)){
  joined_subflow(5);
}

TEST_CASE("JoinedSubflow.6threads" * doctest::timeout(300)){
  joined_subflow(6);
}

TEST_CASE("JoinedSubflow.7threads" * doctest::timeout(300)){
  joined_subflow(7);
}

TEST_CASE("JoinedSubflow.8threads" * doctest::timeout(300)){
  joined_subflow(8);
}

// --------------------------------------------------------
// Testcase: DetachedSubflow
// --------------------------------------------------------

void detached_subflow(unsigned W) {

  using namespace std::literals::chrono_literals;

  SUBCASE("Trivial") {
    tf::Executor executor(W);
    tf::Taskflow tf;

    // empty flow with future
    tf::Task subflow3, subflow3_;
    std::atomic<int> fu3v{0}, fu3v_{0};

    // empty flow
    auto subflow1 = tf.emplace([&] (tf::Subflow& fb) {
      fu3v++;
      fb.detach();
    }).name("subflow1");

    // nested empty flow
    auto subflow2 = tf.emplace([&] (tf::Subflow& fb) {
      fu3v++;
      fb.emplace([&] (tf::Subflow& fb2) {
        fu3v++;
        fb2.emplace( [&] (tf::Subflow& fb3) {
          fu3v++;
          fb3.join();
        }).name("subflow2_1_1");
        fb2.detach();
      }).name("subflow2_1");
      fb.detach();
    }).name("subflow2");

    subflow3 = tf.emplace([&] (tf::Subflow& fb) {

      REQUIRE((fu3v >= 2 && fu3v <= 4));

      fu3v++;
      fu3v_++;

      subflow3_ = fb.emplace([&] (tf::Subflow& fb2) {
        REQUIRE(fu3v_ == 3);
        fu3v++;
        fu3v_++;
        fb2.join();
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

      //return 100;
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

TEST_CASE("DetachedSubflow.1thread" * doctest::timeout(300)) {
  detached_subflow(1);
}

TEST_CASE("DetachedSubflow.2threads" * doctest::timeout(300)) {
  detached_subflow(2);
}

TEST_CASE("DetachedSubflow.3threads" * doctest::timeout(300)) {
  detached_subflow(3);
}

TEST_CASE("DetachedSubflow.4threads" * doctest::timeout(300)) {
  detached_subflow(4);
}

TEST_CASE("DetachedSubflow.5threads" * doctest::timeout(300)) {
  detached_subflow(5);
}

TEST_CASE("DetachedSubflow.6threads" * doctest::timeout(300)) {
  detached_subflow(6);
}

TEST_CASE("DetachedSubflow.7threads" * doctest::timeout(300)) {
  detached_subflow(7);
}

TEST_CASE("DetachedSubflow.8threads" * doctest::timeout(300)) {
  detached_subflow(8);
}


// --------------------------------------------------------
// Testcase: TreeSubflow
// --------------------------------------------------------
void detach_spawn(const int max_depth, std::atomic<int>& counter, int depth, tf::Subflow& subflow)  {
  if(depth < max_depth) {
    counter.fetch_add(1, std::memory_order_relaxed);
    subflow.emplace([&, max_depth, depth=depth+1](tf::Subflow& sfl){
      detach_spawn(max_depth, counter, depth, sfl); }
    );
    subflow.emplace([&, max_depth, depth=depth+1](tf::Subflow& sfr){
      detach_spawn(max_depth, counter, depth, sfr); }
    );
    subflow.detach();
  }
}

void join_spawn(const int max_depth, std::atomic<int>& counter, int depth, tf::Subflow& subflow)  {
  if(depth < max_depth) {
    counter.fetch_add(1, std::memory_order_relaxed);
    subflow.emplace([&, max_depth, depth=depth+1](tf::Subflow& sfl){
      join_spawn(max_depth, counter, depth, sfl); }
    );
    subflow.emplace([&, max_depth, depth=depth+1](tf::Subflow& sfr){
      join_spawn(max_depth, counter, depth, sfr); }
    );
  }
}

void mix_spawn(
  const int max_depth, std::atomic<int>& counter, int depth, tf::Subflow& subflow
) {

  if(depth < max_depth) {
    auto ret = counter.fetch_add(1, std::memory_order_relaxed);
    subflow.emplace([&, max_depth, depth=depth+1](tf::Subflow& sfl){
      mix_spawn(max_depth, counter, depth, sfl); }
    ).name(std::string("left") + std::to_string(ret%2));
    subflow.emplace([&, max_depth, depth=depth+1](tf::Subflow& sfr){
      mix_spawn(max_depth, counter, depth, sfr); }
    ).name(std::string("right") + std::to_string(ret%2));
    if(ret % 2) {
      subflow.detach();
    }
  }
}

TEST_CASE("TreeSubflow" * doctest::timeout(300)) {

  SUBCASE("AllDetach") {
    constexpr int max_depth {10};
    for(int W=1; W<=4; W++) {
      std::atomic<int> counter {0};
      tf::Taskflow tf;
      tf.emplace([&](tf::Subflow& subflow){
        detach_spawn(max_depth, counter, 0, subflow);
      });

      tf::Executor executor(W);
      executor.run(tf).get();
      REQUIRE(counter == (1<<max_depth) - 1);
    }
  }


  SUBCASE("AllJoin") {
    constexpr int max_depth {10};
    for(int W=1; W<=4; W++) {
      std::atomic<int> counter {0};
      tf::Taskflow tf;
      tf.emplace([&](tf::Subflow& subflow){
        join_spawn(max_depth, counter, 0, subflow);
      });
      tf::Executor executor(W);
      executor.run(tf).get();
      REQUIRE(counter == (1<<max_depth) - 1);
    }
  }

  SUBCASE("Mix") {
    constexpr int max_depth {10};
    for(int W=1; W<=4; W++) {
      std::atomic<int> counter {0};
      tf::Taskflow tf;
      tf.emplace([&](tf::Subflow& subflow){
        mix_spawn(max_depth, counter, 0, subflow);
      }).name("top task");

      tf::Executor executor(W);
      executor.run(tf).get();
      REQUIRE(counter == (1<<max_depth) - 1);
    }
  }
}

// --------------------------------------------------------
// Testcase: FibSubflow
// --------------------------------------------------------
int fibonacci_spawn(int n, tf::Subflow& sbf) {
  if (n < 2) return n;
  int res1, res2;
  sbf.emplace([&res1, n] (tf::Subflow& sbfl) { res1 = fibonacci_spawn(n - 1, sbfl); } );
  sbf.emplace([&res2, n] (tf::Subflow& sbfr) { res2 = fibonacci_spawn(n - 2, sbfr); } );
  REQUIRE(sbf.joinable() == true);
  sbf.join();
  REQUIRE(sbf.joinable() == false);
  return res1 + res2;
}

void fibonacci(size_t W) {

  int N = 20;
  int res = -1;  // result

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  taskflow.emplace([&res, N] (tf::Subflow& sbf) {
    res = fibonacci_spawn(N, sbf);
  });

  executor.run(taskflow).wait();

  REQUIRE(res == 6765);
}

TEST_CASE("FibSubflow.1thread") {
  fibonacci(1);
}

TEST_CASE("FibSubflow.2threads") {
  fibonacci(2);
}

TEST_CASE("FibSubflow.4threads") {
  fibonacci(4);
}

TEST_CASE("FibSubflow.5threads") {
  fibonacci(5);
}

TEST_CASE("FibSubflow.6threads") {
  fibonacci(6);
}

TEST_CASE("FibSubflow.7threads") {
  fibonacci(7);
}

TEST_CASE("FibSubflow.8threads") {
  fibonacci(8);
}
