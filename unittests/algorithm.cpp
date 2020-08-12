#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <vector>
#include <utility>
#include <chrono>
#include <limits.h>

// --------------------------------------------------------
// Testcase: parallel_for
// --------------------------------------------------------

void parallel_for(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;
  
  std::vector<int> vec(1024);
  for(int n = 0; n <= 150; n++) {

    std::fill_n(vec.begin(), vec.size(), -1);

    int beg = ::rand()%300 - 150;
    int end = beg + n;

    for(int s=1; s<=n+1; s = s + ::rand()%17 + 1) {
      taskflow.clear();
      std::atomic<int> counter {0};

      taskflow.parallel_for(beg, end, s, [&](int i){
        counter++;
        vec[i-beg] = i;
      });

      executor.run(taskflow).wait();
      REQUIRE(counter == (n + s - 1) / s);

      for(int i=beg; i<end; i+=s) {
        REQUIRE(vec[i-beg] == i);
        vec[i-beg] = -1;
      }

      for(const auto i : vec) {
        REQUIRE(i == -1);
      }
    }
  }

  for(size_t n = 0; n < 150; n++) {
    
    std::fill_n(vec.begin(), vec.size(), -1);

    taskflow.clear();
    std::atomic<int> counter {0};

    taskflow.parallel_for(vec.begin(), vec.begin() + n, [&](int& i){
      counter++;
      i = 1;
    });

    executor.run(taskflow).wait();
    REQUIRE(counter == n);

    for(size_t i=0; i<n; ++i) {
      REQUIRE(vec[i] == 1);
    }

    for(size_t i=n; i<vec.size(); ++i) {
      REQUIRE(vec[i] == -1);
    }
  }
}

TEST_CASE("pf.1thread" * doctest::timeout(300)) {
  parallel_for(1);
}

TEST_CASE("pf.2threads" * doctest::timeout(300)) {
  parallel_for(2);
}

TEST_CASE("pf.3threads" * doctest::timeout(300)) {
  parallel_for(3);
}

TEST_CASE("pf.4threads" * doctest::timeout(300)) {
  parallel_for(4);
}

TEST_CASE("pf.5threads" * doctest::timeout(300)) {
  parallel_for(5);
}

TEST_CASE("pf.6threads" * doctest::timeout(300)) {
  parallel_for(6);
}

TEST_CASE("pf.7threads" * doctest::timeout(300)) {
  parallel_for(7);
}

TEST_CASE("pf.8threads" * doctest::timeout(300)) {
  parallel_for(8);
}

TEST_CASE("pf.9threads" * doctest::timeout(300)) {
  parallel_for(9);
}

TEST_CASE("pf.10threads" * doctest::timeout(300)) {
  parallel_for(10);
}

TEST_CASE("pf.11threads" * doctest::timeout(300)) {
  parallel_for(11);
}

TEST_CASE("pf.12threads" * doctest::timeout(300)) {
  parallel_for(12);
}

// ----------------------------------------------------------------------------
// stateful_parallel_for
// ----------------------------------------------------------------------------

void stateful_parallel_for(unsigned W) {
  
  tf::Executor executor(W);
  tf::Taskflow taskflow;
  std::vector<int> vec;
  std::atomic<int> counter {0};
  
  for(size_t n = 0; n < 1024; n++) {
  
    std::vector<int>::iterator beg, end;
    size_t ibeg = 0, iend = 0;
    size_t half = n/2;
    
    taskflow.clear();
    
    auto init = taskflow.emplace([&](){ 
      vec.resize(n);
      std::fill_n(vec.begin(), vec.size(), -1);

      beg = vec.begin();
      end = beg + half;

      ibeg = half;
      iend = n;

      counter = 0;
    });

    auto pf1 = taskflow.parallel_for(std::ref(beg), std::ref(end), [&](int& i){
      counter++;
      i = 8;
    });

    auto pf2 = taskflow.parallel_for(
      std::ref(ibeg), std::ref(iend), size_t{1}, [&] (size_t i) {
        counter++;
        vec[i] = -8;
      }
    );

    init.precede(pf1, pf2);

    executor.run(taskflow).wait();
    REQUIRE(counter == n);

    for(size_t i=0; i<half; ++i) {
      REQUIRE(vec[i] == 8);
      vec[i] = 0;
    }

    for(size_t i=half; i<n; ++i) {
      REQUIRE(vec[i] == -8);
      vec[i] = 0;
    }
  }
}


TEST_CASE("statefulpf.1thread" * doctest::timeout(300)) {
  stateful_parallel_for(1);
}

TEST_CASE("statefulpf.2threads" * doctest::timeout(300)) {
  stateful_parallel_for(2);
}

TEST_CASE("statefulpf.3threads" * doctest::timeout(300)) {
  stateful_parallel_for(3);
}

TEST_CASE("statefulpf.4threads" * doctest::timeout(300)) {
  stateful_parallel_for(4);
}

TEST_CASE("statefulpf.5threads" * doctest::timeout(300)) {
  stateful_parallel_for(5);
}

TEST_CASE("statefulpf.6threads" * doctest::timeout(300)) {
  stateful_parallel_for(6);
}

TEST_CASE("statefulpf.7threads" * doctest::timeout(300)) {
  stateful_parallel_for(7);
}

TEST_CASE("statefulpf.8threads" * doctest::timeout(300)) {
  stateful_parallel_for(8);
}

TEST_CASE("statefulpf.9threads" * doctest::timeout(300)) {
  stateful_parallel_for(9);
}

TEST_CASE("statefulpf.10threads" * doctest::timeout(300)) {
  stateful_parallel_for(10);
}

TEST_CASE("statefulpf.11threads" * doctest::timeout(300)) {
  stateful_parallel_for(11);
}

TEST_CASE("statefulpf.12threads" * doctest::timeout(300)) {
  stateful_parallel_for(12);
}











