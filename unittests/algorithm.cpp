#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <vector>
#include <utility>
#include <chrono>
#include <limits.h>

// --------------------------------------------------------
// Testcase: Parallel For Static
// --------------------------------------------------------

void pfs(unsigned W) {
  
  tf::Executor executor(W);
  tf::Taskflow taskflow;
  
  for(int n = 0; n < 150; n++) {
    int beg = 0;
    int end = beg + n;
    for(int s=1; s<=n+1; s = s + ::rand()%17 + 1) {
      for(int c=0; c<=65; c = (c << 1) + 1) {
        taskflow.clear();
        std::atomic<int> counter {0};
        taskflow.parallel_for_static(beg, end, s, [&](auto){
          counter++;
        }, c);
        executor.run(taskflow).wait();
        REQUIRE(counter == (n + s - 1) / s);
      }
    }
  }

  std::list<int> list;
  for(int n = 0; n < 150; n++) {
    for(int c=0; c<=65; c = (c << 1) + 1) {
      taskflow.clear();
      std::atomic<int> counter {0};
      taskflow.parallel_for_static(list.begin(), list.end(), [&](auto){
        counter++;
      }, c);
      executor.run(taskflow).wait();
      REQUIRE(counter == n);
    }
    list.push_back(n);
  }
}

TEST_CASE("pfs.1thread" * doctest::timeout(300)) {
  pfs(1);
}

TEST_CASE("pfs.2threads" * doctest::timeout(300)) {
  pfs(2);
}

TEST_CASE("pfs.3threads" * doctest::timeout(300)) {
  pfs(3);
}

TEST_CASE("pfs.4threads" * doctest::timeout(300)) {
  pfs(4);
}

TEST_CASE("pfs.5threads" * doctest::timeout(300)) {
  pfs(5);
}

TEST_CASE("pfs.6threads" * doctest::timeout(300)) {
  pfs(6);
}

TEST_CASE("pfs.7threads" * doctest::timeout(300)) {
  pfs(7);
}

TEST_CASE("pfs.8threads" * doctest::timeout(300)) {
  pfs(8);
}

TEST_CASE("pfs.9threads" * doctest::timeout(300)) {
  pfs(9);
}

TEST_CASE("pfs.10threads" * doctest::timeout(300)) {
  pfs(10);
}

TEST_CASE("pfs.11threads" * doctest::timeout(300)) {
  pfs(11);
}

TEST_CASE("pfs.12threads" * doctest::timeout(300)) {
  pfs(12);
}



