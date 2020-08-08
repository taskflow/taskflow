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

void parallel_for(unsigned W, int type) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;
  
  std::vector<int> vec(1024);
  for(int n = 0; n <= 150; n++) {

    std::fill_n(vec.begin(), vec.size(), -1);

    int beg = ::rand()%300 - 150;
    int end = beg + n;

    for(int s=1; s<=n+1; s = s + ::rand()%17 + 1) {
      for(int c=0; c<=65; c = (c << 1) + 1) {
        taskflow.clear();
        std::atomic<int> counter {0};

        if(type == 0){
          taskflow.parallel_for_static(beg, end, s, [&](int i){
            counter++;
            vec[i-beg] = i;
          }, c);
        }
        else if(type == 1) {
          taskflow.parallel_for_guided(beg, end, s, [&](int i){
            counter++;
            vec[i-beg] = i;
          }, c);
        }
        else if(type == 2) {
          taskflow.parallel_for_dynamic(beg, end, s, [&](int i){
            counter++;
            vec[i-beg] = i;
          }, c);
        }
        else REQUIRE(false);

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
  }

  for(size_t n = 0; n < 150; n++) {
    
    std::fill_n(vec.begin(), vec.size(), -1);

    for(size_t c=0; c<=65; c = (c << 1) + 1) {
      taskflow.clear();
      std::atomic<int> counter {0};

      if(type == 0) {
        taskflow.parallel_for_static(vec.begin(), vec.begin() + n, [&](int& i){
          counter++;
          i = 1;
        }, c);
      }
      else if(type == 1) {
        taskflow.parallel_for_guided(vec.begin(), vec.begin() + n, [&](int& i){
          counter++;
          i = 1;
        }, c);
      }
      else if(type == 2) {
        taskflow.parallel_for_dynamic(vec.begin(), vec.begin() + n, [&](int& i){
          counter++;
          i = 1;
        }, c);
      }
      else REQUIRE(false);

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
}

// static partition
TEST_CASE("pfs.1thread" * doctest::timeout(300)) {
  parallel_for(1, 0);
}

TEST_CASE("pfs.2threads" * doctest::timeout(300)) {
  parallel_for(2, 0);
}

TEST_CASE("pfs.3threads" * doctest::timeout(300)) {
  parallel_for(3, 0);
}

TEST_CASE("pfs.4threads" * doctest::timeout(300)) {
  parallel_for(4, 0);
}

TEST_CASE("pfs.5threads" * doctest::timeout(300)) {
  parallel_for(5, 0);
}

TEST_CASE("pfs.6threads" * doctest::timeout(300)) {
  parallel_for(6, 0);
}

TEST_CASE("pfs.7threads" * doctest::timeout(300)) {
  parallel_for(7, 0);
}

TEST_CASE("pfs.8threads" * doctest::timeout(300)) {
  parallel_for(8, 0);
}

TEST_CASE("pfs.9threads" * doctest::timeout(300)) {
  parallel_for(9, 0);
}

TEST_CASE("pfs.10threads" * doctest::timeout(300)) {
  parallel_for(10, 0);
}

TEST_CASE("pfs.11threads" * doctest::timeout(300)) {
  parallel_for(11, 0);
}

TEST_CASE("pfs.12threads" * doctest::timeout(300)) {
  parallel_for(12, 0);
}

// guided partition
TEST_CASE("pfg.1thread" * doctest::timeout(300)) {
  parallel_for(1, 1);
}

TEST_CASE("pfg.2threads" * doctest::timeout(300)) {
  parallel_for(2, 1);
}

TEST_CASE("pfg.3threads" * doctest::timeout(300)) {
  parallel_for(3, 1);
}

TEST_CASE("pfg.4threads" * doctest::timeout(300)) {
  parallel_for(4, 1);
}

TEST_CASE("pfg.5threads" * doctest::timeout(300)) {
  parallel_for(5, 1);
}

TEST_CASE("pfg.6threads" * doctest::timeout(300)) {
  parallel_for(6, 1);
}

TEST_CASE("pfg.7threads" * doctest::timeout(300)) {
  parallel_for(7, 1);
}

TEST_CASE("pfg.8threads" * doctest::timeout(300)) {
  parallel_for(8, 1);
}

TEST_CASE("pfg.9threads" * doctest::timeout(300)) {
  parallel_for(9, 1);
}

TEST_CASE("pfg.10threads" * doctest::timeout(300)) {
  parallel_for(10, 1);
}

TEST_CASE("pfg.11threads" * doctest::timeout(300)) {
  parallel_for(11, 1);
}

TEST_CASE("pfg.12threads" * doctest::timeout(300)) {
  parallel_for(12, 1);
}

// dynamic partition
TEST_CASE("pfd.1thread" * doctest::timeout(300)) {
  parallel_for(1, 2);
}

TEST_CASE("pfd.2threads" * doctest::timeout(300)) {
  parallel_for(2, 2);
}

TEST_CASE("pfd.3threads" * doctest::timeout(300)) {
  parallel_for(3, 2);
}

TEST_CASE("pfd.4threads" * doctest::timeout(300)) {
  parallel_for(4, 2);
}

TEST_CASE("pfd.5threads" * doctest::timeout(300)) {
  parallel_for(5, 2);
}

TEST_CASE("pfd.6threads" * doctest::timeout(300)) {
  parallel_for(6, 2);
}

TEST_CASE("pfd.7threads" * doctest::timeout(300)) {
  parallel_for(7, 2);
}

TEST_CASE("pfd.8threads" * doctest::timeout(300)) {
  parallel_for(8, 2);
}

TEST_CASE("pfd.9threads" * doctest::timeout(300)) {
  parallel_for(9, 2);
}

TEST_CASE("pfd.10threads" * doctest::timeout(300)) {
  parallel_for(10, 2);
}

TEST_CASE("pfd.11threads" * doctest::timeout(300)) {
  parallel_for(11, 2);
}

TEST_CASE("pfd.12threads" * doctest::timeout(300)) {
  parallel_for(12, 2);
}
