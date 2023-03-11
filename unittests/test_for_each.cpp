#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

// --------------------------------------------------------
// Testcase: for_each
// --------------------------------------------------------

template <typename P>
void for_each(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::vector<int> vec(1024);
  for(int n = 0; n <= 150; n++) {

    std::fill_n(vec.begin(), vec.size(), -1);

    int beg = ::rand()%300 - 150;
    int end = beg + n;

    for(int s=1; s<=16; s*=2) {
      for(size_t c : {0, 1, 3, 7, 99}) {
        taskflow.clear();
        std::atomic<int> counter {0};

        taskflow.for_each_index(tf::ExecutionPolicy<P>(c),
          beg, end, s, [&](int i){ counter++; vec[i-beg] = i;}
        );

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
    for(size_t c : {0, 1, 3, 7, 99}) {

      std::fill_n(vec.begin(), vec.size(), -1);

      taskflow.clear();
      std::atomic<int> counter {0};

      taskflow.for_each(tf::ExecutionPolicy<P>(c),
        vec.begin(), vec.begin() + n, [&](int& i){
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
}

// guided
TEST_CASE("ParallelFor.Guided.1thread" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner>(1);
}

TEST_CASE("ParallelFor.Guided.2threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner>(2);
}

TEST_CASE("ParallelFor.Guided.3threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner>(3);
}

TEST_CASE("ParallelFor.Guided.4threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner>(4);
}

TEST_CASE("ParallelFor.Guided.5threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner>(5);
}

TEST_CASE("ParallelFor.Guided.6threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner>(6);
}

TEST_CASE("ParallelFor.Guided.7threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner>(7);
}

TEST_CASE("ParallelFor.Guided.8threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner>(8);
}

TEST_CASE("ParallelFor.Guided.9threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner>(9);
}

TEST_CASE("ParallelFor.Guided.10threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner>(10);
}

TEST_CASE("ParallelFor.Guided.11threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner>(11);
}

TEST_CASE("ParallelFor.Guided.12threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner>(12);
}

// dynamic
TEST_CASE("ParallelFor.Dynamic.1thread" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner>(1);
}

TEST_CASE("ParallelFor.Dynamic.2threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner>(2);
}

TEST_CASE("ParallelFor.Dynamic.3threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner>(3);
}

TEST_CASE("ParallelFor.Dynamic.4threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner>(4);
}

TEST_CASE("ParallelFor.Dynamic.5threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner>(5);
}

TEST_CASE("ParallelFor.Dynamic.6threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner>(6);
}

TEST_CASE("ParallelFor.Dynamic.7threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner>(7);
}

TEST_CASE("ParallelFor.Dynamic.8threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner>(8);
}

TEST_CASE("ParallelFor.Dynamic.9threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner>(9);
}

TEST_CASE("ParallelFor.Dynamic.10threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner>(10);
}

TEST_CASE("ParallelFor.Dynamic.11threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner>(11);
}

TEST_CASE("ParallelFor.Dynamic.12threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner>(12);
}

// static
TEST_CASE("ParallelFor.Static.1thread" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner>(1);
}

TEST_CASE("ParallelFor.Static.2threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner>(2);
}

TEST_CASE("ParallelFor.Static.3threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner>(3);
}

TEST_CASE("ParallelFor.Static.4threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner>(4);
}

TEST_CASE("ParallelFor.Static.5threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner>(5);
}

TEST_CASE("ParallelFor.Static.6threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner>(6);
}

TEST_CASE("ParallelFor.Static.7threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner>(7);
}

TEST_CASE("ParallelFor.Static.8threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner>(8);
}

TEST_CASE("ParallelFor.Static.9threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner>(9);
}

TEST_CASE("ParallelFor.Static.10threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner>(10);
}

TEST_CASE("ParallelFor.Static.11threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner>(11);
}

TEST_CASE("ParallelFor.Static.12threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner>(12);
}

// random
TEST_CASE("ParallelFor.Random.1thread" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner>(1);
}

TEST_CASE("ParallelFor.Random.2threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner>(2);
}

TEST_CASE("ParallelFor.Random.3threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner>(3);
}

TEST_CASE("ParallelFor.Random.4threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner>(4);
}

TEST_CASE("ParallelFor.Random.5threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner>(5);
}

TEST_CASE("ParallelFor.Random.6threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner>(6);
}

TEST_CASE("ParallelFor.Random.7threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner>(7);
}

TEST_CASE("ParallelFor.Random.8threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner>(8);
}

TEST_CASE("ParallelFor.Random.9threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner>(9);
}

TEST_CASE("ParallelFor.Random.10threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner>(10);
}

TEST_CASE("ParallelFor.Random.11threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner>(11);
}

TEST_CASE("ParallelFor.Random.12threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner>(12);
}

// ----------------------------------------------------------------------------
// stateful_for_each
// ----------------------------------------------------------------------------

template <typename P>
void stateful_for_each(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;
  std::vector<int> vec;
  std::atomic<int> counter {0};

  for(size_t n = 0; n <= 150; n++) {
    for(size_t c : {0, 1, 3, 7, 99}) {

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

      tf::Task pf1, pf2;

      pf1 = taskflow.for_each(tf::ExecutionPolicy<P>(c),
        std::ref(beg), std::ref(end), [&](int& i){
        counter++;
        i = 8;
      });

      pf2 = taskflow.for_each_index(tf::ExecutionPolicy<P>(c),
        std::ref(ibeg), std::ref(iend), size_t{1}, [&] (size_t i) {
          counter++;
          vec[i] = -8;
      });

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
}

// guided
TEST_CASE("StatefulParallelFor.Guided.1thread" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner>(1);
}

TEST_CASE("StatefulParallelFor.Guided.2threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner>(2);
}

TEST_CASE("StatefulParallelFor.Guided.3threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner>(3);
}

TEST_CASE("StatefulParallelFor.Guided.4threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner>(4);
}

TEST_CASE("StatefulParallelFor.Guided.5threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner>(5);
}

TEST_CASE("StatefulParallelFor.Guided.6threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner>(6);
}

TEST_CASE("StatefulParallelFor.Guided.7threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner>(7);
}

TEST_CASE("StatefulParallelFor.Guided.8threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner>(8);
}

TEST_CASE("StatefulParallelFor.Guided.9threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner>(9);
}

TEST_CASE("StatefulParallelFor.Guided.10threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner>(10);
}

TEST_CASE("StatefulParallelFor.Guided.11threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner>(11);
}

TEST_CASE("StatefulParallelFor.Guided.12threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner>(12);
}

// dynamic
TEST_CASE("StatefulParallelFor.Dynamic.1thread" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner>(1);
}

TEST_CASE("StatefulParallelFor.Dynamic.2threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner>(2);
}

TEST_CASE("StatefulParallelFor.Dynamic.3threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner>(3);
}

TEST_CASE("StatefulParallelFor.Dynamic.4threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner>(4);
}

TEST_CASE("StatefulParallelFor.Dynamic.5threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner>(5);
}

TEST_CASE("StatefulParallelFor.Dynamic.6threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner>(6);
}

TEST_CASE("StatefulParallelFor.Dynamic.7threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner>(7);
}

TEST_CASE("StatefulParallelFor.Dynamic.8threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner>(8);
}

TEST_CASE("StatefulParallelFor.Dynamic.9threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner>(9);
}

TEST_CASE("StatefulParallelFor.Dynamic.10threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner>(10);
}

TEST_CASE("StatefulParallelFor.Dynamic.11threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner>(11);
}

TEST_CASE("StatefulParallelFor.Dynamic.12threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner>(12);
}

// static
TEST_CASE("StatefulParallelFor.Static.1thread" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner>(1);
}

TEST_CASE("StatefulParallelFor.Static.2threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner>(2);
}

TEST_CASE("StatefulParallelFor.Static.3threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner>(3);
}

TEST_CASE("StatefulParallelFor.Static.4threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner>(4);
}

TEST_CASE("StatefulParallelFor.Static.5threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner>(5);
}

TEST_CASE("StatefulParallelFor.Static.6threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner>(6);
}

TEST_CASE("StatefulParallelFor.Static.7threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner>(7);
}

TEST_CASE("StatefulParallelFor.Static.8threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner>(8);
}

TEST_CASE("StatefulParallelFor.Static.9threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner>(9);
}

TEST_CASE("StatefulParallelFor.Static.10threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner>(10);
}

TEST_CASE("StatefulParallelFor.Static.11threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner>(11);
}

TEST_CASE("StatefulParallelFor.Static.12threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner>(12);
}

// random
TEST_CASE("StatefulParallelFor.Random.1thread" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner>(1);
}

TEST_CASE("StatefulParallelFor.Random.2threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner>(2);
}

TEST_CASE("StatefulParallelFor.Random.3threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner>(3);
}

TEST_CASE("StatefulParallelFor.Random.4threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner>(4);
}

TEST_CASE("StatefulParallelFor.Random.5threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner>(5);
}

TEST_CASE("StatefulParallelFor.Random.6threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner>(6);
}

TEST_CASE("StatefulParallelFor.Random.7threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner>(7);
}

TEST_CASE("StatefulParallelFor.Random.8threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner>(8);
}

TEST_CASE("StatefulParallelFor.Random.9threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner>(9);
}

TEST_CASE("StatefulParallelFor.Random.10threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner>(10);
}

TEST_CASE("StatefulParallelFor.Random.11threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner>(11);
}

TEST_CASE("StatefulParallelFor.Random.12threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner>(12);
}
