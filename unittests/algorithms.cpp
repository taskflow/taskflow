#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/reduce.hpp>
#include <taskflow/algorithm/sort.hpp>
#include <taskflow/algorithm/transform.hpp>

// ----------------------------------------------------------------------------
// Data Type
// ----------------------------------------------------------------------------

struct MoveOnly1{

  int a {-1234};
  
  MoveOnly1() = default;

  MoveOnly1(const MoveOnly1&) = delete;
  MoveOnly1(MoveOnly1&&) = default;

  MoveOnly1& operator = (const MoveOnly1& rhs) = delete;
  MoveOnly1& operator = (MoveOnly1&& rhs) = default;

};

struct MoveOnly2{

  int b {-1234};

  MoveOnly2() = default;

  MoveOnly2(const MoveOnly2&) = delete;
  MoveOnly2(MoveOnly2&&) = default;

  MoveOnly2& operator = (const MoveOnly2& rhs) = delete;
  MoveOnly2& operator = (MoveOnly2&& rhs) = default;

};

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

        taskflow.for_each_index(beg, end, s, [&](int i){
          counter++;
          vec[i-beg] = i;
        }, P(c));

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

      taskflow.for_each(vec.begin(), vec.begin() + n, [&](int& i){
        counter++;
        i = 1;
      }, P(c));

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

      pf1 = taskflow.for_each(
        std::ref(beg), std::ref(end), [&](int& i){
        counter++;
        i = 8;
      }, P(c));

      pf2 = taskflow.for_each_index(
        std::ref(ibeg), std::ref(iend), size_t{1}, [&] (size_t i) {
          counter++;
          vec[i] = -8;
      }, P(c));

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

// --------------------------------------------------------
// Testcase: reduce
// --------------------------------------------------------

template <typename P>
void reduce(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::vector<int> vec(1000);

  for(auto& i : vec) i = ::rand() % 100 - 50;

  for(size_t n=1; n<vec.size(); n++) {
    for(size_t c : {0, 1, 3, 7, 99}) {

      int smin = std::numeric_limits<int>::max();
      int pmin = std::numeric_limits<int>::max();
      auto beg = vec.end();
      auto end = vec.end();

      taskflow.clear();
      auto stask = taskflow.emplace([&](){
        beg = vec.begin();
        end = vec.begin() + n;
        for(auto itr = beg; itr != end; itr++) {
          smin = std::min(*itr, smin);
        }
      });

      tf::Task ptask;

      ptask = taskflow.reduce(
        std::ref(beg), std::ref(end), pmin, [](int& l, int& r){
        return std::min(l, r);
      }, P(c));

      stask.precede(ptask);

      executor.run(taskflow).wait();

      REQUIRE(smin != std::numeric_limits<int>::max());
      REQUIRE(pmin != std::numeric_limits<int>::max());
      REQUIRE(smin == pmin);
    }
  }
}

// guided
TEST_CASE("Reduce.Guided.1thread" * doctest::timeout(300)) {
  reduce<tf::GuidedPartitioner>(1);
}

TEST_CASE("Reduce.Guided.2threads" * doctest::timeout(300)) {
  reduce<tf::GuidedPartitioner>(2);
}

TEST_CASE("Reduce.Guided.3threads" * doctest::timeout(300)) {
  reduce<tf::GuidedPartitioner>(3);
}

TEST_CASE("Reduce.Guided.4threads" * doctest::timeout(300)) {
  reduce<tf::GuidedPartitioner>(4);
}

TEST_CASE("Reduce.Guided.5threads" * doctest::timeout(300)) {
  reduce<tf::GuidedPartitioner>(5);
}

TEST_CASE("Reduce.Guided.6threads" * doctest::timeout(300)) {
  reduce<tf::GuidedPartitioner>(6);
}

TEST_CASE("Reduce.Guided.7threads" * doctest::timeout(300)) {
  reduce<tf::GuidedPartitioner>(7);
}

TEST_CASE("Reduce.Guided.8threads" * doctest::timeout(300)) {
  reduce<tf::GuidedPartitioner>(8);
}

TEST_CASE("Reduce.Guided.9threads" * doctest::timeout(300)) {
  reduce<tf::GuidedPartitioner>(9);
}

TEST_CASE("Reduce.Guided.10threads" * doctest::timeout(300)) {
  reduce<tf::GuidedPartitioner>(10);
}

TEST_CASE("Reduce.Guided.11threads" * doctest::timeout(300)) {
  reduce<tf::GuidedPartitioner>(11);
}

TEST_CASE("Reduce.Guided.12threads" * doctest::timeout(300)) {
  reduce<tf::GuidedPartitioner>(12);
}

// dynamic
TEST_CASE("Reduce.Dynamic.1thread" * doctest::timeout(300)) {
  reduce<tf::DynamicPartitioner>(1);
}

TEST_CASE("Reduce.Dynamic.2threads" * doctest::timeout(300)) {
  reduce<tf::DynamicPartitioner>(2);
}

TEST_CASE("Reduce.Dynamic.3threads" * doctest::timeout(300)) {
  reduce<tf::DynamicPartitioner>(3);
}

TEST_CASE("Reduce.Dynamic.4threads" * doctest::timeout(300)) {
  reduce<tf::DynamicPartitioner>(4);
}

TEST_CASE("Reduce.Dynamic.5threads" * doctest::timeout(300)) {
  reduce<tf::DynamicPartitioner>(5);
}

TEST_CASE("Reduce.Dynamic.6threads" * doctest::timeout(300)) {
  reduce<tf::DynamicPartitioner>(6);
}

TEST_CASE("Reduce.Dynamic.7threads" * doctest::timeout(300)) {
  reduce<tf::DynamicPartitioner>(7);
}

TEST_CASE("Reduce.Dynamic.8threads" * doctest::timeout(300)) {
  reduce<tf::DynamicPartitioner>(8);
}

TEST_CASE("Reduce.Dynamic.9threads" * doctest::timeout(300)) {
  reduce<tf::DynamicPartitioner>(9);
}

TEST_CASE("Reduce.Dynamic.10threads" * doctest::timeout(300)) {
  reduce<tf::DynamicPartitioner>(10);
}

TEST_CASE("Reduce.Dynamic.11threads" * doctest::timeout(300)) {
  reduce<tf::DynamicPartitioner>(11);
}

TEST_CASE("Reduce.Dynamic.12threads" * doctest::timeout(300)) {
  reduce<tf::DynamicPartitioner>(12);
}

// static
TEST_CASE("Reduce.Static.1thread" * doctest::timeout(300)) {
  reduce<tf::StaticPartitioner>(1);
}

TEST_CASE("Reduce.Static.2threads" * doctest::timeout(300)) {
  reduce<tf::StaticPartitioner>(2);
}

TEST_CASE("Reduce.Static.3threads" * doctest::timeout(300)) {
  reduce<tf::StaticPartitioner>(3);
}

TEST_CASE("Reduce.Static.4threads" * doctest::timeout(300)) {
  reduce<tf::StaticPartitioner>(4);
}

TEST_CASE("Reduce.Static.5threads" * doctest::timeout(300)) {
  reduce<tf::StaticPartitioner>(5);
}

TEST_CASE("Reduce.Static.6threads" * doctest::timeout(300)) {
  reduce<tf::StaticPartitioner>(6);
}

TEST_CASE("Reduce.Static.7threads" * doctest::timeout(300)) {
  reduce<tf::StaticPartitioner>(7);
}

TEST_CASE("Reduce.Static.8threads" * doctest::timeout(300)) {
  reduce<tf::StaticPartitioner>(8);
}

TEST_CASE("Reduce.Static.9threads" * doctest::timeout(300)) {
  reduce<tf::StaticPartitioner>(9);
}

TEST_CASE("Reduce.Static.10threads" * doctest::timeout(300)) {
  reduce<tf::StaticPartitioner>(10);
}

TEST_CASE("Reduce.Static.11threads" * doctest::timeout(300)) {
  reduce<tf::StaticPartitioner>(11);
}

TEST_CASE("Reduce.Static.12threads" * doctest::timeout(300)) {
  reduce<tf::StaticPartitioner>(12);
}

// ----------------------------------------------------------------------------
// transform_reduce
// ----------------------------------------------------------------------------

class Data {

  private:

    int _v {::rand() % 100 - 50};

  public:

    int get() const { return _v; }
};

template <typename P>
void transform_reduce(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::vector<Data> vec(1000);

  for(size_t n=1; n<vec.size(); n++) {
    for(size_t c : {0, 1, 3, 7, 99}) {

      int smin = std::numeric_limits<int>::max();
      int pmin = std::numeric_limits<int>::max();
      auto beg = vec.end();
      auto end = vec.end();

      taskflow.clear();
      auto stask = taskflow.emplace([&](){
        beg = vec.begin();
        end = vec.begin() + n;
        for(auto itr = beg; itr != end; itr++) {
          smin = std::min(itr->get(), smin);
        }
      });

      tf::Task ptask;

      ptask = taskflow.transform_reduce(
        std::ref(beg), std::ref(end), pmin,
        [] (int l, int r)   { return std::min(l, r); },
        [] (const Data& data) { return data.get(); },
        P(c)
      );

      stask.precede(ptask);

      executor.run(taskflow).wait();

      REQUIRE(smin != std::numeric_limits<int>::max());
      REQUIRE(pmin != std::numeric_limits<int>::max());
      REQUIRE(smin == pmin);
    }
  }
}

// guided
TEST_CASE("TransformReduce.Guided.1thread" * doctest::timeout(300)) {
  transform_reduce<tf::GuidedPartitioner>(1);
}

TEST_CASE("TransformReduce.Guided.2threads" * doctest::timeout(300)) {
  transform_reduce<tf::GuidedPartitioner>(2);
}

TEST_CASE("TransformReduce.Guided.3threads" * doctest::timeout(300)) {
  transform_reduce<tf::GuidedPartitioner>(3);
}

TEST_CASE("TransformReduce.Guided.4threads" * doctest::timeout(300)) {
  transform_reduce<tf::GuidedPartitioner>(4);
}

TEST_CASE("TransformReduce.Guided.5threads" * doctest::timeout(300)) {
  transform_reduce<tf::GuidedPartitioner>(5);
}

TEST_CASE("TransformReduce.Guided.6threads" * doctest::timeout(300)) {
  transform_reduce<tf::GuidedPartitioner>(6);
}

TEST_CASE("TransformReduce.Guided.7threads" * doctest::timeout(300)) {
  transform_reduce<tf::GuidedPartitioner>(7);
}

TEST_CASE("TransformReduce.Guided.8threads" * doctest::timeout(300)) {
  transform_reduce<tf::GuidedPartitioner>(8);
}

TEST_CASE("TransformReduce.Guided.9threads" * doctest::timeout(300)) {
  transform_reduce<tf::GuidedPartitioner>(9);
}

TEST_CASE("TransformReduce.Guided.10threads" * doctest::timeout(300)) {
  transform_reduce<tf::GuidedPartitioner>(10);
}

TEST_CASE("TransformReduce.Guided.11threads" * doctest::timeout(300)) {
  transform_reduce<tf::GuidedPartitioner>(11);
}

TEST_CASE("TransformReduce.Guided.12threads" * doctest::timeout(300)) {
  transform_reduce<tf::GuidedPartitioner>(12);
}

// dynamic
TEST_CASE("TransformReduce.Dynamic.1thread" * doctest::timeout(300)) {
  transform_reduce<tf::DynamicPartitioner>(1);
}

TEST_CASE("TransformReduce.Dynamic.2threads" * doctest::timeout(300)) {
  transform_reduce<tf::DynamicPartitioner>(2);
}

TEST_CASE("TransformReduce.Dynamic.3threads" * doctest::timeout(300)) {
  transform_reduce<tf::DynamicPartitioner>(3);
}

TEST_CASE("TransformReduce.Dynamic.4threads" * doctest::timeout(300)) {
  transform_reduce<tf::DynamicPartitioner>(4);
}

TEST_CASE("TransformReduce.Dynamic.5threads" * doctest::timeout(300)) {
  transform_reduce<tf::DynamicPartitioner>(5);
}

TEST_CASE("TransformReduce.Dynamic.6threads" * doctest::timeout(300)) {
  transform_reduce<tf::DynamicPartitioner>(6);
}

TEST_CASE("TransformReduce.Dynamic.7threads" * doctest::timeout(300)) {
  transform_reduce<tf::DynamicPartitioner>(7);
}

TEST_CASE("TransformReduce.Dynamic.8threads" * doctest::timeout(300)) {
  transform_reduce<tf::DynamicPartitioner>(8);
}

TEST_CASE("TransformReduce.Dynamic.9threads" * doctest::timeout(300)) {
  transform_reduce<tf::DynamicPartitioner>(9);
}

TEST_CASE("TransformReduce.Dynamic.10threads" * doctest::timeout(300)) {
  transform_reduce<tf::DynamicPartitioner>(10);
}

TEST_CASE("TransformReduce.Dynamic.11threads" * doctest::timeout(300)) {
  transform_reduce<tf::DynamicPartitioner>(11);
}

TEST_CASE("TransformReduce.Dynamic.12threads" * doctest::timeout(300)) {
  transform_reduce<tf::DynamicPartitioner>(12);
}

// static
TEST_CASE("TransformReduce.Static.1thread" * doctest::timeout(300)) {
  transform_reduce<tf::StaticPartitioner>(1);
}

TEST_CASE("TransformReduce.Static.2threads" * doctest::timeout(300)) {
  transform_reduce<tf::StaticPartitioner>(2);
}

TEST_CASE("TransformReduce.Static.3threads" * doctest::timeout(300)) {
  transform_reduce<tf::StaticPartitioner>(3);
}

TEST_CASE("TransformReduce.Static.4threads" * doctest::timeout(300)) {
  transform_reduce<tf::StaticPartitioner>(4);
}

TEST_CASE("TransformReduce.Static.5threads" * doctest::timeout(300)) {
  transform_reduce<tf::StaticPartitioner>(5);
}

TEST_CASE("TransformReduce.Static.6threads" * doctest::timeout(300)) {
  transform_reduce<tf::StaticPartitioner>(6);
}

TEST_CASE("TransformReduce.Static.7threads" * doctest::timeout(300)) {
  transform_reduce<tf::StaticPartitioner>(7);
}

TEST_CASE("TransformReduce.Static.8threads" * doctest::timeout(300)) {
  transform_reduce<tf::StaticPartitioner>(8);
}

TEST_CASE("TransformReduce.Static.9threads" * doctest::timeout(300)) {
  transform_reduce<tf::StaticPartitioner>(9);
}

TEST_CASE("TransformReduce.Static.10threads" * doctest::timeout(300)) {
  transform_reduce<tf::StaticPartitioner>(10);
}

TEST_CASE("TransformReduce.Static.11threads" * doctest::timeout(300)) {
  transform_reduce<tf::StaticPartitioner>(11);
}

TEST_CASE("TransformReduce.Static.12threads" * doctest::timeout(300)) {
  transform_reduce<tf::StaticPartitioner>(12);
}

// ----------------------------------------------------------------------------
// Transform & Reduce on Movable Data
// ----------------------------------------------------------------------------

void move_only_transform_reduce(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  const size_t N = 100000;
  std::vector<MoveOnly1> vec(N);

  for(auto& i : vec) i.a = 1;

  MoveOnly2 res;
  res.b = 100;

  taskflow.transform_reduce(vec.begin(), vec.end(), res,
    [](MoveOnly2 m1, MoveOnly2 m2) {
      MoveOnly2 res;
      res.b = m1.b + m2.b;
      return res;
    },
    [](const MoveOnly1& m) {
      MoveOnly2 n;
      n.b = m.a;
      return n;
    }
  );

  executor.run(taskflow).wait();

  REQUIRE(res.b == N + 100);
  
  // change vec data
  taskflow.clear();
  res.b = 0;
  
  taskflow.transform_reduce(vec.begin(), vec.end(), res,
    [](MoveOnly2&& m1, MoveOnly2&& m2) {
      MoveOnly2 res;
      res.b = m1.b + m2.b;
      return res;
    },
    [](MoveOnly1& m) {
      MoveOnly2 n;
      n.b = m.a;
      m.a = -7;
      return n;
    }
  );

  executor.run(taskflow).wait();
  REQUIRE(res.b == N);

  for(const auto& i : vec) {
    REQUIRE(i.a == -7);
  }

  // reduce
  taskflow.clear();
  MoveOnly1 red;
  red.a = 0;

  taskflow.reduce(vec.begin(), vec.end(), red,
    [](MoveOnly1& m1, MoveOnly1& m2){
      MoveOnly1 res;
      res.a = m1.a + m2.a;
      return res;
    }
  );  

  executor.run(taskflow).wait();
  REQUIRE(red.a == -7*N);
  
  taskflow.clear();
  red.a = 0;

  taskflow.reduce(vec.begin(), vec.end(), red,
    [](const MoveOnly1& m1, const MoveOnly1& m2){
      MoveOnly1 res;
      res.a = m1.a + m2.a;
      return res;
    }
  );  

  executor.run(taskflow).wait();
  REQUIRE(red.a == -7*N);

}

TEST_CASE("TransformReduce.MoveOnlyData.1thread" * doctest::timeout(300)) {
  move_only_transform_reduce(1);
}

TEST_CASE("TransformReduce.MoveOnlyData.2threads" * doctest::timeout(300)) {
  move_only_transform_reduce(2);
}

TEST_CASE("TransformReduce.MoveOnlyData.3threads" * doctest::timeout(300)) {
  move_only_transform_reduce(3);
}

TEST_CASE("TransformReduce.MoveOnlyData.4threads" * doctest::timeout(300)) {
  move_only_transform_reduce(4);
}

// ----------------------------------------------------------------------------
// parallel sort
// ----------------------------------------------------------------------------

template <typename T>
void ps_pod(size_t W, size_t N) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  std::vector<T> data(N);

  for(auto& d : data) {
    d = ::rand() % 1000 - 500;
  }

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  taskflow.sort(data.begin(), data.end());

  executor.run(taskflow).wait();

  REQUIRE(std::is_sorted(data.begin(), data.end()));
}

TEST_CASE("ParallelSort.int.1.100000") {
  ps_pod<int>(1, 100000);
}

TEST_CASE("ParallelSort.int.2.100000") {
  ps_pod<int>(2, 100000);
}

TEST_CASE("ParallelSort.int.3.100000") {
  ps_pod<int>(3, 100000);
}

TEST_CASE("ParallelSort.int.4.100000") {
  ps_pod<int>(4, 100000);
}

TEST_CASE("ParallelSort.ldouble.1.100000") {
  ps_pod<long double>(1, 100000);
}

TEST_CASE("ParallelSort.ldouble.2.100000") {
  ps_pod<long double>(2, 100000);
}

TEST_CASE("ParallelSort.ldouble.3.100000") {
  ps_pod<long double>(3, 100000);
}

TEST_CASE("ParallelSort.ldouble.4.100000") {
  ps_pod<long double>(4, 100000);
}

struct Object {

  std::array<int, 10> integers;

  int sum() const {
    int s = 0;
    for(const auto i : integers) {
      s += i;
    }
    return s;
  }
};

void ps_object(size_t W, size_t N) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  std::vector<Object> data(N);

  for(auto& d : data) {
    for(auto& i : d.integers) {
      i = ::rand();
    }
  }

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  taskflow.sort(data.begin(), data.end(), [](const auto& l, const auto& r){
    return l.sum() < r.sum();
  });

  executor.run(taskflow).wait();

  REQUIRE(std::is_sorted(data.begin(), data.end(),
    [](const auto& l, const auto& r){ return l.sum() < r.sum(); }
  ));
}

TEST_CASE("ParallelSort.object.1.100000") {
  ps_object(1, 100000);
}

TEST_CASE("ParallelSort.object.2.100000") {
  ps_object(2, 100000);
}

TEST_CASE("ParallelSort.object.3.100000") {
  ps_object(3, 100000);
}

TEST_CASE("ParallelSort.object.4.100000") {
  ps_object(4, 100000);
}

void move_only_ps(unsigned W) {
  
  std::vector<MoveOnly1> vec(1000000);
  for(auto& i : vec) {
    i.a = rand()%100;
  }

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  taskflow.sort(vec.begin(), vec.end(),
    [](const MoveOnly1& m1, const MoveOnly1&m2) {
      return m1.a < m2.a;
    }
  );

  executor.run(taskflow).wait();

  for(size_t i=1; i<vec.size(); i++) {
    REQUIRE(vec[i-1].a <= vec[i].a);
  }

}

TEST_CASE("ParallelSort.MoveOnlyObject.1thread") {
  move_only_ps(1);
}

TEST_CASE("ParallelSort.MoveOnlyObject.2threads") {
  move_only_ps(2);
}

TEST_CASE("ParallelSort.MoveOnlyObject.3threads") {
  move_only_ps(3);
}

TEST_CASE("ParallelSort.MoveOnlyObject.4threads") {
  move_only_ps(4);
}

// ----------------------------------------------------------------------------
// Parallel Transform 1
// ----------------------------------------------------------------------------

template<typename T, typename P>
void parallel_transform(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  for(size_t N=0; N<1000; N=(N+1)*2) {
    for(size_t c : {0, 1, 3, 7, 99}) {

      taskflow.clear();

      typename T::const_iterator src_beg;
      typename T::const_iterator src_end;
      std::list<std::string>::iterator tgt_beg;

      T src;
      std::list<std::string> tgt;

      taskflow.clear();

      auto from = taskflow.emplace([&](){
        src.resize(N);
        for(auto& d : src) {
          d = ::rand() % 10;
          tgt.emplace_back("hi");
        }
        src_beg = src.begin();
        src_end = src.end();
        tgt_beg = tgt.begin();
      });

      auto to = taskflow.transform(
        std::ref(src_beg), std::ref(src_end), std::ref(tgt_beg),
        [] (const auto& in) {
          return std::to_string(in+10);
        },
        P(c)
      );

      from.precede(to);

      executor.run(taskflow).wait();

      auto s_itr = src.begin();
      auto d_itr = tgt.begin();
      while(s_itr != src.end()) {
        REQUIRE(*d_itr++ == std::to_string(*s_itr++ + 10));
      }
    }
  }
}

// guided
TEST_CASE("ParallelTransform.Guided.1thread") {
  parallel_transform<std::vector<int>, tf::GuidedPartitioner>(1);
  parallel_transform<std::list<int>, tf::GuidedPartitioner>(1);
}

TEST_CASE("ParallelTransform.Guided.2threads") {
  parallel_transform<std::vector<int>, tf::GuidedPartitioner>(2);
  parallel_transform<std::list<int>, tf::GuidedPartitioner>(2);
}

TEST_CASE("ParallelTransform.Guided.3threads") {
  parallel_transform<std::vector<int>, tf::GuidedPartitioner>(3);
  parallel_transform<std::list<int>, tf::GuidedPartitioner>(3);
}

TEST_CASE("ParallelTransform.Guided.4threads") {
  parallel_transform<std::vector<int>, tf::GuidedPartitioner>(4);
  parallel_transform<std::list<int>, tf::GuidedPartitioner>(4);
}

// dynamic
TEST_CASE("ParallelTransform.Dynamic.1thread") {
  parallel_transform<std::vector<int>, tf::DynamicPartitioner>(1);
  parallel_transform<std::list<int>, tf::DynamicPartitioner>(1);
}

TEST_CASE("ParallelTransform.Dynamic.2threads") {
  parallel_transform<std::vector<int>, tf::DynamicPartitioner>(2);
  parallel_transform<std::list<int>, tf::DynamicPartitioner>(2);
}

TEST_CASE("ParallelTransform.Dynamic.3threads") {
  parallel_transform<std::vector<int>, tf::DynamicPartitioner>(3);
  parallel_transform<std::list<int>, tf::DynamicPartitioner>(3);
}

TEST_CASE("ParallelTransform.Dynamic.4threads") {
  parallel_transform<std::vector<int>, tf::DynamicPartitioner>(4);
  parallel_transform<std::list<int>, tf::DynamicPartitioner>(4);
}

// static
TEST_CASE("ParallelTransform.Static.1thread") {
  parallel_transform<std::vector<int>, tf::StaticPartitioner>(1);
  parallel_transform<std::list<int>, tf::StaticPartitioner>(1);
}

TEST_CASE("ParallelTransform.Static.2threads") {
  parallel_transform<std::vector<int>, tf::StaticPartitioner>(2);
  parallel_transform<std::list<int>, tf::StaticPartitioner>(2);
}

TEST_CASE("ParallelTransform.Static.3threads") {
  parallel_transform<std::vector<int>, tf::StaticPartitioner>(3);
  parallel_transform<std::list<int>, tf::StaticPartitioner>(3);
}

TEST_CASE("ParallelTransform.Static.4threads") {
  parallel_transform<std::vector<int>, tf::StaticPartitioner>(4);
  parallel_transform<std::list<int>, tf::StaticPartitioner>(4);
}

// ----------------------------------------------------------------------------
// Parallel Transform 2
// ----------------------------------------------------------------------------

template<typename T, typename P>
void parallel_transform2(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  for(size_t N=0; N<1000; N=(N+1)*2) {
    for(size_t c : {0, 1, 3, 7, 99}) {

      taskflow.clear();

      typename T::const_iterator src_beg;
      typename T::const_iterator src_end;
      std::list<std::string>::iterator tgt_beg;

      T src;
      std::list<std::string> tgt;

      taskflow.clear();

      auto from = taskflow.emplace([&](){
        src.resize(N);
        for(auto& d : src) {
          d = ::rand() % 10;
          tgt.emplace_back("hi");
        }
        src_beg = src.begin();
        src_end = src.end();
        tgt_beg = tgt.begin();
      });

      auto to = taskflow.transform(
        std::ref(src_beg), std::ref(src_end), std::ref(src_beg), std::ref(tgt_beg),
        [] (const auto& in1, const auto& in2) {
          return std::to_string(in1 + in2 + 10);
        },
        P(c)
      );

      from.precede(to);

      executor.run(taskflow).wait();

      auto s_itr = src.begin();
      auto d_itr = tgt.begin();
      while(s_itr != src.end()) {
        REQUIRE(*d_itr++ == std::to_string(2 * *s_itr++ + 10));
      }
    }
  }
}

// guided
TEST_CASE("ParallelTransform2.Guided.1thread") {
  parallel_transform2<std::vector<int>, tf::GuidedPartitioner>(1);
  parallel_transform2<std::list<int>, tf::GuidedPartitioner>(1);
}

TEST_CASE("ParallelTransform2.Guided.2threads") {
  parallel_transform2<std::vector<int>, tf::GuidedPartitioner>(2);
  parallel_transform2<std::list<int>, tf::GuidedPartitioner>(2);
}

TEST_CASE("ParallelTransform2.Guided.3threads") {
  parallel_transform2<std::vector<int>, tf::GuidedPartitioner>(3);
  parallel_transform2<std::list<int>, tf::GuidedPartitioner>(3);
}

TEST_CASE("ParallelTransform2.Guided.4threads") {
  parallel_transform2<std::vector<int>, tf::GuidedPartitioner>(4);
  parallel_transform2<std::list<int>, tf::GuidedPartitioner>(4);
}

// dynamic
TEST_CASE("ParallelTransform2.Dynamic.1thread") {
  parallel_transform2<std::vector<int>, tf::DynamicPartitioner>(1);
  parallel_transform2<std::list<int>, tf::DynamicPartitioner>(1);
}

TEST_CASE("ParallelTransform2.Dynamic.2threads") {
  parallel_transform2<std::vector<int>, tf::DynamicPartitioner>(2);
  parallel_transform2<std::list<int>, tf::DynamicPartitioner>(2);
}

TEST_CASE("ParallelTransform2.Dynamic.3threads") {
  parallel_transform2<std::vector<int>, tf::DynamicPartitioner>(3);
  parallel_transform2<std::list<int>, tf::DynamicPartitioner>(3);
}

TEST_CASE("ParallelTransform2.Dynamic.4threads") {
  parallel_transform2<std::vector<int>, tf::DynamicPartitioner>(4);
  parallel_transform2<std::list<int>, tf::DynamicPartitioner>(4);
}

// static
TEST_CASE("ParallelTransform2.Static.1thread") {
  parallel_transform2<std::vector<int>, tf::StaticPartitioner>(1);
  parallel_transform2<std::list<int>, tf::StaticPartitioner>(1);
}

TEST_CASE("ParallelTransform2.Static.2threads") {
  parallel_transform2<std::vector<int>, tf::StaticPartitioner>(2);
  parallel_transform2<std::list<int>, tf::StaticPartitioner>(2);
}

TEST_CASE("ParallelTransform2.Static.3threads") {
  parallel_transform2<std::vector<int>, tf::StaticPartitioner>(3);
  parallel_transform2<std::list<int>, tf::StaticPartitioner>(3);
}

TEST_CASE("ParallelTransform2.Static.4threads") {
  parallel_transform2<std::vector<int>, tf::StaticPartitioner>(4);
  parallel_transform2<std::list<int>, tf::StaticPartitioner>(4);
}


// ----------------------------------------------------------------------------
// Parallel Transform 3
// ----------------------------------------------------------------------------

template <typename P>
void parallel_transform3(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  using std::string;
  using std::size_t;

  for(size_t N=0; N<1000; N=(N+1)*2) {

    std::multimap<int, size_t> src;

    /** Reference implementation with std::transform */
    std::vector<string> ref;

    /** Target implementation with Subflow::transform */
    std::vector<string> tgt;

    std::vector<string>::iterator tgt_beg;

    /** A generic function to cast integers to string */
    const auto myFunction = [](const size_t x) -> string {
      return "id_" + std::to_string(x);
    };

    taskflow.clear();

    /** Group integers 0..(N-1) into ten groups,
     * each having an unique key `d`.
     */
    auto from = taskflow.emplace([&, N](){
      for(size_t i = 0; i < N; i++) {
        const int d = ::rand() % 10;
        src.emplace(d, i);
      }

      ref.resize(N);

      tgt.resize(N);
      tgt_beg = tgt.begin();
    });

    auto to_ref = taskflow.emplace([&]() {

      // Find entries matching key = 0.
      // This can return empty results.
      const auto [src_beg, src_end] = src.equal_range(0);
      const size_t n_matching = std::distance(src_beg, src_end);
      ref.resize(n_matching);

      // Extract all values having matching key value.
      std::transform(src_beg, src_end, ref.begin(),
        [&](const auto& x) -> string {
          return myFunction(x.second);
        }
      );
    });

    /** Dynamic scheduling with Subflow::transform */
    auto to_tgt = taskflow.emplace([&](tf::Subflow& subflow) {

      // Find entries matching key = 0
      const auto [src_beg, src_end] = src.equal_range(0);
      const size_t n_matching = std::distance(src_beg, src_end);
      tgt.resize(n_matching);

      subflow.transform(std::ref(src_beg), std::ref(src_end), std::ref(tgt_beg),
        [&] (const auto& x) -> string {
          return myFunction(x.second);
      }, P(1));

      subflow.join();
    });

    from.precede(to_ref);
    from.precede(to_tgt);

    executor.run(taskflow).wait();

    /** Target entries much match. */
    REQUIRE(std::equal(tgt.begin(), tgt.end(), ref.begin()));
  }
}

// guided
TEST_CASE("ParallelTransform3.Guided.1thread") {
  parallel_transform3<tf::GuidedPartitioner>(1);
  parallel_transform3<tf::GuidedPartitioner>(1);
}

TEST_CASE("ParallelTransform3.Guided.2threads") {
  parallel_transform3<tf::GuidedPartitioner>(2);
  parallel_transform3<tf::GuidedPartitioner>(2);
}

TEST_CASE("ParallelTransform3.Guided.3threads") {
  parallel_transform3<tf::GuidedPartitioner>(3);
  parallel_transform3<tf::GuidedPartitioner>(3);
}

TEST_CASE("ParallelTransform3.Guided.4threads") {
  parallel_transform3<tf::GuidedPartitioner>(4);
  parallel_transform3<tf::GuidedPartitioner>(4);
}

// dynamic
TEST_CASE("ParallelTransform3.Dynamic.1thread") {
  parallel_transform3<tf::DynamicPartitioner>(1);
  parallel_transform3<tf::DynamicPartitioner>(1);
}

TEST_CASE("ParallelTransform3.Dynamic.2threads") {
  parallel_transform3<tf::DynamicPartitioner>(2);
  parallel_transform3<tf::DynamicPartitioner>(2);
}

TEST_CASE("ParallelTransform3.Dynamic.3threads") {
  parallel_transform3<tf::DynamicPartitioner>(3);
  parallel_transform3<tf::DynamicPartitioner>(3);
}

TEST_CASE("ParallelTransform3.Dynamic.4threads") {
  parallel_transform3<tf::DynamicPartitioner>(4);
  parallel_transform3<tf::DynamicPartitioner>(4);
}

// static
TEST_CASE("ParallelTransform3.Static.1thread") {
  parallel_transform3<tf::StaticPartitioner>(1);
  parallel_transform3<tf::StaticPartitioner>(1);
}

TEST_CASE("ParallelTransform3.Static.2threads") {
  parallel_transform3<tf::StaticPartitioner>(2);
  parallel_transform3<tf::StaticPartitioner>(2);
}

TEST_CASE("ParallelTransform3.Static.3threads") {
  parallel_transform3<tf::StaticPartitioner>(3);
  parallel_transform3<tf::StaticPartitioner>(3);
}

TEST_CASE("ParallelTransform3.Static.4threads") {
  parallel_transform3<tf::StaticPartitioner>(4);
  parallel_transform3<tf::StaticPartitioner>(4);
}
