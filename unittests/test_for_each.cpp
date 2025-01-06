#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <cstdint>

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

        taskflow.for_each_index(
          beg, end, s, [&](int i){ counter++; vec[i-beg] = i;}, P(c)
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

      taskflow.for_each(
        vec.begin(), vec.begin() + n, [&](int& i){
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
  for_each<tf::GuidedPartitioner<>>(1);
}

TEST_CASE("ParallelFor.Guided.2threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner<>>(2);
}

TEST_CASE("ParallelFor.Guided.3threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner<>>(3);
}

TEST_CASE("ParallelFor.Guided.4threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner<>>(4);
}

TEST_CASE("ParallelFor.Guided.5threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner<>>(5);
}

TEST_CASE("ParallelFor.Guided.6threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner<>>(6);
}

TEST_CASE("ParallelFor.Guided.7threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner<>>(7);
}

TEST_CASE("ParallelFor.Guided.8threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner<>>(8);
}

TEST_CASE("ParallelFor.Guided.9threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner<>>(9);
}

TEST_CASE("ParallelFor.Guided.10threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner<>>(10);
}

TEST_CASE("ParallelFor.Guided.11threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner<>>(11);
}

TEST_CASE("ParallelFor.Guided.12threads" * doctest::timeout(300)) {
  for_each<tf::GuidedPartitioner<>>(12);
}

// dynamic
TEST_CASE("ParallelFor.Dynamic.1thread" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner<>>(1);
}

TEST_CASE("ParallelFor.Dynamic.2threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner<>>(2);
}

TEST_CASE("ParallelFor.Dynamic.3threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner<>>(3);
}

TEST_CASE("ParallelFor.Dynamic.4threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner<>>(4);
}

TEST_CASE("ParallelFor.Dynamic.5threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner<>>(5);
}

TEST_CASE("ParallelFor.Dynamic.6threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner<>>(6);
}

TEST_CASE("ParallelFor.Dynamic.7threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner<>>(7);
}

TEST_CASE("ParallelFor.Dynamic.8threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner<>>(8);
}

TEST_CASE("ParallelFor.Dynamic.9threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner<>>(9);
}

TEST_CASE("ParallelFor.Dynamic.10threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner<>>(10);
}

TEST_CASE("ParallelFor.Dynamic.11threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner<>>(11);
}

TEST_CASE("ParallelFor.Dynamic.12threads" * doctest::timeout(300)) {
  for_each<tf::DynamicPartitioner<>>(12);
}

// static
TEST_CASE("ParallelFor.Static.1thread" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner<>>(1);
}

TEST_CASE("ParallelFor.Static.2threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner<>>(2);
}

TEST_CASE("ParallelFor.Static.3threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner<>>(3);
}

TEST_CASE("ParallelFor.Static.4threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner<>>(4);
}

TEST_CASE("ParallelFor.Static.5threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner<>>(5);
}

TEST_CASE("ParallelFor.Static.6threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner<>>(6);
}

TEST_CASE("ParallelFor.Static.7threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner<>>(7);
}

TEST_CASE("ParallelFor.Static.8threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner<>>(8);
}

TEST_CASE("ParallelFor.Static.9threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner<>>(9);
}

TEST_CASE("ParallelFor.Static.10threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner<>>(10);
}

TEST_CASE("ParallelFor.Static.11threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner<>>(11);
}

TEST_CASE("ParallelFor.Static.12threads" * doctest::timeout(300)) {
  for_each<tf::StaticPartitioner<>>(12);
}

// random
TEST_CASE("ParallelFor.Random.1thread" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner<>>(1);
}

TEST_CASE("ParallelFor.Random.2threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner<>>(2);
}

TEST_CASE("ParallelFor.Random.3threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner<>>(3);
}

TEST_CASE("ParallelFor.Random.4threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner<>>(4);
}

TEST_CASE("ParallelFor.Random.5threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner<>>(5);
}

TEST_CASE("ParallelFor.Random.6threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner<>>(6);
}

TEST_CASE("ParallelFor.Random.7threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner<>>(7);
}

TEST_CASE("ParallelFor.Random.8threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner<>>(8);
}

TEST_CASE("ParallelFor.Random.9threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner<>>(9);
}

TEST_CASE("ParallelFor.Random.10threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner<>>(10);
}

TEST_CASE("ParallelFor.Random.11threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner<>>(11);
}

TEST_CASE("ParallelFor.Random.12threads" * doctest::timeout(300)) {
  for_each<tf::RandomPartitioner<>>(12);
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
  stateful_for_each<tf::GuidedPartitioner<>>(1);
}

TEST_CASE("StatefulParallelFor.Guided.2threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner<>>(2);
}

TEST_CASE("StatefulParallelFor.Guided.3threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner<>>(3);
}

TEST_CASE("StatefulParallelFor.Guided.4threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner<>>(4);
}

TEST_CASE("StatefulParallelFor.Guided.5threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner<>>(5);
}

TEST_CASE("StatefulParallelFor.Guided.6threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner<>>(6);
}

TEST_CASE("StatefulParallelFor.Guided.7threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner<>>(7);
}

TEST_CASE("StatefulParallelFor.Guided.8threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner<>>(8);
}

TEST_CASE("StatefulParallelFor.Guided.9threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner<>>(9);
}

TEST_CASE("StatefulParallelFor.Guided.10threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner<>>(10);
}

TEST_CASE("StatefulParallelFor.Guided.11threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner<>>(11);
}

TEST_CASE("StatefulParallelFor.Guided.12threads" * doctest::timeout(300)) {
  stateful_for_each<tf::GuidedPartitioner<>>(12);
}

// dynamic
TEST_CASE("StatefulParallelFor.Dynamic.1thread" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner<>>(1);
}

TEST_CASE("StatefulParallelFor.Dynamic.2threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner<>>(2);
}

TEST_CASE("StatefulParallelFor.Dynamic.3threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner<>>(3);
}

TEST_CASE("StatefulParallelFor.Dynamic.4threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner<>>(4);
}

TEST_CASE("StatefulParallelFor.Dynamic.5threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner<>>(5);
}

TEST_CASE("StatefulParallelFor.Dynamic.6threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner<>>(6);
}

TEST_CASE("StatefulParallelFor.Dynamic.7threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner<>>(7);
}

TEST_CASE("StatefulParallelFor.Dynamic.8threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner<>>(8);
}

TEST_CASE("StatefulParallelFor.Dynamic.9threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner<>>(9);
}

TEST_CASE("StatefulParallelFor.Dynamic.10threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner<>>(10);
}

TEST_CASE("StatefulParallelFor.Dynamic.11threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner<>>(11);
}

TEST_CASE("StatefulParallelFor.Dynamic.12threads" * doctest::timeout(300)) {
  stateful_for_each<tf::DynamicPartitioner<>>(12);
}

// static
TEST_CASE("StatefulParallelFor.Static.1thread" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner<>>(1);
}

TEST_CASE("StatefulParallelFor.Static.2threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner<>>(2);
}

TEST_CASE("StatefulParallelFor.Static.3threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner<>>(3);
}

TEST_CASE("StatefulParallelFor.Static.4threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner<>>(4);
}

TEST_CASE("StatefulParallelFor.Static.5threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner<>>(5);
}

TEST_CASE("StatefulParallelFor.Static.6threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner<>>(6);
}

TEST_CASE("StatefulParallelFor.Static.7threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner<>>(7);
}

TEST_CASE("StatefulParallelFor.Static.8threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner<>>(8);
}

TEST_CASE("StatefulParallelFor.Static.9threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner<>>(9);
}

TEST_CASE("StatefulParallelFor.Static.10threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner<>>(10);
}

TEST_CASE("StatefulParallelFor.Static.11threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner<>>(11);
}

TEST_CASE("StatefulParallelFor.Static.12threads" * doctest::timeout(300)) {
  stateful_for_each<tf::StaticPartitioner<>>(12);
}

// random
TEST_CASE("StatefulParallelFor.Random.1thread" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner<>>(1);
}

TEST_CASE("StatefulParallelFor.Random.2threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner<>>(2);
}

TEST_CASE("StatefulParallelFor.Random.3threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner<>>(3);
}

TEST_CASE("StatefulParallelFor.Random.4threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner<>>(4);
}

TEST_CASE("StatefulParallelFor.Random.5threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner<>>(5);
}

TEST_CASE("StatefulParallelFor.Random.6threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner<>>(6);
}

TEST_CASE("StatefulParallelFor.Random.7threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner<>>(7);
}

TEST_CASE("StatefulParallelFor.Random.8threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner<>>(8);
}

TEST_CASE("StatefulParallelFor.Random.9threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner<>>(9);
}

TEST_CASE("StatefulParallelFor.Random.10threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner<>>(10);
}

TEST_CASE("StatefulParallelFor.Random.11threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner<>>(11);
}

TEST_CASE("StatefulParallelFor.Random.12threads" * doctest::timeout(300)) {
  stateful_for_each<tf::RandomPartitioner<>>(12);
}

// ----------------------------------------------------------------------------
// for_each_index negative index
// ----------------------------------------------------------------------------

void test_for_each_index_negative(unsigned w) {    
  tf::Executor executor(w);
  for(int beg=10; beg>=-10; --beg) {
    for(int end=beg; end>=-10; --end) {
      for(int s=1; s<=beg-end; ++s) {
        int n = 0;
        for(int b = beg; b>end; b-=s) {
          ++n;
        }
        //for(size_t c=0; c<10; c++) {
          tf::Taskflow tf;
          std::atomic<int> counter {0};
          tf.for_each_index(beg, end, -s, [&] (auto) {
            counter.fetch_add(1, std::memory_order_relaxed);
          }/*, c*/);
          executor.run(tf);
          executor.wait_for_all();
          REQUIRE(n == counter);
        //}
      }
    }
  }
}

TEST_CASE("ForEachIndex.NegativeIndex.1thread" * doctest::timeout(300)) {
  test_for_each_index_negative(1);
}

TEST_CASE("ForEachIndex.NegativeIndex.2threads" * doctest::timeout(300)) {
  test_for_each_index_negative(2);
}

TEST_CASE("ForEachIndex.NegativeIndex.3threads" * doctest::timeout(300)) {
  test_for_each_index_negative(3);
}

TEST_CASE("ForEachIndex.NegativeIndex.4threads" * doctest::timeout(300)) {
  test_for_each_index_negative(4);
}

// ----------------------------------------------------------------------------
// ForEachIndex.InvalidRange
// ----------------------------------------------------------------------------

TEST_CASE("ForEachIndex.InvalidRange" * doctest::timeout(300)) {
  std::atomic<size_t> counter(0);
	tf::Executor ex;
	tf::Taskflow flow;
	flow.for_each_index(0, -1, 1, [&](int i) {
		counter.fetch_add(i, std::memory_order_relaxed);
	});
	ex.run(flow).wait();
	REQUIRE(counter == 0);
}

// ----------------------------------------------------------------------------
// ForEachIndex.HeterogeneousRange
// ----------------------------------------------------------------------------

TEST_CASE("ForEachIndex.HeterogeneousRange" * doctest::timeout(300)) {
	std::atomic<size_t> counter(0);
	tf::Executor ex;
	tf::Taskflow flow;

	size_t from = 1;
	size_t to = 10;
	size_t step = 1;

	flow.for_each_index(from, to, step, [&](size_t i) {
		counter.fetch_add(i, std::memory_order_relaxed);
	});
	ex.run(flow).wait();
	REQUIRE(counter == to * (to - 1) / 2);
}

// ----------------------------------------------------------------------------
// range-based for_each_index 
// ----------------------------------------------------------------------------

template <typename P>
void range_based_for_each_index(unsigned w) {    
  tf::Executor executor(w);
  tf::Taskflow taskflow;
  std::atomic<size_t> counter {0};

  for(int beg=10; beg>=-10; --beg) {
    for(int end=beg; end>=-10; --end) {
      for(int s=1; s<=beg-end; ++s) {

        size_t n = tf::distance(beg, end, -s);

        for(size_t c=0; c<10; c++) {
          taskflow.clear();
          counter = 0;

          tf::IndexRange range(beg, end, -s);
          REQUIRE(range.size() == n);

          taskflow.for_each_index(range, [&] (tf::IndexRange<int> range) {
            size_t l = 0;
            for(auto j=range.begin(); j>range.end(); j+=range.step_size()) {
              l++;
            }
            REQUIRE(range.size() == l);
            counter.fetch_add(l, std::memory_order_relaxed);
          }, P(c));
          executor.run(taskflow).wait();
          REQUIRE(n == counter);
        }
      }
    }
  }
}

TEST_CASE("ForEach.NegativeIndexRange.Static.1thread" * doctest::timeout(300)) {
  range_based_for_each_index<tf::StaticPartitioner<>>(1);
}

TEST_CASE("ForEach.NegativeIndexRange.Static.2threads" * doctest::timeout(300)) {
  range_based_for_each_index<tf::StaticPartitioner<>>(2);
}

TEST_CASE("ForEach.NegativeIndexRange.Static.3threads" * doctest::timeout(300)) {
  range_based_for_each_index<tf::StaticPartitioner<>>(3);
}

TEST_CASE("ForEach.NegativeIndexRange.Static.4threads" * doctest::timeout(300)) {
  range_based_for_each_index<tf::StaticPartitioner<>>(4);
}

TEST_CASE("ForEach.NegativeIndexRange.Guided.1thread" * doctest::timeout(300)) {
  range_based_for_each_index<tf::GuidedPartitioner<>>(1);
}

TEST_CASE("ForEach.NegativeIndexRange.Guided.2threads" * doctest::timeout(300)) {
  range_based_for_each_index<tf::GuidedPartitioner<>>(2);
}

TEST_CASE("ForEach.NegativeIndexRange.Guided.3threads" * doctest::timeout(300)) {
  range_based_for_each_index<tf::GuidedPartitioner<>>(3);
}

TEST_CASE("ForEach.NegativeIndexRange.Guided.4threads" * doctest::timeout(300)) {
  range_based_for_each_index<tf::GuidedPartitioner<>>(4);
}

TEST_CASE("ForEach.NegativeIndexRange.Dynamic.1thread" * doctest::timeout(300)) {
  range_based_for_each_index<tf::DynamicPartitioner<>>(1);
}

TEST_CASE("ForEach.NegativeIndexRange.Dynamic.2threads" * doctest::timeout(300)) {
  range_based_for_each_index<tf::DynamicPartitioner<>>(2);
}

TEST_CASE("ForEach.NegativeIndexRange.Dynamic.3threads" * doctest::timeout(300)) {
  range_based_for_each_index<tf::DynamicPartitioner<>>(3);
}

TEST_CASE("ForEach.NegativeIndexRange.Dynamic.4threads" * doctest::timeout(300)) {
  range_based_for_each_index<tf::DynamicPartitioner<>>(4);
}

// ----------------------------------------------------------------------------
// stateful range-based for_each_index 
// ----------------------------------------------------------------------------

template <typename P>
void stateful_range_based_for_each_index(unsigned w) {    

  tf::Executor executor(w);
  tf::Taskflow taskflow;
  std::atomic<size_t> counter {0};

  for(int beg=10; beg>=-10; --beg) {
    for(int end=beg; end>=-10; --end) {
      for(int s=1; s<=beg-end; ++s) {

        size_t n = tf::distance(beg, end, -s);

        for(size_t c=0; c<10; c++) {
          taskflow.clear();
          counter = 0;
          
          tf::IndexRange range(0, 0, 0);

          auto set_range = taskflow.emplace([&](){
            range.begin(beg)
                 .end(end)
                 .step_size(-s);
            REQUIRE(range.size() == n);
          });

          auto loop_range = taskflow.for_each_index(std::ref(range), [&] (tf::IndexRange<int> range) {
            size_t l = 0;
            for(auto j=range.begin(); j>range.end(); j+=range.step_size()) {
              l++;
            }
            REQUIRE(range.size() == l);
            counter.fetch_add(l, std::memory_order_relaxed);
          }, P(c));

          set_range.precede(loop_range);

          executor.run(taskflow).wait();
          REQUIRE(n == counter);
        }
      }
    }
  }
}

TEST_CASE("StatefulForEach.NegativeIndexRange.Static.1thread" * doctest::timeout(300)) {
  stateful_range_based_for_each_index<tf::StaticPartitioner<>>(1);
}

TEST_CASE("StatefulForEach.NegativeIndexRange.Static.2threads" * doctest::timeout(300)) {
  stateful_range_based_for_each_index<tf::StaticPartitioner<>>(2);
}

TEST_CASE("StatefulForEach.NegativeIndexRange.Static.3threads" * doctest::timeout(300)) {
  stateful_range_based_for_each_index<tf::StaticPartitioner<>>(3);
}

TEST_CASE("StatefulForEach.NegativeIndexRange.Static.4threads" * doctest::timeout(300)) {
  stateful_range_based_for_each_index<tf::StaticPartitioner<>>(4);
}

TEST_CASE("StatefulForEach.NegativeIndexRange.Dynamic.1thread" * doctest::timeout(300)) {
  stateful_range_based_for_each_index<tf::DynamicPartitioner<>>(1);
}

TEST_CASE("StatefulForEach.NegativeIndexRange.Dynamic.2threads" * doctest::timeout(300)) {
  stateful_range_based_for_each_index<tf::DynamicPartitioner<>>(2);
}

TEST_CASE("StatefulForEach.NegativeIndexRange.Dynamic.3threads" * doctest::timeout(300)) {
  stateful_range_based_for_each_index<tf::DynamicPartitioner<>>(3);
}

TEST_CASE("StatefulForEach.NegativeIndexRange.Dynamic.4threads" * doctest::timeout(300)) {
  stateful_range_based_for_each_index<tf::DynamicPartitioner<>>(4);
}

TEST_CASE("StatefulForEach.NegativeIndexRange.Guided.1thread" * doctest::timeout(300)) {
  stateful_range_based_for_each_index<tf::GuidedPartitioner<>>(1);
}

TEST_CASE("StatefulForEach.NegativeIndexRange.Guided.2threads" * doctest::timeout(300)) {
  stateful_range_based_for_each_index<tf::GuidedPartitioner<>>(2);
}

TEST_CASE("StatefulForEach.NegativeIndexRange.Guided.3threads" * doctest::timeout(300)) {
  stateful_range_based_for_each_index<tf::GuidedPartitioner<>>(3);
}

TEST_CASE("StatefulForEach.NegativeIndexRange.Guided.4threads" * doctest::timeout(300)) {
  stateful_range_based_for_each_index<tf::GuidedPartitioner<>>(4);
}

// ----------------------------------------------------------------------------
// Closure Wrapper
// ----------------------------------------------------------------------------

int& GetThreadSpecificContext()
{
    thread_local int context = 0;
    return context;
}

const int UPPER = 1000;

// ChunkContext
TEST_CASE("ClosureWrapper.for_each_index.Static" * doctest::timeout(300))
{
  for (int tc = 1; tc < 64; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    std::atomic<int> count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(100, 0);
    taskflow.for_each_index(0, 100, 1, 
      [&](int i) { 
        count++; 
        range[i] = GetThreadSpecificContext(); 
      },
      tf::StaticPartitioner(1, [&](auto &&task){
        wrapper_called_count++;
        GetThreadSpecificContext() = tc;
        task();
        GetThreadSpecificContext() = 0;
      })
    );
    executor.run(taskflow).wait();

    REQUIRE(count == 100);
    REQUIRE(range == std::vector<int>(100, tc));
    REQUIRE(wrapper_called_count == tc);
  }
}

// ChunkContext
TEST_CASE("ClosureWrapper.for_each_index.Dynamic" * doctest::timeout(300))
{
  for (int tc = 1; tc < 64; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    std::atomic<int> count = 0;
    std::vector<int> range(UPPER, 0);
    tf::Taskflow taskflow;
    taskflow.for_each_index(0, UPPER, 1, 
      [&](int i){ 
        count++; range[i] = GetThreadSpecificContext();
      },
      tf::DynamicPartitioner(UPPER/tc/2, [&](auto&& task){
        wrapper_called_count++;
        GetThreadSpecificContext() = tc;
        task();
        GetThreadSpecificContext() = 0;
      })
    );
    executor.run(taskflow).wait();

    REQUIRE(count == UPPER);
    REQUIRE(range == std::vector<int>(UPPER, tc));
    // Dynamic partitioner will spawn sometimes less tasks
    REQUIRE(wrapper_called_count <= tc);
  }
}


TEST_CASE("ClosureWrapper.for_each.Static" * doctest::timeout(300))
{
  for (int tc = 1; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    std::atomic<int> count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER, 0);
    taskflow.for_each(range.begin(), range.end(), 
      [&](int& i) { 
        count++; 
        i = GetThreadSpecificContext(); 
      },
      tf::StaticPartitioner(1, [&](auto&& task){
        wrapper_called_count++;
        GetThreadSpecificContext() = tc;
        task();
        GetThreadSpecificContext() = 0;
      })
    );
    executor.run(taskflow).wait();

    REQUIRE(count == UPPER);
    REQUIRE(range == std::vector<int>(UPPER, tc));
    REQUIRE(wrapper_called_count == tc);
  }
}

TEST_CASE("ClosureWrapper.for_each.Dynamic" * doctest::timeout(300))
{
  for (int tc = 1; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER, 0);
    taskflow.for_each(range.begin(), range.end(), 
      [](int& i) {
        i = GetThreadSpecificContext(); 
      },
      tf::DynamicPartitioner(1, [&](auto&& task){
        wrapper_called_count++;
        GetThreadSpecificContext() = tc;
        task();
        GetThreadSpecificContext() = 0;
      })
    );
    executor.run(taskflow).wait();

    REQUIRE(range == std::vector<int>(UPPER, tc));
    // Dynamic scheduling is not obliged to load all threads with iterations, so <= is appropriate here
    REQUIRE(wrapper_called_count <= tc); 
  }
}

//// ----------------------------------------------------------------------------
//// Parallel For Exception
//// ----------------------------------------------------------------------------
//
//void parallel_for_exception(unsigned W) {
//
//  tf::Taskflow taskflow;
//  tf::Executor executor(W);
//
//  std::vector<int> data(1000000);
//
//  // for_each
//  taskflow.for_each(data.begin(), data.end(), [](int){
//    throw std::runtime_error("x");
//  });
//  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "x", std::runtime_error);
//  
//  // for_each_index
//  taskflow.clear();
//  taskflow.for_each_index(0, 10000, 1, [](int){
//    throw std::runtime_error("y");
//  });
//  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "y", std::runtime_error);
//}
//
//TEST_CASE("ParallelFor.Exception.1thread") {
//  parallel_for_exception(1);
//}
//
//TEST_CASE("ParallelFor.Exception.2threads") {
//  parallel_for_exception(2);
//}
//
//TEST_CASE("ParallelFor.Exception.3threads") {
//  parallel_for_exception(3);
//}
//
//TEST_CASE("ParallelFor.Exception.4threads") {
//  parallel_for_exception(4);
//}

// ----------------------------------------------------------------------------
// Multiple For Each
// ----------------------------------------------------------------------------

template <typename P>
void multiple_for_each(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  const int N = 1000;
  const int M = 1000;

  std::array<std::vector<int>, N> vectors;

  for(auto& vec : vectors) {
    vec.resize(M);
  }

  for(int i=0; i<N; i++) {

    // chain i in charge of vectors[i]

    auto init = taskflow.emplace([&, i](){
      for(auto& j : vectors[i]) {
        j = -i;
      }
    });

    size_t c = rand() % 20;

    auto for_each = taskflow.for_each(vectors[i].begin(), vectors[i].end(), [i] (auto& j) {
      REQUIRE(j == -i);
      j = i;
      //executor.async([](){});
    }, P(c));

    auto for_each_index = taskflow.for_each_index(0, M, 1, [i, &vec=vectors[i]](size_t j){
      REQUIRE(vec[j] == i);
    }, P(c));

    init.precede(for_each);
    for_each.precede(for_each_index);
  }

  executor.run(taskflow).wait();
}

TEST_CASE("MultipleParallelForEach.Static.1thread") {
  multiple_for_each<tf::StaticPartitioner<>>(1);
}

TEST_CASE("MultipleParallelForEach.Static.2threads") {
  multiple_for_each<tf::StaticPartitioner<>>(2);
}

TEST_CASE("MultipleParallelForEach.Static.3threads") {
  multiple_for_each<tf::StaticPartitioner<>>(3);
}

TEST_CASE("MultipleParallelForEach.Static.4threads") {
  multiple_for_each<tf::StaticPartitioner<>>(4);
}

TEST_CASE("MultipleParallelForEach.Dynamic.1thread") {
  multiple_for_each<tf::DynamicPartitioner<>>(1);
}

TEST_CASE("MultipleParallelForEach.Dynamic.2threads") {
  multiple_for_each<tf::DynamicPartitioner<>>(2);
}

TEST_CASE("MultipleParallelForEach.Dynamic.3threads") {
  multiple_for_each<tf::DynamicPartitioner<>>(3);
}

TEST_CASE("MultipleParallelForEach.Dynamic.4threads") {
  multiple_for_each<tf::DynamicPartitioner<>>(4);
}

TEST_CASE("MultipleParallelForEach.Guided.1thread") {
  multiple_for_each<tf::GuidedPartitioner<>>(1);
}

TEST_CASE("MultipleParallelForEach.Guided.2threads") {
  multiple_for_each<tf::GuidedPartitioner<>>(2);
}

TEST_CASE("MultipleParallelForEach.Guided.3threads") {
  multiple_for_each<tf::GuidedPartitioner<>>(3);
}

TEST_CASE("MultipleParallelForEach.Guided.4threads") {
  multiple_for_each<tf::GuidedPartitioner<>>(4);
}


// ----------------------------------------------------------------------------
// Async
// ----------------------------------------------------------------------------
void async(unsigned W) {

  tf::Executor executor(W);
  
  std::vector<int> data;

  for(size_t N=0; N<=65536; N =((N == 0) ? 1 : N << 1)) {

    data.resize(N);
  
    // initialize data to -10 and 10
    executor.async(tf::make_for_each_task(
      data.begin(), data.begin() + N/2, [](int& d){ d = -10; }
    )); 
    
    executor.async(tf::make_for_each_index_task(
      N/2, N, size_t{1}, [&] (size_t i) { data[i] = 10; }
    ));

    executor.wait_for_all();

    for(size_t i=0; i<N; i++) {
      REQUIRE(data[i] == ((i<N/2) ? -10 : 10));
    }
  }

}

TEST_CASE("ParallelFor.Async.1thread" * doctest::timeout(300)) {
  async(1);
}

TEST_CASE("ParallelFor.Async.2threads" * doctest::timeout(300)) {
  async(2);
}

TEST_CASE("ParallelFor.Async.3threads" * doctest::timeout(300)) {
  async(3);
}

TEST_CASE("ParallelFor.Async.4threads" * doctest::timeout(300)) {
  async(4);
}

TEST_CASE("ParallelFor.Async.5threads" * doctest::timeout(300)) {
  async(5);
}

TEST_CASE("ParallelFor.Async.6threads" * doctest::timeout(300)) {
  async(6);
}

TEST_CASE("ParallelFor.Async.7threads" * doctest::timeout(300)) {
  async(7);
}

TEST_CASE("ParallelFor.Async.8threads" * doctest::timeout(300)) {
  async(8);
}

// ----------------------------------------------------------------------------
// Silent Async
// ----------------------------------------------------------------------------

void silent_async(unsigned W) {

  size_t N = 65536;

  tf::Executor executor(W);
  
  std::vector<int> data(N);
  
  // initialize data to 10 and -10
  executor.silent_async(tf::make_for_each_task(
    data.begin(), data.begin() + N/2, [](int& d){ d = 10; }
  )); 
  
  executor.silent_async(tf::make_for_each_index_task(
    N/2, N, size_t{1}, [&] (size_t i) { data[i] = -10; }
  ));

  executor.wait_for_all();

  for(size_t i=0; i<N; i++) {
    REQUIRE(data[i] == ((i<N/2) ? 10 : -10));
  }
}

TEST_CASE("ParallelFor.SilentAsync.1thread" * doctest::timeout(300)) {
  silent_async(1);
}

TEST_CASE("ParallelFor.SilentAsync.2threads" * doctest::timeout(300)) {
  silent_async(2);
}

TEST_CASE("ParallelFor.SilentAsync.3threads" * doctest::timeout(300)) {
  silent_async(3);
}

TEST_CASE("ParallelFor.SilentAsync.4threads" * doctest::timeout(300)) {
  silent_async(4);
}

TEST_CASE("ParallelFor.SilentAsync.5threads" * doctest::timeout(300)) {
  silent_async(5);
}

TEST_CASE("ParallelFor.SilentAsync.6threads" * doctest::timeout(300)) {
  silent_async(6);
}

TEST_CASE("ParallelFor.SilentAsync.7threads" * doctest::timeout(300)) {
  silent_async(7);
}

TEST_CASE("ParallelFor.SilentAsync.8threads" * doctest::timeout(300)) {
  silent_async(8);
}

// ----------------------------------------------------------------------------
// DependentAsync
// ----------------------------------------------------------------------------

void dependent_async(unsigned W) {

  tf::Executor executor(W);
  
  std::vector<int> data;

  for(size_t N=0; N<=65536; N =((N == 0) ? 1 : N << 1)) {

    data.resize(N);
  
    // initialize data to -10 and 10
    executor.dependent_async(tf::make_for_each_task(
      data.begin(), data.begin() + N/2, [](int& d){ d = -10; }
    )); 
    
    executor.dependent_async(tf::make_for_each_index_task(
      N/2, N, size_t{1}, [&] (size_t i) { data[i] = 10; }
    ));

    executor.wait_for_all();

    for(size_t i=0; i<N; i++) {
      REQUIRE(data[i] == ((i<N/2) ? -10 : 10));
    }
  }

}

TEST_CASE("ParallelFor.DependentAsync.1thread" * doctest::timeout(300)) {
  dependent_async(1);
}

TEST_CASE("ParallelFor.DependentAsync.2threads" * doctest::timeout(300)) {
  dependent_async(2);
}

TEST_CASE("ParallelFor.DependentAsync.3threads" * doctest::timeout(300)) {
  dependent_async(3);
}

TEST_CASE("ParallelFor.DependentAsync.4threads" * doctest::timeout(300)) {
  dependent_async(4);
}

TEST_CASE("ParallelFor.DependentAsync.5threads" * doctest::timeout(300)) {
  dependent_async(5);
}

TEST_CASE("ParallelFor.DependentAsync.6threads" * doctest::timeout(300)) {
  dependent_async(6);
}

TEST_CASE("ParallelFor.DependentAsync.7threads" * doctest::timeout(300)) {
  dependent_async(7);
}

TEST_CASE("ParallelFor.DependentAsync.8threads" * doctest::timeout(300)) {
  dependent_async(8);
} 

// ----------------------------------------------------------------------------
// Silent DependentAsync
// ----------------------------------------------------------------------------

void silent_dependent_async(unsigned W) {

  size_t N = 65536;

  tf::Executor executor(W);
  
  std::vector<int> data(N);
  
  // initialize data to 10 and -10
  executor.silent_dependent_async(tf::make_for_each_task(
    data.begin(), data.begin() + N/2, [](int& d){ d = 10; }
  )); 
  
  executor.silent_dependent_async(tf::make_for_each_index_task(
    N/2, N, size_t{1}, [&] (size_t i) { data[i] = -10; }
  ));

  executor.wait_for_all();

  for(size_t i=0; i<N; i++) {
    REQUIRE(data[i] == ((i<N/2) ? 10 : -10));
  }
}

TEST_CASE("ParallelFor.SilentDependentAsync.1thread" * doctest::timeout(300)) {
  silent_dependent_async(1);
}

TEST_CASE("ParallelFor.SilentDependentAsync.2threads" * doctest::timeout(300)) {
  silent_dependent_async(2);
}

TEST_CASE("ParallelFor.SilentDependentAsync.3threads" * doctest::timeout(300)) {
  silent_dependent_async(3);
}

TEST_CASE("ParallelFor.SilentDependentAsync.4threads" * doctest::timeout(300)) {
  silent_dependent_async(4);
}

TEST_CASE("ParallelFor.SilentDependentAsync.5threads" * doctest::timeout(300)) {
  silent_dependent_async(5);
}

TEST_CASE("ParallelFor.SilentDependentAsync.6threads" * doctest::timeout(300)) {
  silent_dependent_async(6);
}

TEST_CASE("ParallelFor.SilentDependentAsync.7threads" * doctest::timeout(300)) {
  silent_dependent_async(7);
}

TEST_CASE("ParallelFor.SilentDependentAsync.8threads" * doctest::timeout(300)) {
  silent_dependent_async(8);
}

