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

          taskflow.for_each_by_index(range, [&] (tf::IndexRange<int> lrange) {
            size_t l = 0;
            for(auto j=lrange.begin(); j>lrange.end(); j+=lrange.step_size()) {
              l++;
            }
            REQUIRE(lrange.size() == l);
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

          auto loop_range = taskflow.for_each_by_index(std::ref(range), [&] (tf::IndexRange<int> lrange) {
            size_t l = 0;
            for(auto j=lrange.begin(); j>lrange.end(); j+=lrange.step_size()) {
              l++;
            }
            REQUIRE(lrange.size() == l);
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

// ----------------------------------------------------------------------------
// Nested for loop
// ----------------------------------------------------------------------------

void nested_for_loop(unsigned W) {

  int N1 = 2048;
  int N2 = 2048;

  tf::Executor executor(W);
  
  // initialize the data
  std::vector<std::vector<int>> data(N1);
  
  for(int i=0; i<N1; ++i) {
    data[i].resize(N2);
  } 

  // initialize data[i][j] = i
  executor.async(tf::make_for_each_index_task(0, N1, 1, [&](int i){ 
    executor.async(tf::make_for_each_index_task(0, N2, 1, [&, i](int j) {
      data[i][j] = i + j;
    }));
  })); 
  
  executor.wait_for_all();

  for(int i=0; i<N1; i++) {
    for(int j=0; j<N2; ++j) {
      REQUIRE(data[i][j] == i + j);
    }
  }
}

TEST_CASE("ParallelFor.Nested.1thread" * doctest::timeout(300)) {
  nested_for_loop(1);
}

TEST_CASE("ParallelFor.Nested.2threads" * doctest::timeout(300)) {
  nested_for_loop(2);
}

TEST_CASE("ParallelFor.Nested.3threads" * doctest::timeout(300)) {
  nested_for_loop(3);
}

TEST_CASE("ParallelFor.Nested.4threads" * doctest::timeout(300)) {
  nested_for_loop(4);
}

TEST_CASE("ParallelFor.Nested.5threads" * doctest::timeout(300)) {
  nested_for_loop(5);
}

TEST_CASE("ParallelFor.Nested.6threads" * doctest::timeout(300)) {
  nested_for_loop(6);
}

TEST_CASE("ParallelFor.Nested.7threads" * doctest::timeout(300)) {
  nested_for_loop(7);
}

TEST_CASE("ParallelFor.Nested.8threads" * doctest::timeout(300)) {
  nested_for_loop(8);
}

// ----------------------------------------------------------------------------
// MD range-based for_each_by_index (IndexRangeMDLike)
// ----------------------------------------------------------------------------

// 2D: verifies every (i, j) sub-box is visited exactly once and covers the
// full Cartesian product. Sweeps di x dj sizes to stress consume_chunk
// boundary alignment across all chunk sizes.
template <typename P>
void md_for_each_by_index_2d(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  for(int di = 1; di <= 17; di++) {
    for(int dj = 1; dj <= 23; dj++) {
      for(size_t c : {0, 1, 3, 7, 99}) {

        tf::IndexRange<int, 2> range(
          tf::IndexRange<int>(0, di, 1),
          tf::IndexRange<int>(0, dj, 1)
        );

        const size_t N = range.size();  // di * dj

        std::vector<int> visited(N, 0);
        std::atomic<size_t> total{0};

        taskflow.clear();
        taskflow.for_each_by_index(range,
          [&, dj](const tf::IndexRange<int, 2>& box) {
            for(int i = box.dim(0).begin(); i < box.dim(0).end(); i += box.dim(0).step_size()) {
              for(int j = box.dim(1).begin(); j < box.dim(1).end(); j += box.dim(1).step_size()) {
                visited[i * dj + j]++;
                total.fetch_add(1, std::memory_order_relaxed);
              }
            }
          }, P(c)
        );

        executor.run(taskflow).wait();

        REQUIRE(total == N);
        for(size_t k = 0; k < N; k++) {
          REQUIRE(visited[k] == 1);
        }
      }
    }
  }
}

// 3D: same idea extended to three dimensions, with non-unit step sizes to
// exercise coordinate arithmetic in consume_chunk more thoroughly.
template <typename P>
void md_for_each_by_index_3d(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  const int D0 = 4, D1 = 10, D2 = 9;
  const int S0 = 1, S1 = 2, S2 = 3;

  tf::IndexRange<int, 3> range(
    tf::IndexRange<int>(0, D0 * S0, S0),
    tf::IndexRange<int>(0, D1 * S1, S1),
    tf::IndexRange<int>(0, D2 * S2, S2)
  );

  const size_t N = range.size();  // D0 * D1 * D2

  for(size_t c : {0, 1, 3, 7, 99}) {

    std::vector<int> visited(N, 0);
    std::atomic<size_t> total{0};

    taskflow.clear();
    taskflow.for_each_by_index(range,
      [&](const tf::IndexRange<int, 3>& box) {
        for(int i = box.dim(0).begin(); i < box.dim(0).end(); i += box.dim(0).step_size()) {
          for(int j = box.dim(1).begin(); j < box.dim(1).end(); j += box.dim(1).step_size()) {
            for(int k = box.dim(2).begin(); k < box.dim(2).end(); k += box.dim(2).step_size()) {
              size_t fi = (i / S0) * (D1 * D2) + (j / S1) * D2 + (k / S2);
              visited[fi]++;
              total.fetch_add(1, std::memory_order_relaxed);
            }
          }
        }
      }, P(c)
    );

    executor.run(taskflow).wait();

    REQUIRE(total == N);
    for(size_t k = 0; k < N; k++) {
      REQUIRE(visited[k] == 1);
    }
  }
}

// Stateful 2D: range is not known until an upstream init task runs,
// matching the stateful pattern used in stateful_range_based_for_each_index.
template <typename P>
void stateful_md_for_each_by_index_2d(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  tf::IndexRange<int, 2> range(
    tf::IndexRange<int>(0, 0, 1),
    tf::IndexRange<int>(0, 0, 1)
  );

  for(int di = 1; di <= 17; di++) {
    for(int dj = 1; dj <= 23; dj++) {
      for(size_t c : {0, 1, 3, 7, 99}) {

        const size_t N = static_cast<size_t>(di * dj);

        std::vector<int> visited(N, 0);
        std::atomic<size_t> total{0};

        taskflow.clear();

        auto init = taskflow.emplace([&, di, dj]() {
          range.dim(0).reset(0, di, 1);
          range.dim(1).reset(0, dj, 1);
        });

        auto loop = taskflow.for_each_by_index(std::ref(range),
          [&, dj](const tf::IndexRange<int, 2>& box) {
            for(int i = box.dim(0).begin(); i < box.dim(0).end(); i += box.dim(0).step_size()) {
              for(int j = box.dim(1).begin(); j < box.dim(1).end(); j += box.dim(1).step_size()) {
                visited[i * dj + j]++;
                total.fetch_add(1, std::memory_order_relaxed);
              }
            }
          }, P(c)
        );

        init.precede(loop);

        executor.run(taskflow).wait();

        REQUIRE(total == N);
        for(size_t k = 0; k < N; k++) {
          REQUIRE(visited[k] == 1);
        }
      }
    }
  }
}

// ---- 2D TEST CASES ----------------------------------------------------------

TEST_CASE("MDForEachByIndex.2D.Guided.1thread" * doctest::timeout(300)) {
  md_for_each_by_index_2d<tf::GuidedPartitioner<>>(1);
}
TEST_CASE("MDForEachByIndex.2D.Guided.2threads" * doctest::timeout(300)) {
  md_for_each_by_index_2d<tf::GuidedPartitioner<>>(2);
}
TEST_CASE("MDForEachByIndex.2D.Guided.4threads" * doctest::timeout(300)) {
  md_for_each_by_index_2d<tf::GuidedPartitioner<>>(4);
}
TEST_CASE("MDForEachByIndex.2D.Guided.8threads" * doctest::timeout(300)) {
  md_for_each_by_index_2d<tf::GuidedPartitioner<>>(8);
}

TEST_CASE("MDForEachByIndex.2D.Dynamic.1thread" * doctest::timeout(300)) {
  md_for_each_by_index_2d<tf::DynamicPartitioner<>>(1);
}
TEST_CASE("MDForEachByIndex.2D.Dynamic.2threads" * doctest::timeout(300)) {
  md_for_each_by_index_2d<tf::DynamicPartitioner<>>(2);
}
TEST_CASE("MDForEachByIndex.2D.Dynamic.4threads" * doctest::timeout(300)) {
  md_for_each_by_index_2d<tf::DynamicPartitioner<>>(4);
}
TEST_CASE("MDForEachByIndex.2D.Dynamic.8threads" * doctest::timeout(300)) {
  md_for_each_by_index_2d<tf::DynamicPartitioner<>>(8);
}

TEST_CASE("MDForEachByIndex.2D.Static.1thread" * doctest::timeout(300)) {
  md_for_each_by_index_2d<tf::StaticPartitioner<>>(1);
}
TEST_CASE("MDForEachByIndex.2D.Static.2threads" * doctest::timeout(300)) {
  md_for_each_by_index_2d<tf::StaticPartitioner<>>(2);
}
TEST_CASE("MDForEachByIndex.2D.Static.4threads" * doctest::timeout(300)) {
  md_for_each_by_index_2d<tf::StaticPartitioner<>>(4);
}
TEST_CASE("MDForEachByIndex.2D.Static.8threads" * doctest::timeout(300)) {
  md_for_each_by_index_2d<tf::StaticPartitioner<>>(8);
}

TEST_CASE("MDForEachByIndex.2D.Random.1thread" * doctest::timeout(300)) {
  md_for_each_by_index_2d<tf::RandomPartitioner<>>(1);
}
TEST_CASE("MDForEachByIndex.2D.Random.2threads" * doctest::timeout(300)) {
  md_for_each_by_index_2d<tf::RandomPartitioner<>>(2);
}
TEST_CASE("MDForEachByIndex.2D.Random.4threads" * doctest::timeout(300)) {
  md_for_each_by_index_2d<tf::RandomPartitioner<>>(4);
}
TEST_CASE("MDForEachByIndex.2D.Random.8threads" * doctest::timeout(300)) {
  md_for_each_by_index_2d<tf::RandomPartitioner<>>(8);
}

// ---- 3D TEST CASES ----------------------------------------------------------

TEST_CASE("MDForEachByIndex.3D.Guided.1thread" * doctest::timeout(300)) {
  md_for_each_by_index_3d<tf::GuidedPartitioner<>>(1);
}
TEST_CASE("MDForEachByIndex.3D.Guided.2threads" * doctest::timeout(300)) {
  md_for_each_by_index_3d<tf::GuidedPartitioner<>>(2);
}
TEST_CASE("MDForEachByIndex.3D.Guided.4threads" * doctest::timeout(300)) {
  md_for_each_by_index_3d<tf::GuidedPartitioner<>>(4);
}
TEST_CASE("MDForEachByIndex.3D.Guided.8threads" * doctest::timeout(300)) {
  md_for_each_by_index_3d<tf::GuidedPartitioner<>>(8);
}

TEST_CASE("MDForEachByIndex.3D.Dynamic.1thread" * doctest::timeout(300)) {
  md_for_each_by_index_3d<tf::DynamicPartitioner<>>(1);
}
TEST_CASE("MDForEachByIndex.3D.Dynamic.2threads" * doctest::timeout(300)) {
  md_for_each_by_index_3d<tf::DynamicPartitioner<>>(2);
}
TEST_CASE("MDForEachByIndex.3D.Dynamic.4threads" * doctest::timeout(300)) {
  md_for_each_by_index_3d<tf::DynamicPartitioner<>>(4);
}
TEST_CASE("MDForEachByIndex.3D.Dynamic.8threads" * doctest::timeout(300)) {
  md_for_each_by_index_3d<tf::DynamicPartitioner<>>(8);
}

TEST_CASE("MDForEachByIndex.3D.Static.1thread" * doctest::timeout(300)) {
  md_for_each_by_index_3d<tf::StaticPartitioner<>>(1);
}
TEST_CASE("MDForEachByIndex.3D.Static.2threads" * doctest::timeout(300)) {
  md_for_each_by_index_3d<tf::StaticPartitioner<>>(2);
}
TEST_CASE("MDForEachByIndex.3D.Static.4threads" * doctest::timeout(300)) {
  md_for_each_by_index_3d<tf::StaticPartitioner<>>(4);
}
TEST_CASE("MDForEachByIndex.3D.Static.8threads" * doctest::timeout(300)) {
  md_for_each_by_index_3d<tf::StaticPartitioner<>>(8);
}

TEST_CASE("MDForEachByIndex.3D.Random.1thread" * doctest::timeout(300)) {
  md_for_each_by_index_3d<tf::RandomPartitioner<>>(1);
}
TEST_CASE("MDForEachByIndex.3D.Random.2threads" * doctest::timeout(300)) {
  md_for_each_by_index_3d<tf::RandomPartitioner<>>(2);
}
TEST_CASE("MDForEachByIndex.3D.Random.4threads" * doctest::timeout(300)) {
  md_for_each_by_index_3d<tf::RandomPartitioner<>>(4);
}
TEST_CASE("MDForEachByIndex.3D.Random.8threads" * doctest::timeout(300)) {
  md_for_each_by_index_3d<tf::RandomPartitioner<>>(8);
}

// ---- STATEFUL 2D TEST CASES -------------------------------------------------

TEST_CASE("StatefulMDForEachByIndex.2D.Guided.1thread" * doctest::timeout(300)) {
  stateful_md_for_each_by_index_2d<tf::GuidedPartitioner<>>(1);
}
TEST_CASE("StatefulMDForEachByIndex.2D.Guided.2threads" * doctest::timeout(300)) {
  stateful_md_for_each_by_index_2d<tf::GuidedPartitioner<>>(2);
}
TEST_CASE("StatefulMDForEachByIndex.2D.Guided.4threads" * doctest::timeout(300)) {
  stateful_md_for_each_by_index_2d<tf::GuidedPartitioner<>>(4);
}
TEST_CASE("StatefulMDForEachByIndex.2D.Guided.8threads" * doctest::timeout(300)) {
  stateful_md_for_each_by_index_2d<tf::GuidedPartitioner<>>(8);
}

TEST_CASE("StatefulMDForEachByIndex.2D.Dynamic.1thread" * doctest::timeout(300)) {
  stateful_md_for_each_by_index_2d<tf::DynamicPartitioner<>>(1);
}
TEST_CASE("StatefulMDForEachByIndex.2D.Dynamic.2threads" * doctest::timeout(300)) {
  stateful_md_for_each_by_index_2d<tf::DynamicPartitioner<>>(2);
}
TEST_CASE("StatefulMDForEachByIndex.2D.Dynamic.4threads" * doctest::timeout(300)) {
  stateful_md_for_each_by_index_2d<tf::DynamicPartitioner<>>(4);
}
TEST_CASE("StatefulMDForEachByIndex.2D.Dynamic.8threads" * doctest::timeout(300)) {
  stateful_md_for_each_by_index_2d<tf::DynamicPartitioner<>>(8);
}

TEST_CASE("StatefulMDForEachByIndex.2D.Static.1thread" * doctest::timeout(300)) {
  stateful_md_for_each_by_index_2d<tf::StaticPartitioner<>>(1);
}
TEST_CASE("StatefulMDForEachByIndex.2D.Static.2threads" * doctest::timeout(300)) {
  stateful_md_for_each_by_index_2d<tf::StaticPartitioner<>>(2);
}
TEST_CASE("StatefulMDForEachByIndex.2D.Static.4threads" * doctest::timeout(300)) {
  stateful_md_for_each_by_index_2d<tf::StaticPartitioner<>>(4);
}
TEST_CASE("StatefulMDForEachByIndex.2D.Static.8threads" * doctest::timeout(300)) {
  stateful_md_for_each_by_index_2d<tf::StaticPartitioner<>>(8);
}

TEST_CASE("StatefulMDForEachByIndex.2D.Random.1thread" * doctest::timeout(300)) {
  stateful_md_for_each_by_index_2d<tf::RandomPartitioner<>>(1);
}
TEST_CASE("StatefulMDForEachByIndex.2D.Random.2threads" * doctest::timeout(300)) {
  stateful_md_for_each_by_index_2d<tf::RandomPartitioner<>>(2);
}
TEST_CASE("StatefulMDForEachByIndex.2D.Random.4threads" * doctest::timeout(300)) {
  stateful_md_for_each_by_index_2d<tf::RandomPartitioner<>>(4);
}
TEST_CASE("StatefulMDForEachByIndex.2D.Random.8threads" * doctest::timeout(300)) {
  stateful_md_for_each_by_index_2d<tf::RandomPartitioner<>>(8);
}

// ----------------------------------------------------------------------------
// Zero-dimension for_each_by_index tests
//
// When any dimension has zero size, size() returns the product of outer dims
// only (those before the first zero).  The scheduler fires the callback for
// each active outer iteration; iterating the zero-size dim (and all inner dims)
// inside the callback produces no iterations — matching sequential nested loop
// behaviour.  The callback must be called exactly size() times and the
// outermost active indices must each be visited exactly once.
// ----------------------------------------------------------------------------

// Helper: build a flat outer-index from a box, counting only active dimensions
// (those before the first zero-size one).  Returns the number of outer
// iterations the box covers so the caller can record each.
// For an ND box with active dims [0..d_active), the outer flat index is
// computed from the box's active dim begins, and the count is the product of
// their sizes.


template <typename P>
void md_for_each_by_index_zero_dim(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  // All scenarios verify that every active outer index is visited exactly once
  // using an atomic integer grid — same pattern as md_for_each_by_index_2d/3d.
  // No assumption is made about the number of callback invocations.
  // For zero-size ranges (size()==0) the callback must never fire.

  // ── 1D: zero-size range ─────────────────────────────────────────────────
  {
    tf::IndexRange<int> range(0, 0, 1);
    std::atomic<int> total{0};
    taskflow.clear();
    taskflow.for_each_by_index(range,
      [&](tf::IndexRange<int>) { total.fetch_add(1, std::memory_order_relaxed); },
      P()
    );
    executor.run(taskflow).wait();
    REQUIRE(total == 0);
  }

  // ── 2D: outermost dim zero → size()=0, callback never fires ─────────────
  {
    tf::IndexRange<int, 2> range(
      tf::IndexRange<int>(0, 0, 1),
      tf::IndexRange<int>(0, 7, 1)
    );
    REQUIRE(range.size() == 0);
    std::atomic<int> total{0};
    taskflow.clear();
    taskflow.for_each_by_index(range,
      [&](const tf::IndexRange<int,2>&) { total.fetch_add(1, std::memory_order_relaxed); },
      P()
    );
    executor.run(taskflow).wait();
    REQUIRE(total == 0);
  }

  // ── 2D: innermost dim zero → size()=5, active dim0 only ─────────────────
  // 5 x 0: every i in [0,5) must be visited exactly once.
  // The inner j-loop must never execute — verified by REQUIRE(false) inside.
  for (size_t c : {size_t{0}, size_t{1}, size_t{3}}) {
    tf::IndexRange<int, 2> range(
      tf::IndexRange<int>(0, 5, 1),
      tf::IndexRange<int>(0, 0, 1)
    );
    const int N = 5;
    REQUIRE(range.size() == static_cast<size_t>(N));
    std::vector<std::atomic<int>> visited(N);
    for (auto& v : visited) v.store(0);
    taskflow.clear();
    taskflow.for_each_by_index(range,
      [&](const tf::IndexRange<int,2>& box) {
        for (int i = box.dim(0).begin(); i < box.dim(0).end(); i += box.dim(0).step_size()) {
          visited[i].fetch_add(1, std::memory_order_relaxed);
          for (int j = box.dim(1).begin(); j < box.dim(1).end(); j += box.dim(1).step_size()) {
            REQUIRE(false);  // dim1 is zero-size — this body must never execute
          }
        }
      }, P(c)
    );
    executor.run(taskflow).wait();
    for (int i = 0; i < N; i++) REQUIRE(visited[i] == 1);
  }

  // ── 3D: middle dim zero → size()=4, active dim0 only ────────────────────
  // 4 x 0 x 6: every i in [0,4) visited exactly once.
  // The j-loop must never execute; k-loop would run but is never reached.
  for (size_t c : {size_t{0}, size_t{1}, size_t{3}}) {
    tf::IndexRange<int, 3> range(
      tf::IndexRange<int>(0, 4, 1),
      tf::IndexRange<int>(0, 0, 1),
      tf::IndexRange<int>(0, 6, 1)
    );
    const int N = 4;
    REQUIRE(range.size() == static_cast<size_t>(N));
    std::vector<std::atomic<int>> visited(N);
    for (auto& v : visited) v.store(0);
    taskflow.clear();
    taskflow.for_each_by_index(range,
      [&](const tf::IndexRange<int,3>& box) {
        for (int i = box.dim(0).begin(); i < box.dim(0).end(); i += box.dim(0).step_size()) {
          visited[i].fetch_add(1, std::memory_order_relaxed);
          for (int j = box.dim(1).begin(); j < box.dim(1).end(); j += box.dim(1).step_size()) {
            REQUIRE(false);  // dim1 is zero-size — this body must never execute
            for (int k = box.dim(2).begin(); k < box.dim(2).end(); k += box.dim(2).step_size()) {
              REQUIRE(false);  // unreachable
            }
          }
        }
      }, P(c)
    );
    executor.run(taskflow).wait();
    for (int i = 0; i < N; i++) REQUIRE(visited[i] == 1);
  }

  // ── 3D: innermost dim zero → size()=15, active dim0 x dim1 ─────────────
  // 3 x 5 x 0: every (i,j) pair visited exactly once. Flat index = i*5+j.
  // The k-loop must never execute — verified by REQUIRE(false) inside.
  for (size_t c : {size_t{0}, size_t{1}, size_t{5}}) {
    tf::IndexRange<int, 3> range(
      tf::IndexRange<int>(0, 3, 1),
      tf::IndexRange<int>(0, 5, 1),
      tf::IndexRange<int>(0, 0, 1)
    );
    const int N = 15;
    REQUIRE(range.size() == static_cast<size_t>(N));
    std::vector<std::atomic<int>> visited(N);
    for (auto& v : visited) v.store(0);
    taskflow.clear();
    taskflow.for_each_by_index(range,
      [&](const tf::IndexRange<int,3>& box) {
        for (int i = box.dim(0).begin(); i < box.dim(0).end(); i += box.dim(0).step_size()) {
          for (int j = box.dim(1).begin(); j < box.dim(1).end(); j += box.dim(1).step_size()) {
            visited[i*5+j].fetch_add(1, std::memory_order_relaxed);
            for (int k = box.dim(2).begin(); k < box.dim(2).end(); k += box.dim(2).step_size()) {
              REQUIRE(false);  // dim2 is zero-size — this body must never execute
            }
          }
        }
      }, P(c)
    );
    executor.run(taskflow).wait();
    for (int i = 0; i < N; i++) REQUIRE(visited[i] == 1);
  }

  // ── 5D: zero at d=2 → size()=12, active dim0 x dim1 ────────────────────
  // 3 x 4 x 0 x 5 x 6: every (i,j) visited once. Flat = i*4+j.
  // The dim2 loop must never execute; dim3 and dim4 are unreachable.
  for (size_t c : {size_t{0}, size_t{1}, size_t{4}}) {
    tf::IndexRange<int, 5> range(
      tf::IndexRange<int>(0, 3, 1),
      tf::IndexRange<int>(0, 4, 1),
      tf::IndexRange<int>(0, 0, 1),
      tf::IndexRange<int>(0, 5, 1),
      tf::IndexRange<int>(0, 6, 1)
    );
    const int N = 12;
    REQUIRE(range.size() == static_cast<size_t>(N));
    std::vector<std::atomic<int>> visited(N);
    for (auto& v : visited) v.store(0);
    taskflow.clear();
    taskflow.for_each_by_index(range,
      [&](const tf::IndexRange<int,5>& box) {
        for (int i = box.dim(0).begin(); i < box.dim(0).end(); i += box.dim(0).step_size()) {
          for (int j = box.dim(1).begin(); j < box.dim(1).end(); j += box.dim(1).step_size()) {
            visited[i*4+j].fetch_add(1, std::memory_order_relaxed);
            for (int k = box.dim(2).begin(); k < box.dim(2).end(); k += box.dim(2).step_size()) {
              REQUIRE(false);  // dim2 is zero-size — this body must never execute
              for (int l = box.dim(3).begin(); l < box.dim(3).end(); l += box.dim(3).step_size()) {
                REQUIRE(false);  // unreachable
                for (int m = box.dim(4).begin(); m < box.dim(4).end(); m += box.dim(4).step_size()) {
                  REQUIRE(false);  // unreachable
                }
              }
            }
          }
        }
      }, P(c)
    );
    executor.run(taskflow).wait();
    for (int i = 0; i < N; i++) REQUIRE(visited[i] == 1);
  }

  // ── 5D: outermost dim zero → size()=0, callback never fires ─────────────
  {
    tf::IndexRange<int, 5> range(
      tf::IndexRange<int>(0, 0, 1),
      tf::IndexRange<int>(0, 3, 1),
      tf::IndexRange<int>(0, 4, 1),
      tf::IndexRange<int>(0, 5, 1),
      tf::IndexRange<int>(0, 6, 1)
    );
    REQUIRE(range.size() == 0);
    std::atomic<int> total{0};
    taskflow.clear();
    taskflow.for_each_by_index(range,
      [&](const tf::IndexRange<int,5>&) { total.fetch_add(1, std::memory_order_relaxed); },
      P()
    );
    executor.run(taskflow).wait();
    REQUIRE(total == 0);
  }

  // ── 9D: zero at d=4 → size()=16, active dim0..dim3 ──────────────────────
  // 2x2x2x2x0x2x2x2x2: every (a,b,c,d) visited once. Flat = a*8+b*4+c*2+d.
  // The dim4 loop must never execute; dims 5..8 are unreachable.
  for (size_t cs : {size_t{0}, size_t{1}, size_t{4}}) {
    tf::IndexRange<int, 9> range(
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,0,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1)
    );
    const int N = 16;
    REQUIRE(range.size() == static_cast<size_t>(N));
    std::vector<std::atomic<int>> visited(N);
    for (auto& v : visited) v.store(0);
    taskflow.clear();
    taskflow.for_each_by_index(range,
      [&](const tf::IndexRange<int,9>& box) {
        for (int a = box.dim(0).begin(); a < box.dim(0).end(); ++a)
          for (int b = box.dim(1).begin(); b < box.dim(1).end(); ++b)
            for (int c = box.dim(2).begin(); c < box.dim(2).end(); ++c)
              for (int d = box.dim(3).begin(); d < box.dim(3).end(); ++d) {
                visited[a*8+b*4+c*2+d].fetch_add(1, std::memory_order_relaxed);
                for (int e = box.dim(4).begin(); e < box.dim(4).end(); ++e) {
                  REQUIRE(false);  // dim4 is zero-size — must never execute
                }
              }
      }, P(cs)
    );
    executor.run(taskflow).wait();
    for (int i = 0; i < N; i++) REQUIRE(visited[i] == 1);
  }

  // ── 11D: multiple zeros (d=3 and d=7) → size()=8, active dim0..dim2 ─────
  // 2x2x2x0x3x4x5x0x6x7x8: every (i,j,k) visited once. Flat = i*4+j*2+k.
  // First zero at d=3 — the dim3 loop must never execute.
  // (Second zero at d=7 is unreachable since dim3 already blocks execution.)
  for (size_t cs : {size_t{0}, size_t{1}, size_t{4}}) {
    tf::IndexRange<int, 11> range(
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,0,1),
      tf::IndexRange<int>(0,3,1), tf::IndexRange<int>(0,4,1),
      tf::IndexRange<int>(0,5,1), tf::IndexRange<int>(0,0,1),
      tf::IndexRange<int>(0,6,1), tf::IndexRange<int>(0,7,1),
      tf::IndexRange<int>(0,8,1)
    );
    const int N = 8;
    REQUIRE(range.size() == static_cast<size_t>(N));
    std::vector<std::atomic<int>> visited(N);
    for (auto& v : visited) v.store(0);
    taskflow.clear();
    taskflow.for_each_by_index(range,
      [&](const tf::IndexRange<int,11>& box) {
        for (int i = box.dim(0).begin(); i < box.dim(0).end(); ++i)
          for (int j = box.dim(1).begin(); j < box.dim(1).end(); ++j)
            for (int k = box.dim(2).begin(); k < box.dim(2).end(); ++k) {
              visited[i*4+j*2+k].fetch_add(1, std::memory_order_relaxed);
              for (int l = box.dim(3).begin(); l < box.dim(3).end(); ++l) {
                REQUIRE(false);  // dim3 is zero-size — must never execute
              }
            }
      }, P(cs)
    );
    executor.run(taskflow).wait();
    for (int i = 0; i < N; i++) REQUIRE(visited[i] == 1);
  }
}

// ---- ZERO-DIM TEST CASES ----------------------------------------------------

TEST_CASE("MDForEachByIndex.ZeroDim.Guided.1thread" * doctest::timeout(300)) {
  md_for_each_by_index_zero_dim<tf::GuidedPartitioner<>>(1);
}
TEST_CASE("MDForEachByIndex.ZeroDim.Guided.2threads" * doctest::timeout(300)) {
  md_for_each_by_index_zero_dim<tf::GuidedPartitioner<>>(2);
}
TEST_CASE("MDForEachByIndex.ZeroDim.Guided.4threads" * doctest::timeout(300)) {
  md_for_each_by_index_zero_dim<tf::GuidedPartitioner<>>(4);
}
TEST_CASE("MDForEachByIndex.ZeroDim.Guided.8threads" * doctest::timeout(300)) {
  md_for_each_by_index_zero_dim<tf::GuidedPartitioner<>>(8);
}

TEST_CASE("MDForEachByIndex.ZeroDim.Dynamic.1thread" * doctest::timeout(300)) {
  md_for_each_by_index_zero_dim<tf::DynamicPartitioner<>>(1);
}
TEST_CASE("MDForEachByIndex.ZeroDim.Dynamic.2threads" * doctest::timeout(300)) {
  md_for_each_by_index_zero_dim<tf::DynamicPartitioner<>>(2);
}
TEST_CASE("MDForEachByIndex.ZeroDim.Dynamic.4threads" * doctest::timeout(300)) {
  md_for_each_by_index_zero_dim<tf::DynamicPartitioner<>>(4);
}
TEST_CASE("MDForEachByIndex.ZeroDim.Dynamic.8threads" * doctest::timeout(300)) {
  md_for_each_by_index_zero_dim<tf::DynamicPartitioner<>>(8);
}

TEST_CASE("MDForEachByIndex.ZeroDim.Static.1thread" * doctest::timeout(300)) {
  md_for_each_by_index_zero_dim<tf::StaticPartitioner<>>(1);
}
TEST_CASE("MDForEachByIndex.ZeroDim.Static.2threads" * doctest::timeout(300)) {
  md_for_each_by_index_zero_dim<tf::StaticPartitioner<>>(2);
}
TEST_CASE("MDForEachByIndex.ZeroDim.Static.4threads" * doctest::timeout(300)) {
  md_for_each_by_index_zero_dim<tf::StaticPartitioner<>>(4);
}
TEST_CASE("MDForEachByIndex.ZeroDim.Static.8threads" * doctest::timeout(300)) {
  md_for_each_by_index_zero_dim<tf::StaticPartitioner<>>(8);
}

TEST_CASE("MDForEachByIndex.ZeroDim.Random.1thread" * doctest::timeout(300)) {
  md_for_each_by_index_zero_dim<tf::RandomPartitioner<>>(1);
}
TEST_CASE("MDForEachByIndex.ZeroDim.Random.2threads" * doctest::timeout(300)) {
  md_for_each_by_index_zero_dim<tf::RandomPartitioner<>>(2);
}
TEST_CASE("MDForEachByIndex.ZeroDim.Random.4threads" * doctest::timeout(300)) {
  md_for_each_by_index_zero_dim<tf::RandomPartitioner<>>(4);
}
TEST_CASE("MDForEachByIndex.ZeroDim.Random.8threads" * doctest::timeout(300)) {
  md_for_each_by_index_zero_dim<tf::RandomPartitioner<>>(8);
}
