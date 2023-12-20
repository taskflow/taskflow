#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/find.hpp>

template <typename P>
void test_find_if(unsigned W) {
  
  tf::Executor executor(W);
  tf::Taskflow taskflow;
  std::vector<int> input;
  
  for(size_t n = 0; n <= 65536; n <= 256 ? n++ : n=2*n+1) {
    for(size_t c : {0, 1, 3, 7, 99}) {

      taskflow.clear();

      input.resize(n);

      for(auto& i : input) {
        i = ::rand() % (2 * n) + 1;
      }

      auto P1 = [] (int i) { return i == 5; };
      auto P2 = [] (int i) { return i == 0; };

      auto res1 = std::find_if(input.begin(), input.end(), P1);
      auto res2 = std::find_if(input.begin(), input.end(), P2);
      
      REQUIRE(res2 == input.end());

      std::vector<int>::iterator itr1, itr2, beg2, end2;

      taskflow.find_if(input.begin(), input.end(), itr1, P1, P(c));
      
      auto init2 = taskflow.emplace([&](){
        beg2 = input.begin();
        end2 = input.end();
      });

      auto find2 = taskflow.find_if(
        std::ref(beg2), std::ref(end2), itr2, P2, P(c)
      );

      init2.precede(find2);

      executor.run(taskflow).wait();
      
      REQUIRE(itr1 == res1);
      REQUIRE(itr2 == res2);
    }
  }
}

// static partitioner
TEST_CASE("find_if.StaticPartitioner.1thread" * doctest::timeout(300)) {
  test_find_if<tf::StaticPartitioner<>>(1);
}

TEST_CASE("find_if.StaticPartitioner.2threads" * doctest::timeout(300)) {
  test_find_if<tf::StaticPartitioner<>>(2);
}

TEST_CASE("find_if.StaticPartitioner.3threads" * doctest::timeout(300)) {
  test_find_if<tf::StaticPartitioner<>>(3);
}

TEST_CASE("find_if.StaticPartitioner.4threads" * doctest::timeout(300)) {
  test_find_if<tf::StaticPartitioner<>>(4);
}

TEST_CASE("find_if.StaticPartitioner.5threads" * doctest::timeout(300)) {
  test_find_if<tf::StaticPartitioner<>>(5);
}

TEST_CASE("find_if.StaticPartitioner.6threads" * doctest::timeout(300)) {
  test_find_if<tf::StaticPartitioner<>>(6);
}

TEST_CASE("find_if.StaticPartitioner.7threads" * doctest::timeout(300)) {
  test_find_if<tf::StaticPartitioner<>>(7);
}

TEST_CASE("find_if.StaticPartitioner.8threads" * doctest::timeout(300)) {
  test_find_if<tf::StaticPartitioner<>>(8);
}

// guided partitioner
TEST_CASE("find_if.GuidedPartitioner.1thread" * doctest::timeout(300)) {
  test_find_if<tf::GuidedPartitioner<>>(1);
}

TEST_CASE("find_if.GuidedPartitioner.2threads" * doctest::timeout(300)) {
  test_find_if<tf::GuidedPartitioner<>>(2);
}

TEST_CASE("find_if.GuidedPartitioner.3threads" * doctest::timeout(300)) {
  test_find_if<tf::GuidedPartitioner<>>(3);
}

TEST_CASE("find_if.GuidedPartitioner.4threads" * doctest::timeout(300)) {
  test_find_if<tf::GuidedPartitioner<>>(4);
}

TEST_CASE("find_if.GuidedPartitioner.5threads" * doctest::timeout(300)) {
  test_find_if<tf::GuidedPartitioner<>>(5);
}

TEST_CASE("find_if.GuidedPartitioner.6threads" * doctest::timeout(300)) {
  test_find_if<tf::GuidedPartitioner<>>(6);
}

TEST_CASE("find_if.GuidedPartitioner.7threads" * doctest::timeout(300)) {
  test_find_if<tf::GuidedPartitioner<>>(7);
}

TEST_CASE("find_if.GuidedPartitioner.8threads" * doctest::timeout(300)) {
  test_find_if<tf::GuidedPartitioner<>>(8);
}

// dynamic partitioner
TEST_CASE("find_if.DynamicPartitioner.1thread" * doctest::timeout(300)) {
  test_find_if<tf::DynamicPartitioner<>>(1);
}

TEST_CASE("find_if.DynamicPartitioner.2threads" * doctest::timeout(300)) {
  test_find_if<tf::DynamicPartitioner<>>(2);
}

TEST_CASE("find_if.DynamicPartitioner.3threads" * doctest::timeout(300)) {
  test_find_if<tf::DynamicPartitioner<>>(3);
}

TEST_CASE("find_if.DynamicPartitioner.4threads" * doctest::timeout(300)) {
  test_find_if<tf::DynamicPartitioner<>>(4);
}

TEST_CASE("find_if.DynamicPartitioner.5threads" * doctest::timeout(300)) {
  test_find_if<tf::DynamicPartitioner<>>(5);
}

TEST_CASE("find_if.DynamicPartitioner.6threads" * doctest::timeout(300)) {
  test_find_if<tf::DynamicPartitioner<>>(6);
}

TEST_CASE("find_if.DynamicPartitioner.7threads" * doctest::timeout(300)) {
  test_find_if<tf::DynamicPartitioner<>>(7);
}

TEST_CASE("find_if.DynamicPartitioner.8threads" * doctest::timeout(300)) {
  test_find_if<tf::DynamicPartitioner<>>(8);
}

// ----------------------------------------------------------------------------
// find_if_not
// ----------------------------------------------------------------------------

template <typename P>
void test_find_if_not(unsigned W) {
  
  tf::Executor executor(W);
  tf::Taskflow taskflow;
  std::vector<int> input;
  
  for(size_t n = 0; n <= 65536; n <= 256 ? n++ : n=2*n+1) {
    for(size_t c : {0, 1, 3, 7, 99}) {

      taskflow.clear();

      input.resize(n);

      for(auto& i : input) {
        if(rand()% (n+1) > 0) {
          i = 5;
        }
        else {
          i = 0;
        }
      }

      auto P1 = [] (int i) { return i == 5; };
      auto P2 = [] (int i) { return i == 0; };

      auto res1 = std::find_if_not(input.begin(), input.end(), P1);
      auto res2 = std::find_if_not(input.begin(), input.end(), P2);
      
      std::vector<int>::iterator itr1, itr2, beg2, end2;

      taskflow.find_if_not(input.begin(), input.end(), itr1, P1, P(c));
      
      auto init2 = taskflow.emplace([&](){
        beg2 = input.begin();
        end2 = input.end();
      });

      auto find2 = taskflow.find_if_not(
        std::ref(beg2), std::ref(end2), itr2, P2, P(c)
      );

      init2.precede(find2);

      executor.run(taskflow).wait();
      
      REQUIRE(itr1 == res1);
      REQUIRE(itr2 == res2);
    }
  }
}

// static partitioner
TEST_CASE("find_if_not.StaticPartitioner.1thread" * doctest::timeout(300)) {
  test_find_if_not<tf::StaticPartitioner<>>(1);
}

TEST_CASE("find_if_not.StaticPartitioner.2threads" * doctest::timeout(300)) {
  test_find_if_not<tf::StaticPartitioner<>>(2);
}

TEST_CASE("find_if_not.StaticPartitioner.3threads" * doctest::timeout(300)) {
  test_find_if_not<tf::StaticPartitioner<>>(3);
}

TEST_CASE("find_if_not.StaticPartitioner.4threads" * doctest::timeout(300)) {
  test_find_if_not<tf::StaticPartitioner<>>(4);
}

TEST_CASE("find_if_not.StaticPartitioner.5threads" * doctest::timeout(300)) {
  test_find_if_not<tf::StaticPartitioner<>>(5);
}

TEST_CASE("find_if_not.StaticPartitioner.6threads" * doctest::timeout(300)) {
  test_find_if_not<tf::StaticPartitioner<>>(6);
}

TEST_CASE("find_if_not.StaticPartitioner.7threads" * doctest::timeout(300)) {
  test_find_if_not<tf::StaticPartitioner<>>(7);
}

TEST_CASE("find_if_not.StaticPartitioner.8threads" * doctest::timeout(300)) {
  test_find_if_not<tf::StaticPartitioner<>>(8);
}

// guided partitioner
TEST_CASE("find_if_not.GuidedPartitioner.1thread" * doctest::timeout(300)) {
  test_find_if_not<tf::GuidedPartitioner<>>(1);
}

TEST_CASE("find_if_not.GuidedPartitioner.2threads" * doctest::timeout(300)) {
  test_find_if_not<tf::GuidedPartitioner<>>(2);
}

TEST_CASE("find_if_not.GuidedPartitioner.3threads" * doctest::timeout(300)) {
  test_find_if_not<tf::GuidedPartitioner<>>(3);
}

TEST_CASE("find_if_not.GuidedPartitioner.4threads" * doctest::timeout(300)) {
  test_find_if_not<tf::GuidedPartitioner<>>(4);
}

TEST_CASE("find_if_not.GuidedPartitioner.5threads" * doctest::timeout(300)) {
  test_find_if_not<tf::GuidedPartitioner<>>(5);
}

TEST_CASE("find_if_not.GuidedPartitioner.6threads" * doctest::timeout(300)) {
  test_find_if_not<tf::GuidedPartitioner<>>(6);
}

TEST_CASE("find_if_not.GuidedPartitioner.7threads" * doctest::timeout(300)) {
  test_find_if_not<tf::GuidedPartitioner<>>(7);
}

TEST_CASE("find_if_not.GuidedPartitioner.8threads" * doctest::timeout(300)) {
  test_find_if_not<tf::GuidedPartitioner<>>(8);
}

// dynamic partitioner
TEST_CASE("find_if_not.DynamicPartitioner.1thread" * doctest::timeout(300)) {
  test_find_if_not<tf::DynamicPartitioner<>>(1);
}

TEST_CASE("find_if_not.DynamicPartitioner.2threads" * doctest::timeout(300)) {
  test_find_if_not<tf::DynamicPartitioner<>>(2);
}

TEST_CASE("find_if_not.DynamicPartitioner.3threads" * doctest::timeout(300)) {
  test_find_if_not<tf::DynamicPartitioner<>>(3);
}

TEST_CASE("find_if_not.DynamicPartitioner.4threads" * doctest::timeout(300)) {
  test_find_if_not<tf::DynamicPartitioner<>>(4);
}

TEST_CASE("find_if_not.DynamicPartitioner.5threads" * doctest::timeout(300)) {
  test_find_if_not<tf::DynamicPartitioner<>>(5);
}

TEST_CASE("find_if_not.DynamicPartitioner.6threads" * doctest::timeout(300)) {
  test_find_if_not<tf::DynamicPartitioner<>>(6);
}

TEST_CASE("find_if_not.DynamicPartitioner.7threads" * doctest::timeout(300)) {
  test_find_if_not<tf::DynamicPartitioner<>>(7);
}

TEST_CASE("find_if_not.DynamicPartitioner.8threads" * doctest::timeout(300)) {
  test_find_if_not<tf::DynamicPartitioner<>>(8);
}

// ----------------------------------------------------------------------------
// min_element
// ----------------------------------------------------------------------------

template <typename P>
void test_min_element(unsigned W) {
  
  tf::Executor executor(W);
  tf::Taskflow taskflow;
  std::vector<int> input;
  
  for(size_t n = 0; n <= 65536; n <= 256 ? n++ : n=2*n+1) {
    for(size_t c : {0, 1, 3, 7, 99}) {

      taskflow.clear();

      input.resize(n);

      int min_element = std::numeric_limits<int>::max();

      for(auto& i : input) {
        i = ::rand() % (2 * n) + 1;
        min_element = std::min(min_element, i);
      }

      auto min_g = std::min_element(input.begin(), input.end(), std::less<int>{});
      
      if(n != 0) {
        REQUIRE(*min_g == min_element);
      }

      std::vector<int>::iterator res_1, res_2, beg, end;

      taskflow.min_element(input.begin(), input.end(), res_1, std::less<int>{}, P(c));
      
      auto init2 = taskflow.emplace([&](){
        beg = input.begin();
        end = input.end();
      });

      auto find2 = taskflow.min_element(
        std::ref(beg), std::ref(end), res_2, std::less<int>{}, P(c)
      );

      init2.precede(find2);

      executor.run(taskflow).wait();
      
      if(n == 0) {
        REQUIRE(res_1 == input.end());
        REQUIRE(res_2 == input.end());
      }
      else {
        REQUIRE(*res_1 == min_element);
        REQUIRE(*res_2 == min_element);
      }
    }
  }
}

// static partitioner
TEST_CASE("min_element.StaticPartitioner.1thread" * doctest::timeout(300)) {
  test_min_element<tf::StaticPartitioner<>>(1);
}

TEST_CASE("min_element.StaticPartitioner.2threads" * doctest::timeout(300)) {
  test_min_element<tf::StaticPartitioner<>>(2);
}

TEST_CASE("min_element.StaticPartitioner.3threads" * doctest::timeout(300)) {
  test_min_element<tf::StaticPartitioner<>>(3);
}

TEST_CASE("min_element.StaticPartitioner.4threads" * doctest::timeout(300)) {
  test_min_element<tf::StaticPartitioner<>>(4);
}

TEST_CASE("min_element.StaticPartitioner.5threads" * doctest::timeout(300)) {
  test_min_element<tf::StaticPartitioner<>>(5);
}

TEST_CASE("min_element.StaticPartitioner.6threads" * doctest::timeout(300)) {
  test_min_element<tf::StaticPartitioner<>>(6);
}

TEST_CASE("min_element.StaticPartitioner.7threads" * doctest::timeout(300)) {
  test_min_element<tf::StaticPartitioner<>>(7);
}

TEST_CASE("min_element.StaticPartitioner.8threads" * doctest::timeout(300)) {
  test_min_element<tf::StaticPartitioner<>>(8);
}

// guided partitioner
TEST_CASE("min_element.GuidedPartitioner.1thread" * doctest::timeout(300)) {
  test_min_element<tf::GuidedPartitioner<>>(1);
}

TEST_CASE("min_element.GuidedPartitioner.2threads" * doctest::timeout(300)) {
  test_min_element<tf::GuidedPartitioner<>>(2);
}

TEST_CASE("min_element.GuidedPartitioner.3threads" * doctest::timeout(300)) {
  test_min_element<tf::GuidedPartitioner<>>(3);
}

TEST_CASE("min_element.GuidedPartitioner.4threads" * doctest::timeout(300)) {
  test_min_element<tf::GuidedPartitioner<>>(4);
}

TEST_CASE("min_element.GuidedPartitioner.5threads" * doctest::timeout(300)) {
  test_min_element<tf::GuidedPartitioner<>>(5);
}

TEST_CASE("min_element.GuidedPartitioner.6threads" * doctest::timeout(300)) {
  test_min_element<tf::GuidedPartitioner<>>(6);
}

TEST_CASE("min_element.GuidedPartitioner.7threads" * doctest::timeout(300)) {
  test_min_element<tf::GuidedPartitioner<>>(7);
}

TEST_CASE("min_element.GuidedPartitioner.8threads" * doctest::timeout(300)) {
  test_min_element<tf::GuidedPartitioner<>>(8);
}

// dynamic partitioner
TEST_CASE("min_element.DynamicPartitioner.1thread" * doctest::timeout(300)) {
  test_min_element<tf::DynamicPartitioner<>>(1);
}

TEST_CASE("min_element.DynamicPartitioner.2threads" * doctest::timeout(300)) {
  test_min_element<tf::DynamicPartitioner<>>(2);
}

TEST_CASE("min_element.DynamicPartitioner.3threads" * doctest::timeout(300)) {
  test_min_element<tf::DynamicPartitioner<>>(3);
}

TEST_CASE("min_element.DynamicPartitioner.4threads" * doctest::timeout(300)) {
  test_min_element<tf::DynamicPartitioner<>>(4);
}

TEST_CASE("min_element.DynamicPartitioner.5threads" * doctest::timeout(300)) {
  test_min_element<tf::DynamicPartitioner<>>(5);
}

TEST_CASE("min_element.DynamicPartitioner.6threads" * doctest::timeout(300)) {
  test_min_element<tf::DynamicPartitioner<>>(6);
}

TEST_CASE("min_element.DynamicPartitioner.7threads" * doctest::timeout(300)) {
  test_min_element<tf::DynamicPartitioner<>>(7);
}

TEST_CASE("min_element.DynamicPartitioner.8threads" * doctest::timeout(300)) {
  test_min_element<tf::DynamicPartitioner<>>(8);
}

// ----------------------------------------------------------------------------
// max_element
// ----------------------------------------------------------------------------

template <typename P>
void test_max_element(unsigned W) {
  
  tf::Executor executor(W);
  tf::Taskflow taskflow;
  std::vector<int> input;
  
  for(size_t n = 0; n <= 65536; n <= 256 ? n++ : n=2*n+1) {
    for(size_t c : {0, 1, 3, 7, 99}) {

      taskflow.clear();

      input.resize(n);

      int max_element = std::numeric_limits<int>::min();

      for(auto& i : input) {
        i = ::rand() % (2 * n) + 1;
        max_element = std::max(max_element, i);
      }

      auto max_g = std::max_element(input.begin(), input.end(), std::less<int>{});
      
      if(n != 0) {
        REQUIRE(*max_g == max_element);
      }

      std::vector<int>::iterator res_1, res_2, beg, end;

      taskflow.max_element(input.begin(), input.end(), res_1, std::less<int>{}, P(c));
      
      auto init2 = taskflow.emplace([&](){
        beg = input.begin();
        end = input.end();
      });

      auto find2 = taskflow.max_element(
        std::ref(beg), std::ref(end), res_2, std::less<int>{}, P(c)
      );

      init2.precede(find2);

      executor.run(taskflow).wait();
      
      if(n == 0) {
        REQUIRE(res_1 == input.end());
        REQUIRE(res_2 == input.end());
      }
      else {
        REQUIRE(*res_1 == max_element);
        REQUIRE(*res_2 == max_element);
      }
    }
  }
}

// static partitioner
TEST_CASE("max_element.StaticPartitioner.1thread" * doctest::timeout(300)) {
  test_max_element<tf::StaticPartitioner<>>(1);
}

TEST_CASE("max_element.StaticPartitioner.2threads" * doctest::timeout(300)) {
  test_max_element<tf::StaticPartitioner<>>(2);
}

TEST_CASE("max_element.StaticPartitioner.3threads" * doctest::timeout(300)) {
  test_max_element<tf::StaticPartitioner<>>(3);
}

TEST_CASE("max_element.StaticPartitioner.4threads" * doctest::timeout(300)) {
  test_max_element<tf::StaticPartitioner<>>(4);
}

TEST_CASE("max_element.StaticPartitioner.5threads" * doctest::timeout(300)) {
  test_max_element<tf::StaticPartitioner<>>(5);
}

TEST_CASE("max_element.StaticPartitioner.6threads" * doctest::timeout(300)) {
  test_max_element<tf::StaticPartitioner<>>(6);
}

TEST_CASE("max_element.StaticPartitioner.7threads" * doctest::timeout(300)) {
  test_max_element<tf::StaticPartitioner<>>(7);
}

TEST_CASE("max_element.StaticPartitioner.8threads" * doctest::timeout(300)) {
  test_max_element<tf::StaticPartitioner<>>(8);
}

// guided partitioner
TEST_CASE("max_element.GuidedPartitioner.1thread" * doctest::timeout(300)) {
  test_max_element<tf::GuidedPartitioner<>>(1);
}

TEST_CASE("max_element.GuidedPartitioner.2threads" * doctest::timeout(300)) {
  test_max_element<tf::GuidedPartitioner<>>(2);
}

TEST_CASE("max_element.GuidedPartitioner.3threads" * doctest::timeout(300)) {
  test_max_element<tf::GuidedPartitioner<>>(3);
}

TEST_CASE("max_element.GuidedPartitioner.4threads" * doctest::timeout(300)) {
  test_max_element<tf::GuidedPartitioner<>>(4);
}

TEST_CASE("max_element.GuidedPartitioner.5threads" * doctest::timeout(300)) {
  test_max_element<tf::GuidedPartitioner<>>(5);
}

TEST_CASE("max_element.GuidedPartitioner.6threads" * doctest::timeout(300)) {
  test_max_element<tf::GuidedPartitioner<>>(6);
}

TEST_CASE("max_element.GuidedPartitioner.7threads" * doctest::timeout(300)) {
  test_max_element<tf::GuidedPartitioner<>>(7);
}

TEST_CASE("max_element.GuidedPartitioner.8threads" * doctest::timeout(300)) {
  test_max_element<tf::GuidedPartitioner<>>(8);
}

// dynamic partitioner
TEST_CASE("max_element.DynamicPartitioner.1thread" * doctest::timeout(300)) {
  test_max_element<tf::DynamicPartitioner<>>(1);
}

TEST_CASE("max_element.DynamicPartitioner.2threads" * doctest::timeout(300)) {
  test_max_element<tf::DynamicPartitioner<>>(2);
}

TEST_CASE("max_element.DynamicPartitioner.3threads" * doctest::timeout(300)) {
  test_max_element<tf::DynamicPartitioner<>>(3);
}

TEST_CASE("max_element.DynamicPartitioner.4threads" * doctest::timeout(300)) {
  test_max_element<tf::DynamicPartitioner<>>(4);
}

TEST_CASE("max_element.DynamicPartitioner.5threads" * doctest::timeout(300)) {
  test_max_element<tf::DynamicPartitioner<>>(5);
}

TEST_CASE("max_element.DynamicPartitioner.6threads" * doctest::timeout(300)) {
  test_max_element<tf::DynamicPartitioner<>>(6);
}

TEST_CASE("max_element.DynamicPartitioner.7threads" * doctest::timeout(300)) {
  test_max_element<tf::DynamicPartitioner<>>(7);
}

TEST_CASE("max_element.DynamicPartitioner.8threads" * doctest::timeout(300)) {
  test_max_element<tf::DynamicPartitioner<>>(8);
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

// write a test for find using static partitioner
TEST_CASE("ClosureWrapper.find.Static" * doctest::timeout(300))
{
  for (int tc = 1; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER);
    std::vector<int>::iterator result;
    std::iota(range.begin(), range.end(), 0);
    taskflow.find_if(range.begin(), range.end(), result, 
      [&](int i) { 
        return i == 500 && GetThreadSpecificContext() == tc; 
      },
      tf::StaticPartitioner(1, [&](auto&& task) {
        wrapper_called_count++;
        GetThreadSpecificContext() = tc;
        task();
        GetThreadSpecificContext() = 0;
      })
    );
    executor.run(taskflow).wait();

    REQUIRE(wrapper_called_count == tc);
    REQUIRE(*result == 500);
  }
}

// Same as above testcase but with dynamic partitioner
TEST_CASE("ClosureWrapper.find.Dynamic" * doctest::timeout(300))
{
  for (int tc = 1; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER);
    std::vector<int>::iterator result;
    std::iota(range.begin(), range.end(), 0);
    taskflow.find_if(range.begin(), range.end(), result, 
      [&](int i) { 
        return i == 500 && GetThreadSpecificContext() == tc; 
      },
      tf::DynamicPartitioner(1, [&](auto&& task) {
        wrapper_called_count++;
        GetThreadSpecificContext() = tc;
        task();
        GetThreadSpecificContext() = 0;
      })
    );
    executor.run(taskflow).wait();

    REQUIRE(wrapper_called_count <= tc);
    REQUIRE(*result == 500);
  }
}

// static partitioner for find_if_not
TEST_CASE("ClosureWrapper.find_if_not.Static" * doctest::timeout(300))
{
  for (int tc = 1; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER);
    std::vector<int>::iterator result;
    std::iota(range.begin(), range.end(), 0);
    taskflow.find_if_not(range.begin(), range.end(), result, 
      [&](int i) { 
        return i != 500 && GetThreadSpecificContext() == tc; 
      },
      tf::StaticPartitioner(1, [&](auto&& task){
        wrapper_called_count++;
        GetThreadSpecificContext() = tc;
        task();
        GetThreadSpecificContext() = 0;
      })
    );
    executor.run(taskflow).wait();

    REQUIRE(wrapper_called_count == tc);
    REQUIRE(*result == 500);
  }
}

// dynamic ic partitioner for find_if_not
TEST_CASE("ClosureWrapper.find_if_not.Dynamic" * doctest::timeout(300))
{
  for (int tc = 1; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER);
    std::vector<int>::iterator result;
    std::iota(range.begin(), range.end(), 0);
    taskflow.find_if_not(range.begin(), range.end(), result, 
      [&](int i) { 
        return i != 500 && GetThreadSpecificContext() == tc; 
      },
      tf::DynamicPartitioner(1, [&](auto&& task){
        wrapper_called_count++;
        GetThreadSpecificContext() = tc;
        task();
        GetThreadSpecificContext() = 0;
      })
    );
    executor.run(taskflow).wait();

    REQUIRE(wrapper_called_count <= tc);
    REQUIRE(*result == 500);
  }
}

// static partitioner for min_element
TEST_CASE("ClosureWrapper.min_element.Static" * doctest::timeout(300)) {
  for (int tc = 1; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER);
    std::vector<int>::iterator result;
    std::iota(range.begin(), range.end(), 0);
    range[500] = -10;
    taskflow.min_element(range.begin(), range.end(), result, 
      [&](int a, int b) { 
        REQUIRE(GetThreadSpecificContext() == tc);
        return a < b;
      },
      tf::StaticPartitioner(1, [&](auto&& task){
        wrapper_called_count++;
        GetThreadSpecificContext() = tc;
        task();
        GetThreadSpecificContext() = 0;
      })
    );
    executor.run(taskflow).wait();

    REQUIRE(wrapper_called_count == tc);
    REQUIRE(*result == -10);
  }
}

// dynamic partitioner for min_element
TEST_CASE("ClosureWrapper.min_element.Dynamic" * doctest::timeout(300)) {
  for (int tc = 1; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER);
    std::vector<int>::iterator result;
    std::iota(range.begin(), range.end(), 0);
    range[500] = -10;
    taskflow.min_element(range.begin(), range.end(), result, 
      [&](int a, int b) { 
        REQUIRE(GetThreadSpecificContext() == tc);
        return a < b;
      },
      tf::DynamicPartitioner(1, [&](auto&& task){
        wrapper_called_count++;
        GetThreadSpecificContext() = tc;
        task();
        GetThreadSpecificContext() = 0;
      })
    );
    executor.run(taskflow).wait();

    REQUIRE(wrapper_called_count <= tc);
    REQUIRE(*result == -10);
  }
}

// static partitioner for max_element
TEST_CASE("ClosureWrapper.max_element.Static" * doctest::timeout(300)) {
  for (int tc = 1; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER);
    std::vector<int>::iterator result;
    std::iota(range.begin(), range.end(), 0);
    range[500] = 1000000;
    taskflow.max_element(range.begin(), range.end(), result, 
      [&](int a, int b) { 
        REQUIRE(GetThreadSpecificContext() == tc);
        return a < b;
      },
      tf::StaticPartitioner(1, [&](auto&& task){
        wrapper_called_count++;
        GetThreadSpecificContext() = tc;
        task();
        GetThreadSpecificContext() = 0;
      })
    );
    executor.run(taskflow).wait();

    REQUIRE(wrapper_called_count == tc);
    REQUIRE(*result == 1000000);
  }
}

// dynamic partitioner for max_element
TEST_CASE("ClosureWrapper.max_element.Dynamic" * doctest::timeout(300)) {
  for (int tc = 1; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER);
    std::vector<int>::iterator result;
    std::iota(range.begin(), range.end(), 0);
    range[500] = 1000000;
    taskflow.max_element(range.begin(), range.end(), result, 
      [&](int a, int b) { 
        REQUIRE(GetThreadSpecificContext() == tc);
        return a < b;
      },
      tf::DynamicPartitioner(1, [&](auto&& task){
        wrapper_called_count++;
        GetThreadSpecificContext() = tc;
        task();
        GetThreadSpecificContext() = 0;
      })
    );
    executor.run(taskflow).wait();

    REQUIRE(wrapper_called_count <= tc);
    REQUIRE(*result == 1000000);
  }
}



