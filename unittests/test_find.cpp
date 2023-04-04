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

      taskflow.find_if(input.begin(), input.end(), P1, itr1, P(c));
      
      auto init2 = taskflow.emplace([&](){
        beg2 = input.begin();
        end2 = input.end();
      });

      auto find2 = taskflow.find_if(
        std::ref(beg2), std::ref(end2), P2, itr2, P(c)
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
  test_find_if<tf::StaticPartitioner>(1);
}

TEST_CASE("find_if.StaticPartitioner.2threads" * doctest::timeout(300)) {
  test_find_if<tf::StaticPartitioner>(2);
}

TEST_CASE("find_if.StaticPartitioner.3threads" * doctest::timeout(300)) {
  test_find_if<tf::StaticPartitioner>(3);
}

TEST_CASE("find_if.StaticPartitioner.4threads" * doctest::timeout(300)) {
  test_find_if<tf::StaticPartitioner>(4);
}

TEST_CASE("find_if.StaticPartitioner.5threads" * doctest::timeout(300)) {
  test_find_if<tf::StaticPartitioner>(5);
}

TEST_CASE("find_if.StaticPartitioner.6threads" * doctest::timeout(300)) {
  test_find_if<tf::StaticPartitioner>(6);
}

TEST_CASE("find_if.StaticPartitioner.7threads" * doctest::timeout(300)) {
  test_find_if<tf::StaticPartitioner>(7);
}

TEST_CASE("find_if.StaticPartitioner.8threads" * doctest::timeout(300)) {
  test_find_if<tf::StaticPartitioner>(8);
}

// guided partitioner
TEST_CASE("find_if.GuidedPartitioner.1thread" * doctest::timeout(300)) {
  test_find_if<tf::GuidedPartitioner>(1);
}

TEST_CASE("find_if.GuidedPartitioner.2threads" * doctest::timeout(300)) {
  test_find_if<tf::GuidedPartitioner>(2);
}

TEST_CASE("find_if.GuidedPartitioner.3threads" * doctest::timeout(300)) {
  test_find_if<tf::GuidedPartitioner>(3);
}

TEST_CASE("find_if.GuidedPartitioner.4threads" * doctest::timeout(300)) {
  test_find_if<tf::GuidedPartitioner>(4);
}

TEST_CASE("find_if.GuidedPartitioner.5threads" * doctest::timeout(300)) {
  test_find_if<tf::GuidedPartitioner>(5);
}

TEST_CASE("find_if.GuidedPartitioner.6threads" * doctest::timeout(300)) {
  test_find_if<tf::GuidedPartitioner>(6);
}

TEST_CASE("find_if.GuidedPartitioner.7threads" * doctest::timeout(300)) {
  test_find_if<tf::GuidedPartitioner>(7);
}

TEST_CASE("find_if.GuidedPartitioner.8threads" * doctest::timeout(300)) {
  test_find_if<tf::GuidedPartitioner>(8);
}

// dynamic partitioner
TEST_CASE("find_if.DynamicPartitioner.1thread" * doctest::timeout(300)) {
  test_find_if<tf::DynamicPartitioner>(1);
}

TEST_CASE("find_if.DynamicPartitioner.2threads" * doctest::timeout(300)) {
  test_find_if<tf::DynamicPartitioner>(2);
}

TEST_CASE("find_if.DynamicPartitioner.3threads" * doctest::timeout(300)) {
  test_find_if<tf::DynamicPartitioner>(3);
}

TEST_CASE("find_if.DynamicPartitioner.4threads" * doctest::timeout(300)) {
  test_find_if<tf::DynamicPartitioner>(4);
}

TEST_CASE("find_if.DynamicPartitioner.5threads" * doctest::timeout(300)) {
  test_find_if<tf::DynamicPartitioner>(5);
}

TEST_CASE("find_if.DynamicPartitioner.6threads" * doctest::timeout(300)) {
  test_find_if<tf::DynamicPartitioner>(6);
}

TEST_CASE("find_if.DynamicPartitioner.7threads" * doctest::timeout(300)) {
  test_find_if<tf::DynamicPartitioner>(7);
}

TEST_CASE("find_if.DynamicPartitioner.8threads" * doctest::timeout(300)) {
  test_find_if<tf::DynamicPartitioner>(8);
}
