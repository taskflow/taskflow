#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/scan.hpp>

// --------------------------------------------------------
// Testcase: inclusive_scan
// --------------------------------------------------------

template <typename T>
void test_inclusive_scan(unsigned W) {
  
  tf::Executor executor(W);
  tf::Taskflow taskflow;

  size_t limit = std::is_same_v<T, std::string> ? 2500 : 250000;
  
  for(size_t n=0; n<=limit; n=n*2+1) {

    taskflow.clear();
    
    std::vector<T> input(n), output(n), golden(n); 
    typename std::vector<T>::iterator sbeg, send, dbeg;

    for(size_t i=0; i<n; i++) {
      if constexpr(std::is_same_v<T, std::string>) {
        input[i] = std::to_string(::rand() % 10);
      }
      else {
        input[i] = ::rand() % 10;
      }
    }

    std::inclusive_scan(
      input.begin(), input.end(), golden.begin(), std::plus<T>{}
    );
    
    // out-of-place
    auto task1 = taskflow.inclusive_scan(
      input.begin(), input.end(), output.begin(), std::plus<T>{}
    );  
    
    // enable stateful capture
    auto alloc = taskflow.emplace([&](){
      sbeg = input.begin();
      send = input.end();
      dbeg = input.begin();
    });
    
    // in-place
    auto task2 = taskflow.inclusive_scan(
      std::ref(sbeg), std::ref(send), std::ref(dbeg), std::plus<T>{}
    );

    task1.precede(alloc);
    alloc.precede(task2);

    executor.run(taskflow).wait();

    REQUIRE(input == golden);
    REQUIRE(output == golden);
  }

}

// int data type
TEST_CASE("InclusiveScan.int.1thread" * doctest::timeout(300)) {
  test_inclusive_scan<int>(1);
}

TEST_CASE("InclusiveScan.int.2threads" * doctest::timeout(300)) {
  test_inclusive_scan<int>(2);
}

TEST_CASE("InclusiveScan.int.3threads" * doctest::timeout(300)) {
  test_inclusive_scan<int>(3);
}

TEST_CASE("InclusiveScan.int.4threads" * doctest::timeout(300)) {
  test_inclusive_scan<int>(4);
}

TEST_CASE("InclusiveScan.int.8threads" * doctest::timeout(300)) {
  test_inclusive_scan<int>(8);
}

TEST_CASE("InclusiveScan.int.12threads" * doctest::timeout(300)) {
  test_inclusive_scan<int>(12);
}

// string data type
TEST_CASE("InclusiveScan.string.1thread" * doctest::timeout(300)) {
  test_inclusive_scan<std::string>(1);
}

TEST_CASE("InclusiveScan.string.2threads" * doctest::timeout(300)) {
  test_inclusive_scan<std::string>(2);
}

TEST_CASE("InclusiveScan.string.3threads" * doctest::timeout(300)) {
  test_inclusive_scan<std::string>(3);
}

TEST_CASE("InclusiveScan.string.4threads" * doctest::timeout(300)) {
  test_inclusive_scan<std::string>(4);
}

TEST_CASE("InclusiveScan.string.8threads" * doctest::timeout(300)) {
  test_inclusive_scan<std::string>(8);
}

TEST_CASE("InclusiveScan.string.12threads" * doctest::timeout(300)) {
  test_inclusive_scan<std::string>(12);
}

