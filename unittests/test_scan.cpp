#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/scan.hpp>

// --------------------------------------------------------
// Testcase: inclusive_scan
// --------------------------------------------------------

template <typename T, typename B>
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
      input.begin(), input.end(), golden.begin(), B()
    );
    
    // out-of-place
    auto task1 = taskflow.inclusive_scan(
      input.begin(), input.end(), output.begin(), B()
    );  
    
    // enable stateful capture
    auto alloc = taskflow.emplace([&](){
      sbeg = input.begin();
      send = input.end();
      dbeg = input.begin();
    });
    
    // in-place
    auto task2 = taskflow.inclusive_scan(
      std::ref(sbeg), std::ref(send), std::ref(dbeg), B()
    );

    task1.precede(alloc);
    alloc.precede(task2);

    executor.run(taskflow).wait();

    REQUIRE(input == golden);
    REQUIRE(output == golden);
  }
}

// int data type
TEST_CASE("InclusiveScan.int+.1thread" * doctest::timeout(300)) {
  test_inclusive_scan<int, std::plus<int>>(1);
}

TEST_CASE("InclusiveScan.int+.2threads" * doctest::timeout(300)) {
  test_inclusive_scan<int, std::plus<int>>(2);
}

TEST_CASE("InclusiveScan.int+.3threads" * doctest::timeout(300)) {
  test_inclusive_scan<int, std::plus<int>>(3);
}

TEST_CASE("InclusiveScan.int+.4threads" * doctest::timeout(300)) {
  test_inclusive_scan<int, std::plus<int>>(4);
}

TEST_CASE("InclusiveScan.int+.8threads" * doctest::timeout(300)) {
  test_inclusive_scan<int, std::plus<int>>(8);
}

TEST_CASE("InclusiveScan.int+.12threads" * doctest::timeout(300)) {
  test_inclusive_scan<int, std::plus<int>>(12);
}

// int data type
TEST_CASE("InclusiveScan.int*.1thread" * doctest::timeout(300)) {
  test_inclusive_scan<int, std::multiplies<int>>(1);
}

TEST_CASE("InclusiveScan.int*.2threads" * doctest::timeout(300)) {
  test_inclusive_scan<int, std::multiplies<int>>(2);
}

TEST_CASE("InclusiveScan.int*.3threads" * doctest::timeout(300)) {
  test_inclusive_scan<int, std::multiplies<int>>(3);
}

TEST_CASE("InclusiveScan.int*.4threads" * doctest::timeout(300)) {
  test_inclusive_scan<int, std::multiplies<int>>(4);
}

TEST_CASE("InclusiveScan.int*.8threads" * doctest::timeout(300)) {
  test_inclusive_scan<int, std::multiplies<int>>(8);
}

TEST_CASE("InclusiveScan.int*.12threads" * doctest::timeout(300)) {
  test_inclusive_scan<int, std::multiplies<int>>(12);
}

// string data type
TEST_CASE("InclusiveScan.string.1thread" * doctest::timeout(300)) {
  test_inclusive_scan<std::string, std::plus<std::string>>(1);
}

TEST_CASE("InclusiveScan.string.2threads" * doctest::timeout(300)) {
  test_inclusive_scan<std::string, std::plus<std::string>>(2);
}

TEST_CASE("InclusiveScan.string.3threads" * doctest::timeout(300)) {
  test_inclusive_scan<std::string, std::plus<std::string>>(3);
}

TEST_CASE("InclusiveScan.string.4threads" * doctest::timeout(300)) {
  test_inclusive_scan<std::string, std::plus<std::string>>(4);
}

TEST_CASE("InclusiveScan.string.8threads" * doctest::timeout(300)) {
  test_inclusive_scan<std::string, std::plus<std::string>>(8);
}

TEST_CASE("InclusiveScan.string.12threads" * doctest::timeout(300)) {
  test_inclusive_scan<std::string, std::plus<std::string>>(12);
}

// --------------------------------------------------------
// Testcase: inclusive_scan with initial value
// --------------------------------------------------------

template <typename T, typename B>
void test_initialized_inclusive_scan(unsigned W) {
  
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

    T init;
      
    if constexpr(std::is_same_v<T, std::string>) {
      init = std::to_string(::rand() % 10);
    }
    else {
      init = ::rand() % 10;
    }


    std::inclusive_scan(
      input.begin(), input.end(), golden.begin(), B(), init
    );
    
    // out-of-place
    auto task1 = taskflow.inclusive_scan(
      input.begin(), input.end(), output.begin(), B(), init
    );  
    
    // enable stateful capture
    auto alloc = taskflow.emplace([&](){
      sbeg = input.begin();
      send = input.end();
      dbeg = input.begin();
    });
    
    // in-place
    auto task2 = taskflow.inclusive_scan(
      std::ref(sbeg), std::ref(send), std::ref(dbeg), B(), init
    );

    task1.precede(alloc);
    alloc.precede(task2);

    executor.run(taskflow).wait();

    REQUIRE(input == golden);
    REQUIRE(output == golden);
  }
}

// int data type
TEST_CASE("Initialized.InclusiveScan.int+.1thread" * doctest::timeout(300)) {
  test_initialized_inclusive_scan<int, std::plus<int>>(1);
}

TEST_CASE("Initialized.InclusiveScan.int+.2threads" * doctest::timeout(300)) {
  test_initialized_inclusive_scan<int, std::plus<int>>(2);
}

TEST_CASE("Initialized.InclusiveScan.int+.3threads" * doctest::timeout(300)) {
  test_initialized_inclusive_scan<int, std::plus<int>>(3);
}

TEST_CASE("Initialized.InclusiveScan.int+.4threads" * doctest::timeout(300)) {
  test_initialized_inclusive_scan<int, std::plus<int>>(4);
}

TEST_CASE("Initialized.InclusiveScan.int+.8threads" * doctest::timeout(300)) {
  test_initialized_inclusive_scan<int, std::plus<int>>(8);
}

TEST_CASE("Initialized.InclusiveScan.int+.12threads" * doctest::timeout(300)) {
  test_initialized_inclusive_scan<int, std::plus<int>>(12);
}

// int data type
TEST_CASE("Initialized.InclusiveScan.int*.1thread" * doctest::timeout(300)) {
  test_initialized_inclusive_scan<int, std::multiplies<int>>(1);
}

TEST_CASE("Initialized.InclusiveScan.int*.2threads" * doctest::timeout(300)) {
  test_initialized_inclusive_scan<int, std::multiplies<int>>(2);
}

TEST_CASE("Initialized.InclusiveScan.int*.3threads" * doctest::timeout(300)) {
  test_initialized_inclusive_scan<int, std::multiplies<int>>(3);
}

TEST_CASE("Initialized.InclusiveScan.int*.4threads" * doctest::timeout(300)) {
  test_initialized_inclusive_scan<int, std::multiplies<int>>(4);
}

TEST_CASE("Initialized.InclusiveScan.int*.8threads" * doctest::timeout(300)) {
  test_initialized_inclusive_scan<int, std::multiplies<int>>(8);
}

TEST_CASE("Initialized.InclusiveScan.int*.12threads" * doctest::timeout(300)) {
  test_initialized_inclusive_scan<int, std::multiplies<int>>(12);
}

// --------------------------------------------------------
// Testcase: exclusive_scan
// --------------------------------------------------------

template <typename T, typename B>
void test_exclusive_scan(unsigned W) {
  
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
    
    T init;
    if constexpr(std::is_same_v<T, std::string>) {
      init = std::to_string(::rand() % 10);
    }
    else {
      init = ::rand() % 10;
    }

    std::exclusive_scan(
      input.begin(), input.end(), golden.begin(), init, B()
    );
    
    // out-of-place
    auto task1 = taskflow.exclusive_scan(
      input.begin(), input.end(), output.begin(), init, B()
    );  
    
    // enable stateful capture
    auto alloc = taskflow.emplace([&](){
      sbeg = input.begin();
      send = input.end();
      dbeg = input.begin();
    });
    
    // in-place
    auto task2 = taskflow.exclusive_scan(
      std::ref(sbeg), std::ref(send), std::ref(dbeg), init, B()
    );

    task1.precede(alloc);
    alloc.precede(task2);

    executor.run(taskflow).wait();

    REQUIRE(input == golden);
    REQUIRE(output == golden);
  }
}

// int data type
TEST_CASE("ExclusiveScan.int+.1thread" * doctest::timeout(300)) {
  test_exclusive_scan<int, std::plus<int>>(1);
}

TEST_CASE("ExclusiveScan.int+.2threads" * doctest::timeout(300)) {
  test_exclusive_scan<int, std::plus<int>>(2);
}

TEST_CASE("ExclusiveScan.int+.3threads" * doctest::timeout(300)) {
  test_exclusive_scan<int, std::plus<int>>(3);
}

TEST_CASE("ExclusiveScan.int+.4threads" * doctest::timeout(300)) {
  test_exclusive_scan<int, std::plus<int>>(4);
}

TEST_CASE("ExclusiveScan.int+.8threads" * doctest::timeout(300)) {
  test_exclusive_scan<int, std::plus<int>>(8);
}

TEST_CASE("ExclusiveScan.int+.12threads" * doctest::timeout(300)) {
  test_exclusive_scan<int, std::plus<int>>(12);
}

// int data type
TEST_CASE("ExclusiveScan.int*.1thread" * doctest::timeout(300)) {
  test_exclusive_scan<int, std::multiplies<int>>(1);
}

TEST_CASE("ExclusiveScan.int*.2threads" * doctest::timeout(300)) {
  test_exclusive_scan<int, std::multiplies<int>>(2);
}

TEST_CASE("ExclusiveScan.int*.3threads" * doctest::timeout(300)) {
  test_exclusive_scan<int, std::multiplies<int>>(3);
}

TEST_CASE("ExclusiveScan.int*.4threads" * doctest::timeout(300)) {
  test_exclusive_scan<int, std::multiplies<int>>(4);
}

TEST_CASE("ExclusiveScan.int*.8threads" * doctest::timeout(300)) {
  test_exclusive_scan<int, std::multiplies<int>>(8);
}

TEST_CASE("ExclusiveScan.int*.12threads" * doctest::timeout(300)) {
  test_exclusive_scan<int, std::multiplies<int>>(12);
}

// string data type
TEST_CASE("ExclusiveScan.string.1thread" * doctest::timeout(300)) {
  test_exclusive_scan<std::string, std::plus<std::string>>(1);
}

TEST_CASE("ExclusiveScan.string.2threads" * doctest::timeout(300)) {
  test_exclusive_scan<std::string, std::plus<std::string>>(2);
}

TEST_CASE("ExclusiveScan.string.3threads" * doctest::timeout(300)) {
  test_exclusive_scan<std::string, std::plus<std::string>>(3);
}

TEST_CASE("ExclusiveScan.string.4threads" * doctest::timeout(300)) {
  test_exclusive_scan<std::string, std::plus<std::string>>(4);
}

TEST_CASE("ExclusiveScan.string.8threads" * doctest::timeout(300)) {
  test_exclusive_scan<std::string, std::plus<std::string>>(8);
}

TEST_CASE("ExclusiveScan.string.12threads" * doctest::timeout(300)) {
  test_exclusive_scan<std::string, std::plus<std::string>>(12);
}

// --------------------------------------------------------
// Testcase: transform_inclusive_scan
// --------------------------------------------------------

template <typename T, typename B>
void test_transform_inclusive_scan(unsigned W) {
  
  tf::Executor executor(W);
  tf::Taskflow taskflow;

  size_t limit = std::is_same_v<T, std::string> ? 2500 : 250000;
  
  for(size_t n=0; n<=limit; n=n*2+1) {

    taskflow.clear();
    
    std::vector<T> input(n), output(n), golden(n); 
    typename std::vector<T>::iterator sbeg, send, dbeg;

    for(size_t i=0; i<n; i++) {
      input[i] = ::rand() % 10;
    }

    std::transform_inclusive_scan(
      input.begin(), input.end(), golden.begin(), B(),
      [](auto item){ return item*2; }
    );
    
    // out-of-place
    auto task1 = taskflow.transform_inclusive_scan(
      input.begin(), input.end(), output.begin(), B(),
      [](auto item){ return item*2; }
    );  
    
    // enable stateful capture
    auto alloc = taskflow.emplace([&](){
      sbeg = input.begin();
      send = input.end();
      dbeg = input.begin();
    });
    
    // in-place
    auto task2 = taskflow.transform_inclusive_scan(
      std::ref(sbeg), std::ref(send), std::ref(dbeg), B(),
      [](auto item){ return item*2; }
    );

    task1.precede(alloc);
    alloc.precede(task2);

    executor.run(taskflow).wait();

    REQUIRE(input == golden);
    REQUIRE(output == golden);
  }
}

// int data type
TEST_CASE("TransformedInclusiveScan.int+.1thread" * doctest::timeout(300)) {
  test_transform_inclusive_scan<int, std::plus<int>>(1);
}

TEST_CASE("TransformedInclusiveScan.int+.2threads" * doctest::timeout(300)) {
  test_transform_inclusive_scan<int, std::plus<int>>(2);
}

TEST_CASE("TransformedInclusiveScan.int+.3threads" * doctest::timeout(300)) {
  test_transform_inclusive_scan<int, std::plus<int>>(3);
}

TEST_CASE("TransformedInclusiveScan.int+.4threads" * doctest::timeout(300)) {
  test_transform_inclusive_scan<int, std::plus<int>>(4);
}

TEST_CASE("TransformedInclusiveScan.int+.8threads" * doctest::timeout(300)) {
  test_transform_inclusive_scan<int, std::plus<int>>(8);
}

TEST_CASE("TransformedInclusiveScan.int+.12threads" * doctest::timeout(300)) {
  test_transform_inclusive_scan<int, std::plus<int>>(12);
}

// int data type
TEST_CASE("TransformedInclusiveScan.int*.1thread" * doctest::timeout(300)) {
  test_transform_inclusive_scan<int, std::multiplies<int>>(1);
}

TEST_CASE("TransformedInclusiveScan.int*.2threads" * doctest::timeout(300)) {
  test_transform_inclusive_scan<int, std::multiplies<int>>(2);
}

TEST_CASE("TransformedInclusiveScan.int*.3threads" * doctest::timeout(300)) {
  test_transform_inclusive_scan<int, std::multiplies<int>>(3);
}

TEST_CASE("TransformedInclusiveScan.int*.4threads" * doctest::timeout(300)) {
  test_transform_inclusive_scan<int, std::multiplies<int>>(4);
}

TEST_CASE("TransformedInclusiveScan.int*.8threads" * doctest::timeout(300)) {
  test_transform_inclusive_scan<int, std::multiplies<int>>(8);
}

TEST_CASE("TransformedInclusiveScan.int*.12threads" * doctest::timeout(300)) {
  test_transform_inclusive_scan<int, std::multiplies<int>>(12);
}

// --------------------------------------------------------
// Testcase: initialized_transform_inclusive_scan
// --------------------------------------------------------

template <typename T, typename B>
void test_initialized_transform_inclusive_scan(unsigned W) {
  
  tf::Executor executor(W);
  tf::Taskflow taskflow;

  size_t limit = std::is_same_v<T, std::string> ? 2500 : 250000;
  
  for(size_t n=0; n<=limit; n=n*2+1) {

    taskflow.clear();
    
    std::vector<T> input(n), output(n), golden(n); 
    typename std::vector<T>::iterator sbeg, send, dbeg;

    for(size_t i=0; i<n; i++) {
      input[i] = ::rand() % 10;
    }

    T init = ::rand() % 10;

    std::transform_inclusive_scan(
      input.begin(), input.end(), golden.begin(), B(),
      [](auto item){ return item*2; },
      init
    );
    
    // out-of-place
    auto task1 = taskflow.transform_inclusive_scan(
      input.begin(), input.end(), output.begin(), B(),
      [](auto item){ return item*2; },
      init
    );  
    
    // enable stateful capture
    auto alloc = taskflow.emplace([&](){
      sbeg = input.begin();
      send = input.end();
      dbeg = input.begin();
    });
    
    // in-place
    auto task2 = taskflow.transform_inclusive_scan(
      std::ref(sbeg), std::ref(send), std::ref(dbeg), B(),
      [](auto item){ return item*2; },
      init
    );

    task1.precede(alloc);
    alloc.precede(task2);

    executor.run(taskflow).wait();

    REQUIRE(input == golden);
    REQUIRE(output == golden);
  }
}

// int data type
TEST_CASE("Initialized.TransformedInclusiveScan.int+.1thread" * doctest::timeout(300)) {
  test_initialized_transform_inclusive_scan<int, std::plus<int>>(1);
}

TEST_CASE("Initialized.TransformedInclusiveScan.int+.2threads" * doctest::timeout(300)) {
  test_initialized_transform_inclusive_scan<int, std::plus<int>>(2);
}

TEST_CASE("Initialized.TransformedInclusiveScan.int+.3threads" * doctest::timeout(300)) {
  test_initialized_transform_inclusive_scan<int, std::plus<int>>(3);
}

TEST_CASE("Initialized.TransformedInclusiveScan.int+.4threads" * doctest::timeout(300)) {
  test_initialized_transform_inclusive_scan<int, std::plus<int>>(4);
}

TEST_CASE("Initialized.TransformedInclusiveScan.int+.8threads" * doctest::timeout(300)) {
  test_initialized_transform_inclusive_scan<int, std::plus<int>>(8);
}

TEST_CASE("Initialized.TransformedInclusiveScan.int+.12threads" * doctest::timeout(300)) {
  test_initialized_transform_inclusive_scan<int, std::plus<int>>(12);
}

// int data type
TEST_CASE("Initialized.TransformedInclusiveScan.int*.1thread" * doctest::timeout(300)) {
  test_initialized_transform_inclusive_scan<int, std::multiplies<int>>(1);
}

TEST_CASE("Initialized.TransformedInclusiveScan.int*.2threads" * doctest::timeout(300)) {
  test_initialized_transform_inclusive_scan<int, std::multiplies<int>>(2);
}

TEST_CASE("Initialized.TransformedInclusiveScan.int*.3threads" * doctest::timeout(300)) {
  test_initialized_transform_inclusive_scan<int, std::multiplies<int>>(3);
}

TEST_CASE("Initialized.TransformedInclusiveScan.int*.4threads" * doctest::timeout(300)) {
  test_initialized_transform_inclusive_scan<int, std::multiplies<int>>(4);
}

TEST_CASE("Initialized.TransformedInclusiveScan.int*.8threads" * doctest::timeout(300)) {
  test_initialized_transform_inclusive_scan<int, std::multiplies<int>>(8);
}

TEST_CASE("Initialized.TransformedInclusiveScan.int*.12threads" * doctest::timeout(300)) {
  test_initialized_transform_inclusive_scan<int, std::multiplies<int>>(12);
}

// --------------------------------------------------------
// Testcase: transform_exclusive_scan
// --------------------------------------------------------

template <typename T, typename B>
void test_transform_exclusive_scan(unsigned W) {
  
  tf::Executor executor(W);
  tf::Taskflow taskflow;

  size_t limit = std::is_same_v<T, std::string> ? 2500 : 250000;
  
  for(size_t n=0; n<=limit; n=n*2+1) {

    taskflow.clear();
    
    std::vector<T> input(n), output(n), golden(n); 
    typename std::vector<T>::iterator sbeg, send, dbeg;

    for(size_t i=0; i<n; i++) {
      input[i] = ::rand() % 10;
    }
    
    T init = ::rand() % 10;

    std::transform_exclusive_scan(
      input.begin(), input.end(), golden.begin(), init, B(),
      [](auto item){ return 2*item; }
    );
    
    // out-of-place
    auto task1 = taskflow.transform_exclusive_scan(
      input.begin(), input.end(), output.begin(), init, B(),
      [](auto item){ return 2*item; }
    );  
    
    // enable stateful capture
    auto alloc = taskflow.emplace([&](){
      sbeg = input.begin();
      send = input.end();
      dbeg = input.begin();
    });
    
    // in-place
    auto task2 = taskflow.transform_exclusive_scan(
      std::ref(sbeg), std::ref(send), std::ref(dbeg), init, B(),
      [](auto item){ return 2*item; }
    );

    task1.precede(alloc);
    alloc.precede(task2);

    executor.run(taskflow).wait();

    REQUIRE(input == golden);
    REQUIRE(output == golden);
  }
}

// int data type
TEST_CASE("TransformedExclusiveScan.int+.1thread" * doctest::timeout(300)) {
  test_transform_exclusive_scan<int, std::plus<int>>(1);
}

TEST_CASE("TransformedExclusiveScan.int+.2threads" * doctest::timeout(300)) {
  test_transform_exclusive_scan<int, std::plus<int>>(2);
}

TEST_CASE("TransformedExclusiveScan.int+.3threads" * doctest::timeout(300)) {
  test_transform_exclusive_scan<int, std::plus<int>>(3);
}

TEST_CASE("TransformedExclusiveScan.int+.4threads" * doctest::timeout(300)) {
  test_transform_exclusive_scan<int, std::plus<int>>(4);
}

TEST_CASE("TransformedExclusiveScan.int+.8threads" * doctest::timeout(300)) {
  test_transform_exclusive_scan<int, std::plus<int>>(8);
}

TEST_CASE("TransformedExclusiveScan.int+.12threads" * doctest::timeout(300)) {
  test_transform_exclusive_scan<int, std::plus<int>>(12);
}

// int data type
TEST_CASE("TransformedExclusiveScan.int*.1thread" * doctest::timeout(300)) {
  test_transform_exclusive_scan<int, std::multiplies<int>>(1);
}

TEST_CASE("TransformedExclusiveScan.int*.2threads" * doctest::timeout(300)) {
  test_transform_exclusive_scan<int, std::multiplies<int>>(2);
}

TEST_CASE("TransformedExclusiveScan.int*.3threads" * doctest::timeout(300)) {
  test_transform_exclusive_scan<int, std::multiplies<int>>(3);
}

TEST_CASE("TransformedExclusiveScan.int*.4threads" * doctest::timeout(300)) {
  test_transform_exclusive_scan<int, std::multiplies<int>>(4);
}

TEST_CASE("TransformedExclusiveScan.int*.8threads" * doctest::timeout(300)) {
  test_transform_exclusive_scan<int, std::multiplies<int>>(8);
}

TEST_CASE("TransformedExclusiveScan.int*.12threads" * doctest::timeout(300)) {
  test_transform_exclusive_scan<int, std::multiplies<int>>(12);
}
